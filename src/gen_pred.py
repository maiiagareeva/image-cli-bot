from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Blip2Processor,
)
from peft import PeftModel

from src.BLIP_Qwen.cross_model.projector import MLPProjector
from src.BLIP_Qwen.cross_model.query_mix import QueryMixerBlock
from src.BLIP_Qwen.BLIP import BLIP2Model
from src.dataset import VLMDataset
from src.BLIP_Qwen.args import parse_yaml


BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"


class QwenWithBLIPPrefix(nn.Module):
    def __init__(self, qwen, blip, projector, query_mixer=None):
        super().__init__()
        self.qwen = qwen
        self.blip = blip
        self.projector = projector
        self.query_mixer = query_mixer

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    ):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        query_embeds = self.blip(pixel_values)

        if self.query_mixer is not None:
            mixer_dtype = next(self.query_mixer.parameters()).dtype
            query_embeds = query_embeds.to(dtype=mixer_dtype)
            query_embeds = self.query_mixer(query_embeds)

        projector_dtype = next(self.projector.parameters()).dtype
        query_embeds = query_embeds.to(dtype=projector_dtype)
        prefix_embeds = self.projector(query_embeds).to(dtype=qwen_dtype)

        B, P, _ = prefix_embeds.shape

        token_embeds = self.qwen.get_input_embeddings()(input_ids).to(dtype=qwen_dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        gen_kwargs = dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        if do_sample:
            gen_kwargs["temperature"] = temperature

        output = self.qwen.generate(**gen_kwargs)
        return output


def read_model_meta(ckpt_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(ckpt_dir, "model_meta.json")
    blip_txt_path = os.path.join(ckpt_dir, "blip2model.txt")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if os.path.exists(blip_txt_path):
        with open(blip_txt_path, "r", encoding="utf-8") as f:
            blip2_model_id = f.read().strip()
    else:
        blip2_model_id = meta.get("blip2_model", BLIP2_MODEL_ID)

    base_model_id = meta.get("base_model", BASE_MODEL_ID)

    return {
        "base_model": base_model_id,
        "blip2_model": blip2_model_id,
        "meta": meta,
    }


def load_qwen_with_lora(
    base_model_id: str,
    lora_path: str,
    device,
    dtype=torch.float16,
    merge_lora: bool = True,
):
    qwen = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    qwen.eval()

    if lora_path:
        qwen = PeftModel.from_pretrained(qwen, lora_path)
        if merge_lora:
            qwen = qwen.merge_and_unload()
        qwen = qwen.to(device)
        qwen.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return qwen, tokenizer


def safe_parse_json(text: str):
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(text[l:r + 1])
    except Exception:
        return None


@torch.no_grad()
def infer_once_from_pil(
    image: Image.Image,
    prompt: str,
    model: QwenWithBLIPPrefix,
    tokenizer,
    image_processor,
    device,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.1,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
):
    image = image.convert("RGB")

    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)

    output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    parsed = safe_parse_json(text)
    return text, parsed


PROMPT = (
    "You are a grape leaf disease diagnosis assistant. "
    "Analyze the image and output a diagnosis in the following JSON schema."
)


def build_infer_model(ckpt_dir: str, device):
    paths = read_model_meta(ckpt_dir)
    base_model_id = paths["base_model"]
    blip2_model_id = paths["blip2_model"]

    projector_path = os.path.join(ckpt_dir, "projector.pt")
    query_mixer_path = os.path.join(ckpt_dir, "query_mixer.pt")

    qwen, tokenizer = load_qwen_with_lora(
        base_model_id=base_model_id,
        lora_path=ckpt_dir,
        device=device,
        dtype=torch.float16,
        merge_lora=True,
    )

    image_processor = Blip2Processor.from_pretrained(blip2_model_id)

    blip = BLIP2Model(
        blip2_model_id=blip2_model_id,
        device=str(device),
        dtype=torch.float16,
    )
    blip.eval()

    d_qformer = blip.qformer_dim
    d_qwen = qwen.config.hidden_size
    qwen_dtype = qwen.get_input_embeddings().weight.dtype

    projector = MLPProjector(
        in_dim=d_qformer,
        out_dim=d_qwen,
        hidden_dim=2 * d_qwen,
        use_residual=True,
        dropout=0.0,
    ).to(device, dtype=qwen_dtype)

    projector_state = torch.load(projector_path, map_location=device, weights_only=True)
    projector.load_state_dict(projector_state, strict=True)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False

    query_mixer: Optional[nn.Module] = None
    if os.path.exists(query_mixer_path):
        query_mixer = QueryMixerBlock(
            dim=d_qformer,
            num_heads=8,
            mlp_ratio=2.0,
            dropout=0.1,
        ).to(device, dtype=qwen_dtype)

        query_mixer_state = torch.load(query_mixer_path, map_location=device, weights_only=True)
        query_mixer.load_state_dict(query_mixer_state, strict=True)
        query_mixer.eval()
        for p in query_mixer.parameters():
            p.requires_grad = False

    vlm = QwenWithBLIPPrefix(
        qwen=qwen,
        blip=blip,
        projector=projector,
        query_mixer=query_mixer,
    )
    vlm.eval()

    return vlm, tokenizer, image_processor


def get_split_dataset(cfg, split: str):
    datasets = VLMDataset(cfg.data)
    split = split.lower()

    if split == "train":
        return datasets.train_ds

    if split in ["val", "eval", "validation"]:
        return datasets.eval_ds

    if split == "test":
        if hasattr(datasets, "test_ds"):
            return datasets.test_ds
        raise AttributeError(
            "VLMDataset has no attribute 'test_ds'. "
            "You need to add test_ds to your dataset wrapper, or evaluate on val first."
        )

    raise ValueError(f"Unknown split: {split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/blip_qwen_train.yaml")
    ap.add_argument("--ckpt_dir", required=True, help="Checkpoint/model directory")
    ap.add_argument("--split", default="val", choices=["train", "val", "eval", "test"])
    ap.add_argument("--output", required=True, help="Path to save predictions jsonl")
    ap.add_argument("--prompt", default=None, help="Optional custom prompt")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = ap.parse_args()

    cfg = parse_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, image_processor = build_infer_model(args.ckpt_dir, device)
    ds = get_split_dataset(cfg, args.split)

    prompt = args.prompt if args.prompt is not None else PROMPT

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(len(ds)), desc=f"Generating {args.split} predictions"):
            sample = ds[idx]

            image = sample["image"]
            if not isinstance(image, Image.Image):
                # if your dataset returns a dict like {"path": ...}
                if isinstance(image, dict) and "path" in image:
                    image = Image.open(image["path"]).convert("RGB")
                else:
                    raise TypeError(f"Unsupported image format at idx={idx}: {type(image)}")

            reference_text = sample["answer"]
            prediction_text, prediction_json = infer_once_from_pil(
                image=image,
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            record = {
                "id": sample.get("sample_id", idx),
                "split": args.split,
                "prompt": prompt,
                "reference_text": reference_text,
                "prediction_text": prediction_text,
                "reference_json": safe_parse_json(reference_text),
                "prediction_json": prediction_json,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved predictions to {output_path}")


if __name__ == "__main__":
    main()