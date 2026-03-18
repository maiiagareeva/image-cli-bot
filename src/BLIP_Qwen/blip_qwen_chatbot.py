from __future__ import annotations

import os
import json
import argparse
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Blip2Processor,
)
from peft import PeftModel

from src.BLIP_Qwen.cross_model.projector import MLPProjector
from src.BLIP_Qwen.cross_model.query_mix import QueryMixerBlock
from src.BLIP_Qwen.BLIP import BLIP2Model

BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
CKPT_DIR = "./qwen3-1.7B-greenhouse-ngld-2-2"
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

        # output = self.qwen.generate(
        #     input_ids=input_ids,
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=full_attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     do_sample=do_sample,
        #     temperature=temperature if do_sample else None,
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     use_cache=True,
        #     repetition_penalty=repetition_penalty,
        #     no_repeat_ngram_size=no_repeat_ngram_size,
        # )
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
            
        print("input_ids shape:", input_ids.shape)
        print("inputs_embeds shape:", inputs_embeds.shape)
        print("attention_mask shape:", full_attention_mask.shape)
        print("output shape:", output.shape)
        
        # return output, P, T
        return output

def read_model_meta(ckpt_dir):
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
    base_model_id,
    lora_path,
    device,
    dtype=torch.float16,
    merge_lora=True,
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
        qwen=qwen.to(device)
        qwen.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return qwen, tokenizer


def _extract_json(text: str):
    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(text[l:r+1])
    except Exception:
        return None


@torch.no_grad()
def infer_once(
    image_path,
    prompt,
    model: QwenWithBLIPPrefix,
    tokenizer,
    image_processor,
    device,
    max_new_tokens=256,
):
    img = Image.open(image_path).convert("RGB")

    pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)

    output= model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )

    text = tokenizer.decode(output[0,input_ids.shape[1]:], skip_special_tokens=True).strip()

    out = _extract_json(text)
    if out is not None and "disease" in out:
        return out["disease"], text

    return text, text


PROMPT = (
    "You are a grape leaf disease diagnosis assistant. "
    "Analyze the image and output a diagnosis in the following JSON schema."
)


def build_infer_model(ckpt_dir,device):
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

    projector_state = torch.load(projector_path, map_location=device)
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

        query_mixer_state = torch.load(query_mixer_path, map_location=device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--prompt", default=None, help="Optional custom prompt")
    ap.add_argument("--ckpt_dir", default=CKPT_DIR, help="Checkpoint directory")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vlm, tokenizer, image_processor = build_infer_model(args.ckpt_dir,device)

    print(f"Using image: {args.image}")
    print(f"Using checkpoint dir: {args.ckpt_dir}")
    print("Type 'exit' to quit.\n")

    while True:
        user_in = input("press enter to run diagnosis: ").strip()
        if user_in.lower() == "exit":
            break

        prompt = args.prompt if args.prompt is not None else PROMPT

        answer, raw = infer_once(
            image_path=args.image,
            prompt=prompt,
            model=vlm,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            max_new_tokens=256,
        )

        print("\nVLM assistant:\n", answer, "\n")
        print("\nVLM assistant raw generation:\n", raw, "\n")


if __name__ == "__main__":
    main()
