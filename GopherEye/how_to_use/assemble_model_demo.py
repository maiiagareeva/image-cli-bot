from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


cur = Path(__file__).resolve()
PROJECT_ROOT = cur.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.BLIP_Qwen.BLIP import BLIP2Model
from src.BLIP_Qwen.blip2_support import build_blip2_image_processors
from src.BLIP_Qwen.cross_model.projector import MLPProjector
from src.BLIP_Qwen.model import QwenWithBLIPPrefix
from src.BLIP_Qwen.prediction_format import parse_plain_answer
from src.utils import ensure_pil_rgb


PROMPT = "Identify the grape leaf disease and describe the key visual signs in the image."
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_model_meta(ckpt_dir):
    meta_path = ckpt_dir / "model_meta.json"
    blip_txt_path = ckpt_dir / "blip2model.txt"

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if blip_txt_path.exists():
        blip2_model_id = blip_txt_path.read_text(encoding="utf-8").strip()
    else:
        blip2_model_id = meta.get("blip2_model", BLIP2_MODEL_ID)

    return {
        "base_model": meta.get("base_model", BASE_MODEL_ID),
        "blip2_model": blip2_model_id,
        "meta": meta,
    }


def load_qwen_with_lora(
    base_model_id: str,
    lora_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    qwen = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    qwen.eval()

    qwen = PeftModel.from_pretrained(qwen, str(lora_path))
    qwen = qwen.merge_and_unload()
    qwen = qwen.to(device)
    qwen.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return qwen, tokenizer


def resolve_qformer_stage_dir(ckpt_dir, meta):
    if (ckpt_dir / "stage1_meta.json").exists():
        return ckpt_dir

    raw_stage1_dir = meta.get("qformer_stage1_dir")
    if raw_stage1_dir:
        for candidate in [
            Path(raw_stage1_dir),
            PROJECT_ROOT / raw_stage1_dir,
            ckpt_dir.parent / raw_stage1_dir,
        ]:
            candidate = candidate.resolve()
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        "not find Qformer stage files "
    )


def build_model_from_parts(ckpt_dir, device):
    load_dtype = torch.float16 if device.type == "cuda" else torch.float32

    paths = read_model_meta(ckpt_dir)
    base_model_id = paths["base_model"]
    blip2_model_id = paths["blip2_model"]
    meta = paths["meta"]

    projector_path = ckpt_dir / "projector.pt"
    if not projector_path.exists():
        raise FileNotFoundError(f"Missing projector checkpoint: {projector_path}")

    qformer_stage_dir = resolve_qformer_stage_dir(ckpt_dir, meta)

    qwen, tokenizer = load_qwen_with_lora(
        base_model_id=base_model_id,
        lora_path=ckpt_dir,
        device=device,
        dtype=load_dtype,
    )

    _, image_processor, resolved_lavis_model_type = build_blip2_image_processors(
        blip2_model_id=blip2_model_id,
        lavis_model_type=meta.get("lavis_model_type"),
    )

    blip = BLIP2Model(
        blip2_model_id=blip2_model_id,
        device=str(device),
        dtype=load_dtype,
        qformer_stage1_dir=str(qformer_stage_dir),
        num_query_token=meta.get("num_query_token", meta.get("prefix_len", 32)),
        cross_attention_freq=meta.get("cross_attention_freq", 2),
        lavis_model_type=resolved_lavis_model_type,
        freeze_vision=True,
        freeze_qformer=True,
        train_query_tokens=meta.get("train_query_tokens", False),
    )
    blip.eval()

    projector = MLPProjector(
        in_dim=blip.qformer_dim,
        out_dim=qwen.config.hidden_size,
        hidden_dim=2 * qwen.config.hidden_size,
        use_residual=True,
        dropout=0.0,
    ).to(device, dtype=qwen.get_input_embeddings().weight.dtype)
    projector.load_state_dict(torch.load(projector_path, map_location=device), strict=True)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False

    model = QwenWithBLIPPrefix(qwen=qwen, blip=blip, projector=projector)
    model.eval()
    return model, tokenizer, image_processor


def process_single_image(image_processor, image):
    try:
        image_out = image_processor(images=[image], return_tensors="pt")
        if isinstance(image_out, dict):
            return image_out["pixel_values"]
        return image_out.pixel_values
    except TypeError:
        return image_processor(image).unsqueeze(0)


@torch.no_grad()
def infer_once(
    image,
    prompt,
    model,
    tokenizer,
    image_processor,
    device,
    max_new_tokens,
):
    image = ensure_pil_rgb(image)
    pixel_values = process_single_image(image_processor, image).to(device)

    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)

    output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.1,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

def run_inference(
    image_path,
    prompt,
    model,
    tokenizer,
    image_processor,
    device,
    max_new_tokens,
):
    image = Image.open(image_path).convert("RGB")
    raw_text = infer_once(
        image=image,
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    parsed = parse_plain_answer(raw_text)

    print("\nraw text")
    print(raw_text)
    print("\nparsed text")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="assemble the GopherEye MVP")
    parser.add_argument(
        "--ckpt_dir",
        default="qwen3-1.7B-greenhouse-ngld-faithful-stage2",
        help="stage2 checkpoint directory.",
    )
    parser.add_argument("--image", default=None, help="image path")
    parser.add_argument("--prompt", default=PROMPT, help="prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default=None, help="torch device")
    parser.add_argument("--no_infer", action="store_true", help="do not run image inference")
    args = parser.parse_args()

    ckpt_dir = resolve_path(args.ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, tokenizer, image_processor = build_model_from_parts(ckpt_dir, device)

    if args.no_infer:
        return

    if args.image is None:
        raise ValueError("pls pass --image path/to/grape_leaf.jpg, or use --no_infer.")

    image_path = resolve_path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    run_inference(
        image_path=image_path,
        prompt=args.prompt,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
