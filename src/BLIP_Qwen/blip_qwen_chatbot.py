from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Blip2Model,
)
from peft import PeftModel

class BLIP2Model(nn.Module):
    def __init__(self,blip2_model_id,device,dtype= torch.float16,freeze= True,):
        super().__init__()
        self.device = device

        self.blip2 = Blip2Model.from_pretrained(
            blip2_model_id,
            torch_dtype=dtype,
        ).to(device)
        self.blip2.eval()

        if freeze:
            for p in self.blip2.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values):
        vision_outputs = self.blip2.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state

        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        B = image_embeds.size(0)
        query_tokens = self.blip2.query_tokens.expand(B, -1, -1)

        qformer_outputs = self.blip2.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        return qformer_outputs.last_hidden_state

class QwenWithBLIPPrefix(nn.Module):
    def __init__(self,qwen,blip2model,projector,):
        super().__init__()
        self.qwen = qwen
        self.blip = blip2model
        self.projector = projector

    @torch.no_grad()
    def generate(self,pixel_values,input_ids,attention_mask,tokenizer,max_new_tokens,do_sample= True,temperature= 0.1,):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        query_embeds = self.blip(pixel_values)
        prefix_embeds = self.projector(query_embeds).to(dtype=qwen_dtype)
        B, P, _ = prefix_embeds.shape

        token_embeds = self.qwen.get_input_embeddings()(input_ids).to(dtype=qwen_dtype)

        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

def load_qwen_with_lora(base_model_id,lora_path,device_map= "auto",dtype= torch.float16,merge_lora= True):
    qwen = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    qwen.eval()

    if lora_path:
        qwen = PeftModel.from_pretrained(qwen, lora_path)
        if merge_lora:
            qwen = qwen.merge_and_unload()
        qwen.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return qwen, tokenizer


@torch.no_grad()
def infer_once(image_path,prompt,model: QwenWithBLIPPrefix,tokenizer,image_processor,device,max_new_tokens= 256,):
    img = Image.open(image_path).convert("RGB")

    pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)

    out_ids = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
    )

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()
    return text

BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_PATH = "qwen3-1.7B-ngld-lora"
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
PROJECTOR_PATH = "./qwen3-1.7B-ngld-lora-1/projector.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qwen, tokenizer = load_qwen_with_lora(
        base_model_id=BASE_MODEL_ID,
        lora_path=LORA_PATH,
        device_map="auto",
        dtype=torch.float16,
        merge_lora=True,
    )

    image_processor = AutoProcessor.from_pretrained(BLIP2_MODEL_ID)
    blip2model = HFBLIP2Bridge(
        blip2_model_id=BLIP2_MODEL_ID,
        device=device,
        dtype=torch.float16,
        freeze=True,
    )

    d_qformer = blip2model.blip2.qformer.config.hidden_size
    d_qwen = qwen.config.hidden_size
    projector = nn.Linear(d_qformer, d_qwen).to(device, dtype=torch.float16)

    state_dict = torch.load(PROJECTOR_PATH, map_location=device)
    projector.load_state_dict(state_dict, strict=True)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False

    vlm = QwenWithBLIPPrefix(
        qwen=qwen,
        blip2model=blip2model,
        projector=projector,
    ).to(device)
    vlm.eval()

    print(f"Using image: {args.image}")
    print("Type 'exit' to quit.\n")

    while True:
        user_in = input("User: ").strip()
        if not user_in:
            continue
        if user_in.lower() == "exit":
            break

        answer = infer_once(
            image_path=args.image,
            prompt=user_in,
            model=vlm,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            max_new_tokens=256,
        )
        print("\nVLM assistant:\n", answer, "\n")

if __name__ == "__main__":
    main()
