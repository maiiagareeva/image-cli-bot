from __future__ import annotations
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.BLIP_Qwen.BLIP import BLIP2Model
from src.utils import set_requires_grad

class QwenWithBLIPPrefix(nn.Module):
    def __init__(self, qwen, blip, projector):
        super().__init__()
        self.qwen = qwen
        self.blip = blip
        self.projector = projector

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        with torch.no_grad():
            query_embeds = self.blip(pixel_values)
        prefix_embeds = self.projector(query_embeds).to(dtype=qwen_dtype)
        P = prefix_embeds.size(1)

        token_embeds = self.qwen.get_input_embeddings()(input_ids).to(dtype=qwen_dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        B = input_ids.size(0)
        prefix_attention = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        prefix_labels = torch.full((B, P), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=256, do_sample=False, temperature=1.0, **gen_kwargs):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        query_embeds = self.blip(pixel_values)
        prefix_embeds = self.projector(query_embeds).to(dtype=qwen_dtype)
        P = prefix_embeds.size(1)

        token_embeds = self.qwen.get_input_embeddings()(input_ids).to(dtype=qwen_dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        B = input_ids.size(0)
        prefix_attention = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        return self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=gen_kwargs.get("pad_token_id", None),
            eos_token_id=gen_kwargs.get("eos_token_id", None),
        )

def build_model(model_cfg, device,train_cfg):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    qwen = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        quantization_config=quant_cfg,
        device_map={"": 0},
        trust_remote_code=True,
    )
    qwen.config.use_cache = False
    qwen = prepare_model_for_kbit_training(qwen)

    lora = model_cfg.lora
    peft_cfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=lora.target_modules,
    )
    qwen = get_peft_model(qwen, peft_cfg)

    blip = BLIP2Model(model_cfg.blip2_model, 
                      device=device, 
                      dtype=torch.float16
                      )

    d_qformer = blip.qformer_dim
    d_qwen = qwen.config.hidden_size

    qwen_dtype = qwen.get_input_embeddings().weight.dtype
    projector = nn.Linear(d_qformer, d_qwen).to(device, dtype=qwen_dtype)

    # todo: two stage training
    set_requires_grad(qwen, True)
    set_requires_grad(projector, True)

    return QwenWithBLIPPrefix(qwen=qwen, blip=blip, projector=projector)
