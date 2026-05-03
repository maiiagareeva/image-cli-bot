from __future__ import annotations
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.BLIP_Qwen.BLIP import BLIP2Model
from src.BLIP_Qwen.cross_model.projector import MLPProjector
from src.utils import set_requires_grad

class QwenWithBLIPPrefix(nn.Module):
    def __init__(self, qwen, blip, projector, use_weighted_loss=False):
        super().__init__()
        self.qwen = qwen
        self.blip = blip
        self.projector = projector
        self.query_mixer = None
        self.use_weighted_loss = use_weighted_loss

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                return self.qwen.gradient_checkpointing_enable()
            return self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        raise AttributeError("Inner qwen model does not support gradient checkpointing.")

    def gradient_checkpointing_disable(self):
        if hasattr(self.qwen, "gradient_checkpointing_disable"):
            return self.qwen.gradient_checkpointing_disable()
        raise AttributeError("Inner qwen model does not support gradient checkpointing.")

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        loss_weights = kwargs.get("loss_weights", None)

        query_embeds = self.blip(pixel_values)
        projector_dtype = next(self.projector.parameters()).dtype
        query_embeds = query_embeds.to(dtype=projector_dtype)
        prefix_embeds = self.projector(query_embeds).to(dtype=qwen_dtype)

        P = prefix_embeds.size(1)

        token_embeds = self.qwen.get_input_embeddings()(input_ids).to(dtype=qwen_dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        B = input_ids.size(0)
        prefix_attention = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        prefix_labels = torch.full((B, P), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        if (not self.use_weighted_loss) or (loss_weights is None):
            return self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                return_dict=True,
                use_cache=False,
            )

        prefix_weights = torch.zeros((B, P), dtype=loss_weights.dtype, device=device)
        full_weights = torch.cat([prefix_weights, loss_weights.to(device)], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits

        pre_logits = logits[:, :-1, :].contiguous()
        pre_labels = full_labels[:, 1:].contiguous()
        pre_weights = full_weights[:, 1:].contiguous()

        celoss = nn.CrossEntropyLoss(reduction="none")
        token_loss = celoss(
            pre_logits.view(-1, pre_logits.size(-1)),
            pre_labels.view(-1)
        ).view_as(pre_labels)

        valid_labels = (pre_labels != -100).float()
        weighted = token_loss * valid_labels * pre_weights
        denominator = (valid_labels * pre_weights).sum().clamp_min(1.0)
        outputs.loss = weighted.sum() / denominator
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        **gen_kwargs,
    ):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        query_embeds = self.blip(pixel_values)
        projector_dtype = next(self.projector.parameters()).dtype
        query_embeds = query_embeds.to(dtype=projector_dtype)
        prefix_embeds = self.projector(query_embeds).to(dtype=qwen_dtype)

        P = prefix_embeds.size(1)
        token_embeds = self.qwen.get_input_embeddings()(input_ids).to(dtype=qwen_dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        B = input_ids.size(0)
        prefix_attention = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        return self.qwen.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=gen_kwargs.get("pad_token_id", None),
            eos_token_id=gen_kwargs.get("eos_token_id", None),
            use_cache=True,
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.15),
            no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", 3),
        )

def build_model(model_cfg, device, train_cfg):
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("Stage2 Qwen training currently requires CUDA.")

    if resolved_device.index is None:
        device_map = {"": "cuda:0"}
    else:
        device_map = {"": f"cuda:{resolved_device.index}"}

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_qwen = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        quantization_config=quant_cfg,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_qwen.config.use_cache = False
    base_qwen = prepare_model_for_kbit_training(base_qwen)

    lora = model_cfg.lora
    peft_cfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=lora.target_modules,
    )
    qwen = get_peft_model(base_qwen, peft_cfg)

    blip = BLIP2Model(
        blip2_model_id=model_cfg.blip2_model,
        device=device,
        dtype=torch.float16,
        qformer_stage1_dir=model_cfg.qformer_stage1_dir,
        num_query_token=model_cfg.num_query_token,
        cross_attention_freq=model_cfg.cross_attention_freq,
        lavis_model_type=model_cfg.lavis_model_type,
        freeze_vision=True,
        freeze_qformer=False,
        train_query_tokens=model_cfg.train_query_tokens,
    )

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

    set_requires_grad(projector, True)

    model = QwenWithBLIPPrefix(
        qwen=qwen,
        blip=blip,
        projector=projector,
        use_weighted_loss=train_cfg.use_weighted_loss,
    )
    return model
