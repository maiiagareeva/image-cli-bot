from __future__ import annotations
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.BLIP_Qwen.BLIP import BLIP2Model
from src.utils import set_requires_grad
from src.BLIP_Qwen.cross_model.projector import MLPProjector
from src.BLIP_Qwen.cross_model.query_mix import QueryMixerBlock
import os

class QwenWithBLIPPrefix(nn.Module):
    def __init__(self, qwen, blip, projector,query_mixer=None,use_weighted_loss=False):
        super().__init__()
        self.qwen = qwen
        self.blip = blip
        self.projector = projector
        self.query_mixer=query_mixer
        self.use_weighted_loss=use_weighted_loss

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        device = input_ids.device
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype

        loss_weights=kwargs.get("loss_weights",None)
        disease_id=kwargs.get("disease_id",None)

        with torch.no_grad():
            query_embeds = self.blip(pixel_values)

        if self.query_mixer is not None:
            mixer_dtype = next(self.query_mixer.parameters()).dtype
            query_embeds = query_embeds.to(dtype=mixer_dtype)
            query_embeds = self.query_mixer(query_embeds)

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

        # unweighted
        if( not self.use_weighted_loss) or (loss_weights is None):
             return self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
                return_dict=True,
                use_cache=False,
            )

        # weigted
        prefix_weights=torch.zeros((B,P),dtype=loss_weights.dtype,device=device)
        full_weights=torch.cat([prefix_weights,loss_weights.to(device)],dim=1)
        outputs=self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                # forward without labels
                return_dict=True,
                use_cache=False,
            )
        logits=outputs.logits

        # predict token t using logits @t-1
        pre_logits=logits[:,:-1,:].contiguous()
        # logits[:, t] predict labels[:, t+1]
        pre_labels=full_labels[:,1:].contiguous()
        pre_weights=full_weights[:,1:].contiguous()

        CEloss=nn.CrossEntropyLoss(reduction="none")
        token_loss=CEloss(
            pre_logits.view(-1,pre_logits.size(-1)), #(B*L, V)
            pre_labels.view(-1) # (B*L,)
        ).view_as(pre_labels)

        valid_labels=(pre_labels!=-100).float()
        weighted=token_loss*valid_labels*pre_weights

        denominator=(valid_labels*pre_weights).sum().clamp_min(1.0)
        loss=weighted.sum()/denominator

        outputs.loss=loss
        return outputs


    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=256, do_sample=False, temperature=1.0, **gen_kwargs):
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

# def maybe_load_stage1_weights(model, model_cfg, device):
#     stage1_dir = getattr(model_cfg, "init_from_stage1", None)

#     if not stage1_dir:
#         print("[INFO] No stage1 init dir provided. Train from fresh initialization.")
#         return model

#     print(f"[INFO] Loading stage1 bridge weights from: {stage1_dir}")
#     print("[INFO] Route: generic bridge-only -> task-specific bridge+LoRA")
#     print("[INFO] Old stage1 LoRA adapter will NOT be loaded.")

#     projector_path = os.path.join(stage1_dir, "projector.pt")
#     if os.path.exists(projector_path):
#         sd = torch.load(projector_path, map_location=device, weights_only=True)
#         model.projector.load_state_dict(sd)
#         print(f"[INFO] Loaded projector from: {projector_path}")
#     else:
#         print(f"[WARN] projector.pt not found: {projector_path}")

#     query_mixer_path = os.path.join(stage1_dir, "query_mixer.pt")
#     if model.query_mixer is not None:
#         if os.path.exists(query_mixer_path):
#             sd = torch.load(query_mixer_path, map_location=device, weights_only=True)
#             model.query_mixer.load_state_dict(sd)
#             print(f"[INFO] Loaded query_mixer from: {query_mixer_path}")
#         else:
#             print(f"[WARN] query_mixer.pt not found: {query_mixer_path}")

#     return model



# def maybe_load_stage1_weights(model, model_cfg, device):
#     stage1_dir = getattr(model_cfg, "init_from_stage1", None)
#     load_stage1_lora = getattr(model_cfg, "load_stage1_lora", False)

#     if not stage1_dir:
#         print("[INFO] No stage1 init dir provided. Train from fresh initialization.")
#         return model

#     print(f"[INFO] Loading stage1 weights from: {stage1_dir}")

#     if load_stage1_lora:
#         model.qwen.load_adapter(stage1_dir, adapter_name="default", is_trainable=True)
#     else:
#         print("[INFO] Stage1 LoRA adapter will NOT be loaded.")

#     projector_path = os.path.join(stage1_dir, "projector.pt")
#     if os.path.exists(projector_path):
#         sd = torch.load(projector_path, map_location=device, weights_only=True)
#         model.projector.load_state_dict(sd)
#     else:
#         print(f"[WARN] projector.pt not found: {projector_path}")

#     query_mixer_path = os.path.join(stage1_dir, "query_mixer.pt")
#     if model.query_mixer is not None:
#         if os.path.exists(query_mixer_path):
#             sd = torch.load(query_mixer_path, map_location=device, weights_only=True)
#             model.query_mixer.load_state_dict(sd)
#         else:
#             print(f"[WARN] query_mixer.pt not found: {query_mixer_path}")

#     return model


def maybe_load_stage1_bridge_weights(model, model_cfg, device):
    stage1_dir = getattr(model_cfg, "init_from_stage1", None)

    if not stage1_dir:
        print("[INFO] No stage1 init dir provided. Train from fresh initialization.")
        return model

    print(f"[INFO] Loading stage1 bridge weights from: {stage1_dir}")

    projector_path = os.path.join(stage1_dir, "projector.pt")
    if os.path.exists(projector_path):
        sd = torch.load(projector_path, map_location=device, weights_only=True)
        model.projector.load_state_dict(sd)
    else:
        print(f"[WARN] projector.pt not found: {projector_path}")

    query_mixer_path = os.path.join(stage1_dir, "query_mixer.pt")
    if model.query_mixer is not None:
        if os.path.exists(query_mixer_path):
            sd = torch.load(query_mixer_path, map_location=device, weights_only=True)
            model.query_mixer.load_state_dict(sd)
        else:
            print(f"[WARN] query_mixer.pt not found: {query_mixer_path}")

    return model

def build_model(model_cfg, device,train_cfg):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_qwen = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        quantization_config=quant_cfg,
        device_map={"": 0},
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
    # qwen.config.use_cache = False
    # qwen = prepare_model_for_kbit_training(qwen)

    # lora = model_cfg.lora
    # peft_cfg = LoraConfig(
    #     r=lora.r,
    #     lora_alpha=lora.lora_alpha,
    #     lora_dropout=lora.lora_dropout,
    #     bias=lora.bias,
    #     task_type=lora.task_type,
    #     target_modules=lora.target_modules,
    # )
    # qwen = get_peft_model(qwen, peft_cfg)
    stage1_dir = getattr(model_cfg, "init_from_stage1", None)
    load_stage1_lora = getattr(model_cfg, "load_stage1_lora", False)

    if stage1_dir is not None and load_stage1_lora:
        print(f"[INFO] Loading stage1 LoRA adapter from: {stage1_dir}")
        qwen = PeftModel.from_pretrained(
            base_qwen,
            stage1_dir,
            is_trainable=True,
        )
    else:
        print("[INFO] Creating a fresh LoRA adapter.")
        qwen = get_peft_model(base_qwen, peft_cfg)

    print("[INFO] Trainable params in qwen:")
    qwen.print_trainable_parameters()
    for n, p in qwen.named_parameters():
        if "lora" in n.lower():
            print("[LoRA trainable check]", n, p.requires_grad)

    blip = BLIP2Model(model_cfg.blip2_model, 
                      device=device, 
                      dtype=torch.float16
                      )

    d_qformer = blip.qformer_dim
    d_qwen = qwen.config.hidden_size

    qwen_dtype = qwen.get_input_embeddings().weight.dtype

    query_mixer=QueryMixerBlock(
        dim=d_qformer,
        num_heads=8,
        mlp_ratio=2.0,
        dropout=0.1,
    ).to(device,dtype=qwen_dtype)
    projector = MLPProjector(
        in_dim=d_qformer,
        out_dim=d_qwen,
        hidden_dim=2*d_qwen,
        use_residual=True,
    ).to(device, dtype=qwen_dtype)

    set_requires_grad(projector, True)
    set_requires_grad(query_mixer,True)

    model = QwenWithBLIPPrefix(
        qwen=qwen,
        blip=blip,
        projector=projector,
        query_mixer=query_mixer,
        use_weighted_loss=train_cfg.use_weighted_loss,
    )

    model = maybe_load_stage1_bridge_weights(model, model_cfg, device)

    return model
