from __future__ import annotations
import gc
import json
import os
import torch
import torch.nn as nn
from transformers import Blip2Model

from src.BLIP_Qwen.blip2_support import _resolve_lavis_model_type
from src.BLIP_Qwen.qformer import BertConfig, BertLMHeadModel

class BLIP2Model(nn.Module):
    def __init__(
        self,
        blip2_model_id: str,
        device,
        dtype=torch.float32,
        qformer_stage1_dir: str | None = None,
        num_query_token: int = 32,
        cross_attention_freq: int = 2,
        lavis_model_type: str | None = None,
        freeze_vision: bool = True,
        freeze_qformer: bool = False,
        train_query_tokens: bool = False,
    ):
        super().__init__()
        self.blip2_model_id = blip2_model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.qformer_stage1_dir = qformer_stage1_dir
        self.num_query_token = num_query_token
        self.cross_attention_freq = cross_attention_freq

        resolved_model_type = _resolve_lavis_model_type(blip2_model_id, lavis_model_type)
        self.lavis_model_type = resolved_model_type
        full_model = Blip2Model.from_pretrained(
            blip2_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.vision_model = full_model.vision_model.to(self.device, dtype=self.dtype)

        vision_width = self.vision_model.config.hidden_size
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        # LAVIS adds a single extra BOS token "[DEC]" on top of bert-base-uncased.
        # Set the target vocab size up front so we don't rely on newer
        # transformers tie-weight resize internals for this vendored model class.
        encoder_config.vocab_size = 30523
        self.qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased",
            config=encoder_config,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                num_query_token,
                encoder_config.hidden_size,
                device=self.device,
                dtype=torch.float32,
            ),
            requires_grad=train_query_tokens,
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=encoder_config.initializer_range)

        if qformer_stage1_dir is not None:
            self.load_stage1_qformer(qformer_stage1_dir, strict=True)

        if freeze_vision:
            self.vision_model.eval()
            for p in self.vision_model.parameters():
                p.requires_grad = False

        if freeze_qformer:
            self.qformer.eval()
            for p in self.qformer.parameters():
                p.requires_grad = False
        else:
            self.qformer.train()
            for p in self.qformer.parameters():
                p.requires_grad = True

        del full_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_stage1_qformer(self, stage1_dir: str, strict: bool = False):
        qformer_path = os.path.join(stage1_dir, "qformer_stage1.pt")
        query_tokens_path = os.path.join(stage1_dir, "query_tokens_stage1.pt")
        meta_path = os.path.join(stage1_dir, "stage1_meta.json")

        if not os.path.exists(qformer_path):
            raise FileNotFoundError(f"Missing {qformer_path}")
        if not os.path.exists(query_tokens_path):
            raise FileNotFoundError(f"Missing {query_tokens_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("lavis_model_type") != self.lavis_model_type:
            raise ValueError(
                f"Stage1 lavis_model_type={meta.get('lavis_model_type')} does not match "
                f"stage2 lavis_model_type={self.lavis_model_type}."
            )
        if meta.get("num_query_token") != self.num_query_token:
            raise ValueError(
                f"Stage1 num_query_token={meta.get('num_query_token')} does not match "
                f"stage2 num_query_token={self.num_query_token}."
            )
        if meta.get("cross_attention_freq") != self.cross_attention_freq:
            raise ValueError(
                f"Stage1 cross_attention_freq={meta.get('cross_attention_freq')} does not match "
                f"stage2 cross_attention_freq={self.cross_attention_freq}."
            )
        if meta.get("qformer_hidden_size") != self.query_tokens.shape[-1]:
            raise ValueError(
                f"Stage1 qformer_hidden_size={meta.get('qformer_hidden_size')} does not match "
                f"stage2 hidden_size={self.query_tokens.shape[-1]}."
            )

        sd = torch.load(qformer_path, map_location=self.device)
        self.qformer.load_state_dict(sd, strict=strict)

        qt = torch.load(query_tokens_path, map_location=self.device)
        if qt.ndim == 2:
            qt = qt.unsqueeze(0)

        with torch.no_grad():
            self.query_tokens.copy_(qt.to(self.device, dtype=self.query_tokens.dtype))

    def save_qformer_for_stage2(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.qformer.state_dict(), os.path.join(out_dir, "qformer_stage1.pt"))
        torch.save(self.query_tokens.detach().cpu(), os.path.join(out_dir, "query_tokens_stage1.pt"))
        meta = {
            "blip2_model_id": self.blip2_model_id,
            "lavis_model_type": self.lavis_model_type,
            "num_query_token": self.query_tokens.shape[1],
            "qformer_hidden_size": self.query_tokens.shape[2],
            "cross_attention_freq": self.qformer.config.cross_attention_freq,
        }
        with open(os.path.join(out_dir, "stage1_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def forward(self, pixel_values):
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state.to(dtype=torch.float32)

        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        B = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1).to(dtype=torch.float32)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=False,
            return_dict=True,
        )
        return query_output.last_hidden_state

    @property
    def qformer_dim(self):
        return self.query_tokens.shape[-1]

    @property
    def prefix_len(self):
        return self.query_tokens.shape[1]
