from __future__ import annotations
import gc
import torch
import torch.nn as nn
from transformers import Blip2Model

class BLIP2Model(nn.Module): 
    def __init__(self, blip2_model_id, device, dtype= torch.float16):
        super().__init__()
        self.blip2_model_id = blip2_model_id
        self.device = torch.device(device)
        self.dtype = dtype

        full_model = Blip2Model.from_pretrained(
            blip2_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        self.vision_model = full_model.vision_model.eval().to(self.device, dtype=self.dtype)
        self.qformer = full_model.qformer.eval().to(self.device, dtype=self.dtype)
        self.query_tokens = nn.Parameter(full_model.query_tokens.detach().to(self.device, dtype=self.dtype), requires_grad=False)

        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.qformer.parameters():
            p.requires_grad = False

        del full_model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, pixel_values):
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state

        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        B = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)

        query_embeds = self.qformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_embeds.last_hidden_state

    @property
    def qformer_dim(self):
        return self.query_tokens.shape[-1]

    @property
    def prefix_len(self):
        return self.query_tokens.shape[1]
