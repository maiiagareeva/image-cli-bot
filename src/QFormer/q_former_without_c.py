import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from src.QFormer.cross_attn import CrossAttention
class QFormer(nn.Module):

    def __init__(
        self,
        num_query_tokens,
        vision_hidden_dim=521,
        qformer_hidden_dim=768,
        num_hidden_layers= 12,
        num_heads=12,        
        pretrained_bert= "bert-base-uncased",
    ):
        super().__init__()

        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(
            torch.empty(1, num_query_tokens, qformer_hidden_dim)
        )
        nn.init.normal_(self.query_tokens,mean=0.0,std=0.02)

        config = BertConfig.from_pretrained(pretrained_bert)
        config.encoder_width = vision_hidden_dim
        config.add_cross_attention = False

        config.hidden_size = qformer_hidden_dim
        config.num_hidden_layers = num_hidden_layers
        config.is_decoder = False
        
        self.bert = BertModel.from_pretrained(
            pretrained_bert,
            config=config,
            add_pooling_layer=False,
            use_cache=False,
        )

        self.cross_attn=CrossAttention(
            q_dim=qformer_hidden_dim,
            kv_dim=vision_hidden_dim,
            num_heads=num_heads,
        )

        self.norm=nn.LayerNorm(qformer_hidden_dim)

    def forward(
        self,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor | None = None,
    ):
        """
        image_embeds: (B, N, Dv)  ← CLIP patch / grid features
        image_atts: (B, N)
        """        
        #drop CLS already happedn in QwenwithPrefix forward
        # image_embeds = image_embeds[:, 1:, :]  # drop CLS 
        q = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if image_atts is None:
            B, N = image_embeds.shape[:2]
            image_atts = torch.ones((B, N), dtype=torch.long, device=image_embeds.device)
        
        attention_mask=torch.ones(
            (image_embeds.shape[0],self.num_query_tokens),
            device=q.device,
            dtype=torch.long,
        )

        q = self.bert(
            inputs_embeds=q,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state[:, : self.num_query_tokens, :]

        q = q + self.cross_attn(q,image_embeds,image_atts)

        q = self.norm(q)

        return q
