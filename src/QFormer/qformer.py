import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class QFormer(nn.Module):

    def __init__(
        self,
        num_query_tokens,
        vision_hidden_dim=521,
        qformer_hidden_dim=768,
        num_hidden_layers= 12,
        cross_attention_freq= 2,
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
        config.add_cross_attention = True
        config.cross_attention_freq = cross_attention_freq


        config.hidden_size = qformer_hidden_dim
        config.num_hidden_layers = num_hidden_layers
        config.is_decoder = True
        
        self.transformer = BertModel.from_pretrained(
            pretrained_bert,
            config=config,
            add_pooling_layer=False,
        )

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
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.transformer(
            inputs_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        query_output = query_output.last_hidden_state[:, : self.num_query_tokens, :]

        return query_output
