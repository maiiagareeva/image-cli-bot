import torch.nn as nn
from transformers import BertConfig, BertModel
from .cross_attn import CrossAttention

class QueryMixerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CrossAttention(q_dim=dim, kv_dim=dim, num_heads=num_heads, dropout=dropout)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv=None, kv_mask=None):
        if kv is None:
            kv = q
        q = q + self.attn(self.norm1(q), self.norm1(kv), kv_mask=kv_mask)
        q = q + self.ffn(self.norm2(q))
        return q
