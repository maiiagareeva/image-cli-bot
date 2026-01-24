import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
class CrossAttention:
    def __init__(self,q_dim,kv_dim,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.scale=(q_dim//num_heads)**-0.5
        self.q_proj=nn.Linear(q_dim,q_dim)
        self.k_proj=nn.Linear(kv_dim,q_dim)
        self.v_proj=nn.Linear(kv_dim,q_dim)
        self.out_proj=nn.Linear(q_dim,q_dim)

    def forward(self,q,kv,kv_mask=None):
        B,Q,D=q.shape
        N=kv.size(1)
        H=self.num_heads
        Dh=D//H

        Q=self.q_proj(q).view(B,Q,H,Dh).transpose(1,2) #(B,H,Q,Dh)
        K=self.k_proj(kv).view(B,N,H,Dh).transpose(1,2) #(B,H,N,Dh)
        V=self.k_proj(kv).view(B,N,H,Dh).transpose(1,2) #(B,H,N,Dh)

        attn=(q @ K.transpose(-2,-1))*self.scale   #(B,H,Q,Dh)@(B,H,N,Dhh)-->(B,H,Q,N)

        if kv_mask is not None:
            attn=attn.masked_fill(
                kv_mask[:,None,None,:]==0,
                float("-inf")
            ) #(B,N)-->(B,1,1,N)---braodcast-->(B,H,Q,N)
        
        attn=attn.softmax(dim=-1)
        out=(attn @ V)     #(B,H,Q,N) @(B,H,N,Dh)--->(B,H,Q,Dh)

        out=out.transpose(1,2).contiguous().view(B,Q,D)

        return self.out_proj(out)