from __future__ import annotations
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, use_residual= True,dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim =2*out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.ln  = nn.LayerNorm(out_dim)

        self.use_residual = use_residual
        self.proj = nn.Identity() if (in_dim == out_dim) else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        if self.use_residual:
            y = y + self.proj(x)
        y = self.ln(y)
        return y
