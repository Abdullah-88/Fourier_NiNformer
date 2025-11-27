import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class FFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real


class FourierGatingUnit(nn.Module):
    def __init__(self,dim, hidden_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(dim,dim)      
        self.fft = FFTLayer()
       

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)   
        v = self.fft(v)
        out = u * v
        return out


class FourierNiNformerBlock(nn.Module):
    def __init__(self, d_model, d_ffn,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.mgu = FourierGatingUnit(d_model,d_ffn,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out


class FourierNiNformer(nn.Module):
    def __init__(self, d_model, d_ffn,num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[FourierNiNformerBlock(d_model, d_ffn,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)








