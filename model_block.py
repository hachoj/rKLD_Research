import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention
from mlp import MLP

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.norm_type == 'layer_norm':
            self.norm_1 = nn.LayerNorm(config.n_embd)
            self.norm_2 = nn.LayerNorm(config.n_embd)
        elif config.norm_type == 'rms_norm':
            self.norm_1 = nn.RMSNorm(config.n_embd)
            self.norm_2 = nn.RMSNorm(config.n_embd)
        else:
            raise ValueError(f"norm_type {config.norm_type} not recognized")
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, positions=None):
        x = x + self.attn(self.norm_1(x), positions)
        x = x + self.mlp(self.norm_2(x))
        return x