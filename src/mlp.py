import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import SwiGLU

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        if (config.activation_type == 'swiglu'):
            self.activation = SwiGLU(4 * config.n_embd)
        elif (config.activation_type == 'gelu'):
            self.activation = nn.GELU(approximate='tanh')
        elif (config.activation_type == 'relu'):
            self.activation = nn.ReLU()
        elif (config.activation_type == 'swish'):
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"activation_type {config.activation_type} not recognized")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.REGULARIZE = 1 # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x