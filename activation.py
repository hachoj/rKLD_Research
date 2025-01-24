import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # combine the two weight matrices into one
        # for efficiency
        self.W = nn.Linear(input_dim, 2 * input_dim, bias=False)
        self.silu = nn.SiLU()
        self.input_dim = input_dim

        self.apply(self._init_weights)

    def forward(self, x):
        wv = self.W(x)
        w, v = wv.split(self.input_dim, dim=-1)
        return w * self.silu(v)

    def _init_weights(self, module):
        # kaiming normal initialization of the weights
        if isinstance(module, nn.Linear):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            std = fan_in ** -0.5
            nn.init.normal_(module.weight, 0.0, std)