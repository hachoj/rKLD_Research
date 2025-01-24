import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class config:
    context_len:int = 1024
    vocab_size:int = 50280
    n_layer:int = 12
    n_head:int = 12
    n_embd:int = 768

    # current options: 'learned', 'sinusoidal', 'ROPE'
    pos_embd_type:str = 'learned'
    
    # current options: 'gelu', 'relu', 'swish', 'swiglu
    activation_type:str = 'gelu'

    # current options: 'layer_norm', 'rms_norm'
    norm_type:str = 'layer_norm'


