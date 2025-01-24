import torch
import torch.nn as nn
import torch.nn.functional as F

# this is for an easy PyTorch implementation of RoPE 
import torchtune.modules as ttm

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # makes sure that the number of embedding is some multiple of
        # the number of heads
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.REGULARIZE = 1  # type: ignore
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.config = config

        if config.pos_embd_type == 'ROPE':
            self.head_dim = config.n_embd // config.n_head
            self.rope = ttm.RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.context_len)  # input tensor with shape [b, s, n_h, h_d]

    def forward(self, x, positions=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v  = qkv.split(self.n_embd, dim=2)
        if self.config.pos_embd_type == 'ROPE':
            k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
            # apply ROPE
            # positions = None
            if positions is not None:
                q = self.rope(q, input_pos=positions)
                k = self.rope(k, input_pos=positions)
            else:
                q = self.rope(q)
                k = self.rope(k)

            # now transposing qkv for flash attention

            q = q.transpose(1, 2) # (B, nh, T, hs)
            k = k.transpose(1, 2) # (B, nh, T, hs)
            v = v.transpose(1, 2) # (B, nh, T, hs)
        else:
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs) Flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y