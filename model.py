import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect

from model_block import Block

class SLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        if config.pos_embd_type == 'learned':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.context_len, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            ))
        elif config.pos_embd_type == 'sinusoidal':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.context_len, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            ))
        elif config.pos_embd_type == 'ROPE':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            ))
        else:
            raise ValueError(f"pos_embd_type {config.pos_embd_type} not recognized")

        # the final layer is a linear projection that maps the output of the transformer to the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # tie the weights of the embedding and output layer
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'REGULARIZE'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
     
    def forward(self, idx, targets=None, positions=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.context_len, "Cannot forward, model block size is exhausted."
        
        # forward the token and position embeddings 
        if self.config.pos_embd_type == 'learned':
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
            pos_embd = self.transformer.wpe(pos)
            
            tok_embd = self.transformer.wte(idx) # token embedding of shape (B, T, n_embd)
            x = tok_embd + pos_embd
        elif self.config.pos_embd_type == 'ROPE':
            tok_embd = self.transformer.wte(idx)
            x = tok_embd
        # froward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, positions=positions)
        # forward the final layer norm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create a fused adamW optimizer if possible
        fused_availabel = 'fused' in inspect.signature(torch.optim.AdamW).parameters # type: ignore
        use_fused = fused_availabel and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8) # type: ignore
        return optimizer
