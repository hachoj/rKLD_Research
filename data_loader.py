from operator import is_
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

def load_tokens(filename, is_rope=False):
    """
    loads tokens from the .npy/.npz shards
    """
    if is_rope:
        data = np.load(filename)
        tokens = data['tokens'].astype(np.int32)
        positions = data['positions'].astype(np.int32)
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        positions_t = torch.tensor(positions, dtype=torch.long)
        return tokens_t, positions_t
    else:
        npt = np.load(filename)
        npt = npt.astype(np.int32) 
        ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, is_rope=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.is_rope = is_rope

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        """
        resets position to beginning of shard
        """
        if self.is_rope:
            self.current_shard = 0
            self.tokens, self.positions = load_tokens(self.shards[self.current_shard], is_rope=True)
            self.current_position = self.B * self.T * self.process_rank
        else:
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard], is_rope=False)
            self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        returns the next batch of tokens and positions
        positions will be None unless RoPE is being used in config
        """
        B, T = self.B, self.T
        tokens_buf = self.tokens[self.current_position : self.current_position+B*T+1]
        if self.is_rope:
            positions_buf = self.positions[self.current_position : self.current_position+B*T+1]

        x = tokens_buf[:-1].view(B, T)  # type: ignore
        y = tokens_buf[1:].view(B, T)  # type: ignore

        if self.is_rope:
            pos = positions_buf[:-1].view(B, T) 
        else:
            pos = None

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens, self.positions = load_tokens(self.shards[self.current_shard])
            self.tokens = load_tokens(self.shards[self.current_shard]) 
            self.current_position = B * T * self.process_rank
        return x, y, pos