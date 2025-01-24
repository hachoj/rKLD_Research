import math

def get_lr(it, warmup_steps, max_steps, min_lr, max_lr):
    """
    cosine decay from max_lr to min_lr with an initial linear warmup
    """
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def get_alpha_cosine_decay(current_step, total_steps, initial_alpha=0.9, final_alpha=0.1):
    """
    cosine decay from initial_alpha to final_alpha
    """
    progress = current_step / total_steps
    # Cosine decay from initial_alpha to final_alpha
    alpha = final_alpha + 0.5 * (initial_alpha - final_alpha) * (1 + math.cos(math.pi * progress))
    return alpha

def cosine_lr_kld(current_step, total_steps, initial_lr=2e-4, min_lr=1e-5):
    """
    cosine decay from initial_lr to min_lr
    """
    progress = current_step / total_steps
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))