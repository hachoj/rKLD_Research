import torch.nn.functional as F

def reverse_kl_divergence(student_logits, teacher_logits, temperature=1.0, alpha=0.9, length_penalty=1.0):
    """
    Computes Reverse KLD (KL(student || teacher)) with teacher-mixed sampling and length normalization.
    """
    # Apply temperature scaling
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Teacher-Mixed Sampling: Mix the teacher and student distributions
    mixed_probs = alpha * teacher_probs + (1 - alpha) * F.softmax(student_logits / temperature, dim=-1)

    # Compute reverse KLD (student_probs || mixed_probs)
    reverse_kld_loss = F.kl_div(student_probs, mixed_probs, reduction='batchmean')

    # Apply length normalization (optional step if needed for sequence tasks)
    sequence_lengths = student_logits.size(1)  # Assuming second dim is sequence length
    normalized_loss = reverse_kld_loss / (sequence_lengths ** length_penalty)

    return normalized_loss