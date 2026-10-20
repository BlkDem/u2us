import torch
import torch.nn.functional as F

def distillation_loss(student_logits, hard_targets, teacher_logits=None, pad_id=0,
                      alpha_hard=.8, beta_kl=.2, temperature=2.0):
    """Combined CE + teacher KL. Teacher logits must have same [B,T,V] shape."""
    ce=F.cross_entropy(student_logits.reshape(-1,student_logits.size(-1)), hard_targets.reshape(-1), ignore_index=pad_id)
    if teacher_logits is None or beta_kl == 0: return alpha_hard*ce
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if teacher_logits.shape != student_logits.shape:
        raise ValueError('Teacher and student logits must have identical shapes')
    mask=hard_targets.ne(pad_id)
    s=F.log_softmax(student_logits / temperature, dim=-1)
    t=F.softmax(teacher_logits / temperature, dim=-1)
    kl=F.kl_div(s, t, reduction='none').sum(-1)
    kl=(kl*mask).sum()/mask.sum().clamp_min(1)
    return alpha_hard*ce + beta_kl*(temperature**2)*kl
