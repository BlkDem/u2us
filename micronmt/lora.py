import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        # Transformer eval fast paths read weights without calling forward.
        return self.base.weight + (self.B @ self.A) * self.scaling

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scaling


def inject_lora(module, rank=8, alpha=16, dropout=0.05,
                targets=('linear1', 'linear2', 'out_proj', 'linear', 'output')):
    if rank <= 0:
        raise ValueError('LoRA rank must be positive')
    module.requires_grad_(False)

    def inject(parent):
        count = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.A.requires_grad_(True)
                child.B.requires_grad_(True)
                continue
            # MultiheadAttention consumes out_proj weights directly, not forward;
            # attention adapters are deliberately unsupported (FFN/output remain).
            if isinstance(child, nn.MultiheadAttention):
                continue
            if isinstance(child, nn.Linear) and any(t in name.lower() for t in targets):
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                count += 1
            else:
                count += inject(child)
        return count

    return inject(module)


def adapter_state(module):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
            if name.rsplit('.', 1)[-1] in ('A', 'B')}


def load_adapter(module, state):
    expected = adapter_state(module)
    if expected.keys() != state.keys():
        raise ValueError('Adapter keys do not match injected LoRA modules')
    module.load_state_dict(state, strict=False)
