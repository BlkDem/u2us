import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0):
        super().__init__(); self.base=base; self.rank=rank; self.scaling=alpha/rank
        self.A=nn.Parameter(torch.empty(rank, base.in_features)); self.B=nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.drop=nn.Dropout(dropout)
        for p in self.base.parameters(): p.requires_grad_(False)
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self,x): return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scaling

def inject_lora(module, rank=8, alpha=16, dropout=0.05, targets=('linear1','linear2','out_proj','linear')):
    count=0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name.lower() for t in targets):
            setattr(module,name,LoRALinear(child,rank,alpha,dropout)); count+=1
        else: count+=inject_lora(child,rank,alpha,dropout,targets)
    return count

def adapter_state(module): return {k:v.cpu() for k,v in module.state_dict().items() if '.A' in k or '.B' in k}

def load_adapter(module, state):
    current=module.state_dict(); current.update(state); module.load_state_dict(current, strict=False)
