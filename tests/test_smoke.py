import json
from pathlib import Path
from micronmt.distill import distillation_loss
import torch

def test_distillation_backward():
 s=torch.randn(1,3,8,requires_grad=True); t=torch.randn(1,3,8); y=torch.ones(1,3,dtype=torch.long)
 loss=distillation_loss(s,y,t); loss.backward(); assert s.grad is not None

def test_prepare_data_script_exists():
 assert Path('scripts/prepare_data.py').exists()
