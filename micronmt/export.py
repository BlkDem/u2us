import argparse, torch
from torch import nn
from .checkpoint import load_checkpoint

class ExportWrapper(nn.Module):
 def __init__(self, model): super().__init__(); self.model=model
 def forward(self, src, tgt): return self.model(src,tgt,pad_id=0)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--out',required=True); p.add_argument('--sequence-len',type=int,default=128); a=p.parse_args()
 m,_,_=load_checkpoint(a.checkpoint); m=ExportWrapper(m); src=torch.ones(1,a.sequence_len,dtype=torch.long); tgt=torch.ones(1,a.sequence_len,dtype=torch.long)
 torch.onnx.export(m,(src,tgt),a.out,input_names=['src','tgt'],output_names=['logits'],opset_version=17,dynamo=False)
 print(a.out)
if __name__=='__main__': main()
