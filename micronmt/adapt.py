import argparse, json, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .checkpoint import load_checkpoint
from .data import ParallelDataset, collate
from .lora import inject_lora, adapter_state

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=3); p.add_argument('--rank',type=int,default=8); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--direction',choices=('ru2en','en2ru'),default='ru2en'); a=p.parse_args()
 model,tok,meta=load_checkpoint(a.checkpoint); n=inject_lora(model,a.rank); print(f'injected {n} LoRA modules')
 ds=ParallelDataset(a.data,tok,meta['config']['max_len'],direction=a.direction); dl=DataLoader(ds,16,shuffle=True,collate_fn=lambda x:collate(x,tok,meta['config']['max_len'],direction=a.direction))
 trainable=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(trainable,lr=a.lr); ce=torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id)
 model.train()
 for ep in range(a.epochs):
  total=0
  for src,tgt in dl:
   opt.zero_grad(); logits=model(src,tgt[:,:-1]); loss=ce(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)); loss.backward(); opt.step(); total+=loss.item()
  print(f'epoch={ep+1} loss={total/max(1,len(dl)):.4f}')
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'base_checkpoint':a.checkpoint,'adapter':adapter_state(model),'rank':a.rank},a.out)
if __name__=='__main__': main()
