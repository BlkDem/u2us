import argparse, json, random, shutil, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .checkpoint import load_checkpoint
from .data import ParallelDataset, collate
from .distill import distillation_loss

def merge_replay(new_path, replay_path, out, replay_ratio=.25, seed=42):
 new=Path(new_path).read_text(encoding='utf8').splitlines(); old=Path(replay_path).read_text(encoding='utf8').splitlines() if Path(replay_path).exists() else []
 rng=random.Random(seed); rng.shuffle(old); keep=old[:int(max(0,len(new))*replay_ratio/(1-replay_ratio))] if replay_ratio < 1 else old
 rows=[]; seen=set()
 for line in new+keep:
  try: key=(json.loads(line)['src'],json.loads(line)['tgt'])
  except (ValueError,KeyError): continue
  if key not in seen: seen.add(key); rows.append(json.dumps({'src':key[0],'tgt':key[1]},ensure_ascii=False))
 Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text('\n'.join(rows)+'\n',encoding='utf8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--regression',required=True);p.add_argument('--out',required=True);p.add_argument('--replay');p.add_argument('--epochs',type=int,default=3);p.add_argument('--replay-ratio',type=float,default=.25);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--direction',choices=('ru2en','en2ru'),default='ru2en');a=p.parse_args()
 work=Path(a.out).with_suffix('.replay.jsonl'); merge_replay(a.data,a.replay,work,a.replay_ratio) if a.replay else shutil.copyfile(a.data,work)
 model,tok,meta=load_checkpoint(a.checkpoint); ds=ParallelDataset(str(work),tok,meta['config']['max_len'],direction=a.direction); dl=DataLoader(ds,16,shuffle=True,collate_fn=lambda x:collate(x,tok,meta['config']['max_len'],direction=a.direction))
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr); ce=torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id); model.train()
 for ep in range(a.epochs):
  total=0
  for src,tgt in dl:
   opt.zero_grad(); logits=model(src,tgt[:,:-1]); loss=ce(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()
  print(f'epoch={ep+1} loss={total/max(1,len(dl)):.4f}')
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'config':meta['config'],'tokenizer':meta['tokenizer'],'continual':{'base':a.checkpoint,'replay_ratio':a.replay_ratio,'regression_set':a.regression}},a.out)
if __name__=='__main__':main()
