import argparse, json, random, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .model import MicroTransformer
from .data import ParallelDataset, collate
from .tokenizer import SimpleTokenizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--valid'); p.add_argument('--resume'); a=p.parse_args()
 c=yaml.safe_load(open(a.config)); random.seed(c['seed']); torch.manual_seed(c['seed'])
 if a.resume:
  saved=torch.load(a.resume,map_location='cpu'); tok=SimpleTokenizer(saved.get('tokenizer'))
 else:
  texts=[]
  for line in Path(a.data).read_text(encoding='utf8').splitlines():
   try:
    row=json.loads(line); texts.extend((row['src'],row['tgt']))
   except (ValueError, KeyError): pass
  tok=SimpleTokenizer.from_texts(texts)
 ds=ParallelDataset(a.data,tok,c['max_len']); dl=DataLoader(ds,c['batch_size'],shuffle=True,collate_fn=lambda x:collate(x,tok,c['max_len']))
 m=MicroTransformer(tok.vocab_size,**{k:c[k] for k in ('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')})
 if a.resume: m.load_state_dict(saved['model'])
 opt=torch.optim.AdamW(m.parameters(),lr=c['lr']); loss_fn=torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id,label_smoothing=c['label_smoothing'])
 Path(c['run_dir']).mkdir(parents=True,exist_ok=True)
 for ep in range(c['epochs']):
  m.train(); total=0
  for src,tgt in dl:
   opt.zero_grad(); logits=m(src,tgt[:,:-1]); loss=loss_fn(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); total+=loss.item()
  print(f'epoch={ep+1} loss={total/max(1,len(dl)):.4f}')
  torch.save({'model':m.state_dict(),'config':c,'tokenizer':tok.vocab},Path(c['run_dir'])/'last.pt')
if __name__=='__main__': main()
