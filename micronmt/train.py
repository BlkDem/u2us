import argparse, json, random, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .model import MicroTransformer
from .data import ParallelDataset, collate
from .tokenizer import SimpleTokenizer
from .sp_tokenizer import SentencePieceTokenizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--valid'); p.add_argument('--resume'); p.add_argument('--sp-model'); a=p.parse_args()
 c=yaml.safe_load(open(a.config)); random.seed(c['seed']); torch.manual_seed(c['seed'])
 saved=None
 if a.sp_model: tok=SentencePieceTokenizer(a.sp_model)
 elif a.resume:
  saved=torch.load(a.resume,map_location='cpu'); tok=SimpleTokenizer(saved.get('tokenizer',{}).get('vocab',saved.get('tokenizer',{})))
 else:
  texts=[]
  for line in Path(a.data).read_text(encoding='utf8').splitlines():
   try: row=json.loads(line); texts.extend((row['src'],row['tgt']))
   except (ValueError, KeyError): pass
  tok=SimpleTokenizer.from_texts(texts)
 ds=ParallelDataset(a.data,tok,c['max_len']); dl=DataLoader(ds,c['batch_size'],shuffle=True,collate_fn=lambda x:collate(x,tok,c['max_len']))
 m=MicroTransformer(tok.vocab_size,**{k:c[k] for k in ('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')})
 if saved: m.load_state_dict(saved['model'])
 opt=torch.optim.AdamW(m.parameters(),lr=c['lr']); loss_fn=torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id,label_smoothing=c['label_smoothing'])
 valid_dl=None
 if a.valid:
  valid_ds=ParallelDataset(a.valid,tok,c['max_len']); valid_dl=DataLoader(valid_ds,c['batch_size'],shuffle=False,collate_fn=lambda x:collate(x,tok,c['max_len']))
 Path(c['run_dir']).mkdir(parents=True,exist_ok=True); best=float('inf')
 for ep in range(c['epochs']):
  m.train(); total=0
  for src,tgt in dl:
   opt.zero_grad(); logits=m(src,tgt[:,:-1]); loss=loss_fn(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); total+=loss.item()
  train_loss=total/max(1,len(dl)); valid_loss=None
  if valid_dl:
   m.eval(); total=0
   with torch.no_grad():
    for src,tgt in valid_dl:
     logits=m(src,tgt[:,:-1]); total += loss_fn(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)).item()
   valid_loss=total/max(1,len(valid_dl))
  score=valid_loss if valid_loss is not None else train_loss
  print(f'epoch={ep+1} train_loss={train_loss:.4f}' + (f' valid_loss={valid_loss:.4f}' if valid_loss is not None else ''))
  meta={'type':'sentencepiece','model_file':str(a.sp_model)} if a.sp_model else {'type':'simple','vocab':tok.vocab}
  payload={'model':m.state_dict(),'config':c,'tokenizer':meta,'epoch':ep+1,'train_loss':train_loss,'valid_loss':valid_loss}
  torch.save(payload,Path(c['run_dir'])/'last.pt')
  if score < best: best=score; torch.save(payload,Path(c['run_dir'])/'best.pt')
if __name__=='__main__': main()
