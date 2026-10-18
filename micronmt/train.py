import argparse, json, random, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .model import MicroTransformer
from .data import ParallelDataset, collate
from .tokenizer import SimpleTokenizer
from .sp_tokenizer import SentencePieceTokenizer
from .distill import distillation_loss

MODEL_KEYS=('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')

def load_tokenizer(args, saved, data_path):
 if args.sp_model: return SentencePieceTokenizer(args.sp_model)
 if saved: return SimpleTokenizer(saved.get('tokenizer',{}).get('vocab',saved.get('tokenizer',{})))
 texts=[]
 for line in Path(data_path).read_text(encoding='utf8').splitlines():
  try:
   row=json.loads(line); texts.extend((row['src'],row['tgt']))
  except (ValueError, KeyError): pass
 return SimpleTokenizer.from_texts(texts)

def build_model(vocab_size, config):
 return MicroTransformer(vocab_size, **{k:config[k] for k in MODEL_KEYS})

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--valid')
 p.add_argument('--resume'); p.add_argument('--sp-model'); p.add_argument('--teacher')
 a=p.parse_args(); c=yaml.safe_load(open(a.config)); random.seed(c['seed']); torch.manual_seed(c['seed'])
 saved=torch.load(a.resume,map_location='cpu') if a.resume else None
 tok=load_tokenizer(a,saved,a.data)
 ds=ParallelDataset(a.data,tok,c['max_len']); dl=DataLoader(ds,c['batch_size'],shuffle=True,collate_fn=lambda x:collate(x,tok,c['max_len']))
 m=build_model(tok.vocab_size,c)
 if saved: m.load_state_dict(saved['model'])
 teacher=None
 if a.teacher:
  tq=torch.load(a.teacher,map_location='cpu'); teacher=build_model(tok.vocab_size,tq['config']); teacher.load_state_dict(tq['model']); teacher.eval()
  for param in teacher.parameters(): param.requires_grad_(False)
 opt=torch.optim.AdamW(m.parameters(),lr=c['lr']); ce=torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id,label_smoothing=c['label_smoothing'])
 valid_dl=None
 if a.valid:
  valid_ds=ParallelDataset(a.valid,tok,c['max_len']); valid_dl=DataLoader(valid_ds,c['batch_size'],shuffle=False,collate_fn=lambda x:collate(x,tok,c['max_len']))
 Path(c['run_dir']).mkdir(parents=True,exist_ok=True); best=float('inf')
 alpha=float(c.get('alpha_hard',1.0)); beta=float(c.get('beta_kl',0.0)); temperature=float(c.get('temperature',2.0))
 for ep in range(c['epochs']):
  m.train(); total=0
  for src,tgt in dl:
   opt.zero_grad(); student=m(src,tgt[:,:-1]); hard=tgt[:,1:]
   with torch.no_grad(): teacher_logits=teacher(src,tgt[:,:-1]) if teacher else None
   loss=distillation_loss(student,hard,teacher_logits,tok.pad_id,alpha,beta,temperature) if teacher else ce(student.reshape(-1,student.size(-1)),hard.reshape(-1))
   loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); total+=loss.item()
  train_loss=total/max(1,len(dl)); valid_loss=None
  if valid_dl:
   m.eval(); total=0
   with torch.no_grad():
    for src,tgt in valid_dl:
     logits=m(src,tgt[:,:-1]); total += ce(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)).item()
   valid_loss=total/max(1,len(valid_dl))
  score=valid_loss if valid_loss is not None else train_loss
  print(f'epoch={ep+1} train_loss={train_loss:.4f}' + (f' valid_loss={valid_loss:.4f}' if valid_loss is not None else ''))
  meta={'type':'sentencepiece','model_file':str(a.sp_model)} if a.sp_model else {'type':'simple','vocab':tok.vocab}
  payload={'model':m.state_dict(),'config':c,'tokenizer':meta,'epoch':ep+1,'train_loss':train_loss,'valid_loss':valid_loss,'teacher':a.teacher,'distillation':{'alpha_hard':alpha,'beta_kl':beta,'temperature':temperature}}
  torch.save(payload,Path(c['run_dir'])/'last.pt')
  if score < best: best=score; torch.save(payload,Path(c['run_dir'])/'best.pt')
if __name__=='__main__': main()
