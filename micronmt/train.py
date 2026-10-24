import argparse, json, math, random, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from .batching import LengthBucketBatchSampler
from .model import MicroTransformer
from .data import ParallelDataset, collate
from .tokenizer import SimpleTokenizer
from .sp_tokenizer import SentencePieceTokenizer
from .distill import distillation_loss
from .checkpoint import load_tokenizer_metadata, tokenizer_metadata, tokenizer_fingerprint

MODEL_KEYS=('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')

def resolve_device(name):
 if name in (None,'','auto'):
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if str(name).lower() in ('dml','directml'):
  import torch_directml
  return torch_directml.device()
 return torch.device(name)

def load_tokenizer(args, saved, data_path):
 if args.sp_model:
  tok=SentencePieceTokenizer(args.sp_model)
  if saved and tokenizer_fingerprint(tok)!=tokenizer_fingerprint(load_tokenizer_metadata(saved['tokenizer'],args.resume)):
   raise ValueError('resume tokenizer is incompatible with --sp-model')
  return tok
 if saved: return load_tokenizer_metadata(saved['tokenizer'],args.resume)
 texts=[]
 for line in Path(data_path).read_text(encoding='utf8').splitlines():
  try:
   row=json.loads(line)
   if isinstance(row,dict) and all(isinstance(row.get(k),str) for k in ('src','tgt')):
    texts.extend((row['src'],row['tgt']))
  except (ValueError, KeyError): pass
 return SimpleTokenizer.from_texts(texts)

def build_model(vocab_size, config):
 return MicroTransformer(vocab_size, **{k:config[k] for k in MODEL_KEYS},
                         init_mode=config.get('init_mode', 'controlled'))

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--valid')
 p.add_argument('--resume'); p.add_argument('--sp-model'); p.add_argument('--teacher'); p.add_argument('--direction',choices=['ru2en','en2ru'],default=None); p.add_argument('--device',default='auto'); p.add_argument('--grad-accum',type=int,default=1); p.add_argument('--amp',action='store_true'); p.add_argument('--max-steps',type=int,default=0); p.add_argument('--patience',type=int,default=0); p.add_argument('--eval-samples',type=int,default=0)
 p.add_argument('--history',help='append epoch metrics and fixed validation hypotheses as JSONL')
 a=p.parse_args()
 with open(a.config) as f: c=yaml.safe_load(f)
 if a.grad_accum<1 or min(a.max_steps,a.patience,a.eval_samples)<0: p.error('invalid accumulation or negative step/patience/sample limit')
 if c['epochs']<1 or c['batch_size']<1 or c['max_len']<3 or not math.isfinite(c['lr']) or c['lr']<=0: p.error('epochs, batch_size, lr must be positive and max_len >= 3')
 if not 0<=c['label_smoothing']<=1: p.error('label_smoothing must be between 0 and 1')
 random.seed(c['seed']); torch.manual_seed(c['seed'])
 dev=resolve_device(a.device); amp_device=dev.type if dev.type in ('cpu','cuda') else 'cpu'; amp_enabled=bool(a.amp) and amp_device==dev.type
 print(f'device={dev} amp={amp_enabled}',flush=True)
 saved=torch.load(a.resume,map_location='cpu',weights_only=True) if a.resume else None
 a.direction=a.direction or (saved.get('direction','ru2en') if saved else 'ru2en')
 tok=load_tokenizer(a,saved,a.data)
 def loader(path):
  ds=ParallelDataset(path,tok,c['max_len'],direction=a.direction)
  if not len(ds): raise ValueError(f'empty dataset: {path}')
  sampler=LengthBucketBatchSampler(ds,c['batch_size'],seed=c['seed'])
  return ds,DataLoader(ds,batch_sampler=sampler,collate_fn=lambda x:collate(x,tok,c['max_len'],direction=a.direction),generator=torch.Generator().manual_seed(c['seed']))
 ds,dl=loader(a.data)
 valid_ds,valid_dl=loader(a.valid) if a.valid else (None,None)
 m=build_model(tok.vocab_size,c)
 if saved: m.load_state_dict(saved['model'])
 m.to(dev)
 teacher=None
 if a.teacher:
  tq=torch.load(a.teacher,map_location='cpu',weights_only=True)
  teacher_tok=load_tokenizer_metadata(tq['tokenizer'],a.teacher)
  if tokenizer_fingerprint(tok)!=tokenizer_fingerprint(teacher_tok): raise ValueError('teacher tokenizer is incompatible with student tokenizer')
  teacher=build_model(tok.vocab_size,tq['config']); teacher.load_state_dict(tq['model']); teacher.eval(); teacher.to(dev)
  for param in teacher.parameters(): param.requires_grad_(False)
 opt=torch.optim.AdamW(m.parameters(),lr=c['lr'])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(1,c['epochs']*math.ceil(len(dl)/a.grad_accum)))
 ce=torch.nn.CrossEntropyLoss(ignore_index=tok.pad_id,label_smoothing=c['label_smoothing']); scaler=torch.amp.GradScaler(amp_device,enabled=amp_enabled)
 if saved:
  if 'optimizer' in saved: opt.load_state_dict(saved['optimizer'])
  if 'scheduler' in saved: scheduler.load_state_dict(saved['scheduler'])
  if saved.get('scaler'): scaler.load_state_dict(saved['scaler'])
  if 'python_rng_state' in saved: random.setstate(saved['python_rng_state'])
  if 'torch_rng_state' in saved: torch.set_rng_state(saved['torch_rng_state'])
 Path(c['run_dir']).mkdir(parents=True,exist_ok=True)
 best=saved.get('best',float('inf')) if saved else float('inf'); bad_epochs=saved.get('bad_epochs',0) if saved else 0
 start_epoch=saved.get('epoch',0) if saved else 0; offset=saved.get('batch_offset',0) if saved else 0
 global_step=saved.get('global_step',start_epoch*len(dl)+offset) if saved else 0
 alpha=float(c.get('alpha_hard',1.0)); beta=float(c.get('beta_kl',0.0)); temperature=float(c.get('temperature',2.0))
 if temperature<=0 or alpha<0 or beta<0: p.error('invalid distillation parameters')
 for ep in range(start_epoch,c['epochs']):
  if a.max_steps and global_step>=a.max_steps: break
  m.train(); total=0; processed=0; pending=0; opt.zero_grad()
  remaining=len(dl)-offset
  if a.max_steps: remaining=min(remaining,a.max_steps-global_step)
  for step,(src,tgt) in enumerate(dl):
   if step<offset: continue
   if processed>=remaining: break
   src=src.to(dev); tgt=tgt.to(dev)
   # Normalize each accumulation window by its actual size, including the last partial window.
   window=min(a.grad_accum,remaining-(processed//a.grad_accum)*a.grad_accum)
   with torch.autocast(amp_device,enabled=amp_enabled):
    student=m(src,tgt[:,:-1]); hard=tgt[:,1:]
    with torch.no_grad(): teacher_logits=teacher(src,tgt[:,:-1]) if teacher else None
    loss=distillation_loss(student,hard,teacher_logits,tok.pad_id,alpha,beta,temperature) if teacher else ce(student.reshape(-1,student.size(-1)),hard.reshape(-1))
   scaler.scale(loss/window).backward(); total+=loss.item(); processed+=1; pending+=1; global_step+=1
   if pending==window:
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); scaler.step(opt); scaler.update(); opt.zero_grad(); scheduler.step(); pending=0
  train_loss=total/processed; valid_loss=None
  if valid_dl is not None:
   m.eval(); total=0
   with torch.no_grad():
    for src,tgt in valid_dl:
     src=src.to(dev); tgt=tgt.to(dev)
     logits=m(src,tgt[:,:-1]); total+=ce(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1)).item()
   valid_loss=total/len(valid_dl)
  valid_chrf=None; hyps=[]; refs=[]; rows=[]
  if a.eval_samples and a.valid:
   try:
    import sacrebleu
    rows=valid_ds.rows[:a.eval_samples]; hyps=[]; refs=[]; m.eval()
    with torch.no_grad():
     for row in rows:
      src_text,tgt_text=row[:2]; ids=torch.tensor([tok.encode(('<ru2en>' if a.direction=='ru2en' else '<en2ru>')+' '+src_text,c['max_len'])]).to(dev); hyps.append(tok.decode(m.greedy(ids,max_len=c['max_len'])[0])); refs.append(tgt_text)
    valid_chrf=sacrebleu.corpus_chrf(hyps,[refs]).score
   except ImportError: pass
  score=(-valid_chrf if valid_chrf is not None else (valid_loss if valid_loss is not None else train_loss))
  print(f'epoch={ep+1} train_loss={train_loss:.4f}' + (f' valid_loss={valid_loss:.4f}' if valid_loss is not None else '') + (f' valid_chrf={valid_chrf:.3f}' if valid_chrf is not None else ''))
  if a.history:
   history_path=Path(a.history); history_path.parent.mkdir(parents=True,exist_ok=True)
   with history_path.open('a',encoding='utf8') as f:
    f.write(json.dumps({'epoch':ep+1,'global_step':global_step,'train_loss':train_loss,'valid_loss':valid_loss,'valid_chrf':valid_chrf,'samples':[{'source':r[0],'reference':ref,'hypothesis':hyp} for r,ref,hyp in zip(rows,refs,hyps)]},ensure_ascii=False)+'\n')
  improved=score<best
  if improved: best=score; bad_epochs=0
  else: bad_epochs+=1
  next_offset=offset+processed; complete=next_offset==len(dl)
  payload={'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'config':c,'tokenizer':tokenizer_metadata(tok),'epoch':ep+1 if complete else ep,'batch_offset':0 if complete else next_offset,'global_step':global_step,'best':best,'bad_epochs':bad_epochs,'python_rng_state':random.getstate(),'torch_rng_state':torch.get_rng_state(),'direction':a.direction,'train_loss':train_loss,'valid_loss':valid_loss,'teacher':a.teacher,'distillation':{'alpha_hard':alpha,'beta_kl':beta,'temperature':temperature},'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict()}
  torch.save(payload,Path(c['run_dir'])/'last.pt')
  if improved: torch.save(payload,Path(c['run_dir'])/'best.pt')
  if a.patience and bad_epochs>=a.patience: print('early_stop=true'); break
  if a.max_steps and global_step>=a.max_steps: break
  offset=0
if __name__=='__main__': main()
