import argparse, json, sys, torch
from pathlib import Path
sys.path.insert(0, '.')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

p=argparse.ArgumentParser()
p.add_argument('--input',required=True); p.add_argument('--output',required=True)
p.add_argument('--model',default='Helsinki-NLP/opus-mt-ru-en')
p.add_argument('--limit',type=int,default=0); p.add_argument('--max-length',type=int,default=128); p.add_argument('--beam-size',type=int,default=4)
p.add_argument('--batch-size',type=int,default=16); p.add_argument('--flush-every',type=int,default=2000)
a=p.parse_args()
if a.limit<0 or a.max_length<3 or a.beam_size<1 or a.batch_size<1 or a.flush_every<1: p.error('invalid limit, length, beam, batch or flush')
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'teacher={a.model} device={device} batch={a.batch_size} beam={a.beam_size}',flush=True)
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForSeq2SeqLM.from_pretrained(a.model,use_safetensors=True).to(device); model.eval()
rows=[json.loads(l) for l in Path(a.input).read_text(encoding='utf8').splitlines()]
if a.limit: rows=rows[:a.limit]
out_path=Path(a.output); out_path.parent.mkdir(parents=True,exist_ok=True)
kept=0; buf=[]; done=0
with out_path.open('w',encoding='utf8') as f:
 for i in range(0,len(rows),a.batch_size):
  batch=rows[i:i+a.batch_size]
  inputs=tok([r['src'] for r in batch],return_tensors='pt',padding=True,truncation=True,max_length=a.max_length).to(device)
  with torch.no_grad():
   gens=model.generate(**inputs,max_length=a.max_length,num_beams=a.beam_size)
  for r,g in zip(batch,gens):
   hyp=tok.decode(g,skip_special_tokens=True)
   if hyp.strip():
    buf.append(json.dumps({'src':r['src'],'tgt':hyp,'source':'teacher'},ensure_ascii=False)+'\n'); kept+=1
  done=i+len(batch)
  if len(buf)>=a.flush_every:
   f.write(''.join(buf)); f.flush(); buf=[]; print(done,kept,flush=True)
 f.write(''.join(buf)); f.flush()
print('done',kept,'->',str(out_path))
