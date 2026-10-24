import argparse, json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from micronmt.model import MicroTransformer
from micronmt.checkpoint import load_tokenizer_metadata

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
p.add_argument('--direction',choices=['ru2en','en2ru'],default='en2ru')
p.add_argument('--limit',type=int,default=0); p.add_argument('--min-words',type=int,default=2)
a=p.parse_args()
if a.limit<0 or a.min_words<1: p.error('invalid limit or min-words')
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
config=saved['config']; tok=load_tokenizer_metadata(saved['tokenizer'],a.checkpoint)
model=MicroTransformer(tok.vocab_size, **{k:config[k] for k in ('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')}, init_mode=config.get('init_mode','controlled'))
model.load_state_dict(saved['model']); model.eval()
max_len=config['max_len']; tag='<en2ru>' if a.direction=='en2ru' else '<ru2en>'
rows=[json.loads(l) for l in Path(a.input).read_text(encoding='utf8').splitlines()]
if a.limit: rows=rows[:a.limit]
out=[]
for i,r in enumerate(rows):
 text=r['tgt'] if a.direction=='en2ru' else r['src']
 ids=torch.tensor([tok.encode(tag+' '+text,max_len)])
 with torch.no_grad():
  o=model.greedy(ids,max_len=max_len).tolist()[0]
 seq=o[1:]
 if 3 in seq: seq=seq[:seq.index(3)+1]
 hyp=tok.decode(seq)
 if hyp.strip() and len(hyp.split())>=a.min_words:
  out.append({'src':hyp,'tgt':text,'source':'backtranslation'} if a.direction=='en2ru' else {'src':text,'tgt':hyp,'source':'backtranslation'})
 if (i+1)%1000==0: print(i+1,len(out),flush=True)
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf8')
print('done',len(out),'->',a.output)
