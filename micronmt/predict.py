import argparse, json, torch
from pathlib import Path
from .checkpoint import load_checkpoint

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--source',default='ru'); p.add_argument('--max-len',type=int,default=256); a=p.parse_args()
 m,tok,_=load_checkpoint(a.checkpoint); tag='<ru2en>' if a.source=='ru' else '<en2ru'; out=[]
 for line in Path(a.input).read_text(encoding='utf8').splitlines():
  row=json.loads(line); ids=torch.tensor([tok.encode(tag+' '+row['src'],a.max_len)]); pred=m.greedy(ids,max_len=a.max_len)[0]; out.append({'src':row['src'],'tgt':tok.decode(pred)})
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf8')
 print(f'wrote {len(out)} predictions to {a.output}')
if __name__=='__main__': main()
