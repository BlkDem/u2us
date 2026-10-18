import argparse,json,re
from pathlib import Path

def tokens(s): return re.findall(r'\w+',s.lower())
def signature(s):
 t=tokens(s); return (len(t)//3, t[0] if t else '', t[-1] if t else '')
def jaccard(a,b):
 a,b=set(tokens(a)),set(tokens(b)); return len(a&b)/max(1,len(a|b))
def difficulty(src,tgt):
 n=len(tokens(src))+len(tokens(tgt)); punct=sum(src.count(x)+tgt.count(x) for x in ',;:!?')
 return {'level':1 if n<16 else 2 if n<40 else 3 if n<80 else 4,'score':round(min(1,n/160+punct/40),4)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--near-threshold',type=float,default=.92);a=p.parse_args();rows=[]; buckets={}; rejected=0
 for l in Path(a.input).read_text(encoding='utf8').splitlines():
  try:r=json.loads(l); src=r['src']; tgt=r['tgt']
  except (json.JSONDecodeError,KeyError):continue
  candidates=buckets.get(signature(src),[])
  if any(jaccard(src,x['src'])>=a.near_threshold for x in candidates): rejected+=1;continue
  r['difficulty']=difficulty(src,tgt);rows.append(r);buckets.setdefault(signature(src),[]).append(r)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf8');print(json.dumps({'accepted':len(rows),'near_duplicates':rejected}))
if __name__=='__main__':main()
