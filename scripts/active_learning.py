import argparse,json,math
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=100);a=p.parse_args()
 # Ranking scaffold: prioritize long/rare-looking inputs without pretending to have calibrated confidence.
 rows=[]
 for l in Path(a.input).read_text(encoding='utf8').splitlines():
  try:r=json.loads(l); text=r.get('src',''); score=len(text.split()) + 2*sum(ch.isdigit() for ch in text); rows.append((score,r))
  except json.JSONDecodeError:pass
 rows.sort(reverse=True,key=lambda x:x[0]); out=[dict(r,review_priority=s) for s,r in rows[:a.limit]];Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in out)+'\n',encoding='utf8');print(f'wrote {len(out)} review items')
if __name__=='__main__':main()
