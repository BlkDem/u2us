import argparse,json,math,re
from pathlib import Path

def lexical(src,tgt):
 # Deterministic fallback: shared numerals/names and length plausibility, never claims semantic quality.
 a=set(re.findall(r'\w+',src.lower()));b=set(re.findall(r'\w+',tgt.lower())); nums={x for x in a|b if x.isdigit()};
 return max(.0, min(1., .5 + .1*len(nums) - abs(math.log(max(1,len(src))/max(1,len(tgt))))*.15))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);p.add_argument('--threshold',type=float,default=.35);p.add_argument('--model',help='sentence-transformers model; optional');a=p.parse_args(); rows=[]; rejected=0; mode='lexical'
 embed=None
 if a.model:
  try:
   from sentence_transformers import SentenceTransformer,util
   embed=SentenceTransformer(a.model);mode='embedding'
  except ImportError: print('sentence-transformers unavailable; using lexical fallback')
 for l in Path(a.input).read_text(encoding='utf8').splitlines():
  try: r=json.loads(l)
  except json.JSONDecodeError: continue
  if embed: score=float(util.cos_sim(embed.encode(r['src'],convert_to_tensor=True),embed.encode(r['tgt'],convert_to_tensor=True)).item())
  else: score=lexical(r['src'],r['tgt'])
  r['semantic_score']=round(score,5)
  if score<a.threshold: rejected+=1
  else: rows.append(r)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf8');Path(a.report).write_text(json.dumps({'mode':mode,'threshold':a.threshold,'accepted':len(rows),'rejected':rejected},indent=2),encoding='utf8')
 print(json.dumps({'mode':mode,'accepted':len(rows),'rejected':rejected}))
if __name__=='__main__':main()
