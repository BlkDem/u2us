import argparse, hashlib, html, json, math, re, unicodedata
from pathlib import Path

HTML=re.compile(r'<[^>]+>'); URL=re.compile(r'https?://|www\.',re.I); REPEAT=re.compile(r'\b(\w+)(?:\s+\1){2,}\b',re.I)
CYR=re.compile(r'[А-Яа-яЁё]'); LAT=re.compile(r'[A-Za-z]')

def norm(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',html.unescape(s))).strip()
def lang_score(s, lang):
 c=sum(ch.isalpha() for ch in s); return 0 if not c else (len(CYR.findall(s))/c if lang=='ru' else len(LAT.findall(s))/c)
def score(src,tgt,sl,tl):
 checks=[bool(src and tgt), not HTML.search(src+tgt), not REPEAT.search(src+' '+tgt), len(src)>=2 and len(tgt)>=2, max(len(src),len(tgt))/max(1,min(len(src),len(tgt)))<=3.5, lang_score(src,sl)>=.15, lang_score(tgt,tl)>=.15]
 return sum(checks)/len(checks)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--source',choices=['ru','en'],default='ru');p.add_argument('--threshold',type=float,default=.85);p.add_argument('--report',required=True);a=p.parse_args(); target='en' if a.source=='ru' else 'ru'; seen=set(); out=[]; stats={'read':0,'accepted':0,'duplicates':0,'rejected':0}
 if not math.isfinite(a.threshold) or not 0 <= a.threshold <= 1: p.error('threshold must be finite and between 0 and 1')
 for number,line in enumerate(Path(a.input).read_text(encoding='utf8').splitlines(),1):
  stats['read']+=1
  try:r=json.loads(line)
  except json.JSONDecodeError as exc: raise ValueError(f'{a.input}:{number}: invalid JSON') from exc
  if not isinstance(r,dict) or any(not isinstance(r.get(k),str) for k in ('src','tgt')): raise ValueError(f'{a.input}:{number}: expected string src and tgt')
  src=norm(r['src']);tgt=norm(r['tgt'])
  if not src or not tgt: stats['rejected']+=1;continue
  key=hashlib.sha1((src.lower()+'\0'+tgt.lower()).encode()).hexdigest()
  if key in seen:stats['duplicates']+=1;continue
  seen.add(key); q=score(src,tgt,a.source,target)
  if q<a.threshold:stats['rejected']+=1;continue
  out.append({'src':src,'tgt':tgt,'quality_score':round(q,4)});stats['accepted']+=1
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf8');Path(a.report).write_text(json.dumps(stats,indent=2),encoding='utf8');print(json.dumps(stats))
if __name__=='__main__':main()
