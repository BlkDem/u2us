import argparse, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description='Download a licensed parallel corpus via Hugging Face Datasets')
 p.add_argument('--dataset',required=True,help='HF dataset name, e.g. Helsinki-NLP/opus-100')
 p.add_argument('--config',default=None); p.add_argument('--split',default='train'); p.add_argument('--out',required=True)
 p.add_argument('--source',default='ru'); p.add_argument('--target',default='en'); p.add_argument('--limit',type=int)
 p.add_argument('--revision',required=True,help='Pinned dataset commit SHA')
 p.add_argument('--license',required=True,help='Verified license identifier, or unknown for research-only provenance')
 p.add_argument('--src-key',default='translation'); a=p.parse_args()
 if a.limit is not None and a.limit < 1: p.error('limit must be positive')
 if a.source == a.target: p.error('source and target must differ')
 if Path(a.out).exists(): p.error('output already exists')
 try: from datasets import load_dataset
 except ImportError: raise SystemExit('Install corpus dependencies: pip install datasets')
 kwargs={'path':a.dataset,'split':a.split,'revision':a.revision,'streaming':True}
 if a.config: kwargs['name']=a.config
 ds=load_dataset(**kwargs); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); n=0
 with out.open('w',encoding='utf8') as f:
  for row in ds:
   pair=row.get(a.src_key,row)
   if not isinstance(pair,dict) or any(not isinstance(pair.get(k),str) or not pair[k].strip() for k in (a.source,a.target)): continue
   f.write(json.dumps({'src':pair[a.source],'tgt':pair[a.target],'source_dataset':a.dataset},ensure_ascii=False)+'\n'); n+=1
   if a.limit and n>=a.limit: break
 import hashlib
 manifest={'dataset':a.dataset,'revision':a.revision,'license':a.license,'production_approved':False,'config':a.config,'split':a.split,'source':a.source,'target':a.target,'count':n,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8'); print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
