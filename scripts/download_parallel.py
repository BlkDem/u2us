import argparse, json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description='Download a licensed parallel corpus via Hugging Face Datasets')
 p.add_argument('--dataset',required=True,help='HF dataset name, e.g. opus100')
 p.add_argument('--config',default=None); p.add_argument('--split',default='train'); p.add_argument('--out',required=True)
 p.add_argument('--source',default='ru'); p.add_argument('--target',default='en'); p.add_argument('--limit',type=int)
 p.add_argument('--src-key',default='translation'); a=p.parse_args()
 try: from datasets import load_dataset
 except ImportError: raise SystemExit('Install corpus dependencies: pip install datasets')
 kwargs={'path':a.dataset,'split':a.split}
 if a.config: kwargs['name']=a.config
 ds=load_dataset(**kwargs); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); n=0
 with out.open('w',encoding='utf8') as f:
  for row in ds:
   pair=row.get(a.src_key,row)
   if not isinstance(pair,dict) or not pair.get(a.source) or not pair.get(a.target): continue
   f.write(json.dumps({'src':pair[a.source],'tgt':pair[a.target],'source_dataset':a.dataset},ensure_ascii=False)+'\n'); n+=1
   if a.limit and n>=a.limit: break
 manifest={'dataset':a.dataset,'config':a.config,'split':a.split,'source':a.source,'target':a.target,'count':n}
 out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8'); print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
