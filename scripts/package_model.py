import argparse, hashlib, json, shutil, torch
from pathlib import Path

def sha(x): return hashlib.sha256(x.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--out',required=True);p.add_argument('--model-name',default='micronmt');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cp=Path(a.checkpoint);sp=Path(a.tokenizer)
 q=torch.load(cp,map_location='cpu'); meta=q.get('tokenizer',{}); 
 if meta.get('type')=='sentencepiece': meta['model_file']='tokenizer.model'; q['tokenizer']=meta
 torch.save(q,out/'model.pt'); shutil.copy2(sp,out/'tokenizer.model')
 info={'model_name':a.model_name,'checkpoint_sha256':sha(cp),'packaged_model_sha256':sha(out/'model.pt'),'tokenizer_sha256':sha(sp),'files':['model.pt','tokenizer.model'],'format':'pytorch-checkpoint'};(out/'metadata.json').write_text(json.dumps(info,indent=2),encoding='utf8');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
