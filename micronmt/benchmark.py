import argparse, json, os, time
from pathlib import Path
import torch
from .checkpoint import load_checkpoint

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--text',default='Я решил остаться здесь.'); p.add_argument('--source',choices=['ru','en'],default='ru'); p.add_argument('--runs',type=int,default=10); p.add_argument('--out'); a=p.parse_args()
 if a.runs < 1: p.error('runs must be positive')
 model,tok,meta=load_checkpoint(a.checkpoint); tag='<ru2en>' if a.source=='ru' else '<en2ru'; src=torch.tensor([tok.encode(tag+' '+a.text,meta['config']['max_len'])]);
 for _ in range(2): model.greedy_cached(src,max_len=meta['config']['max_len'])
 start=time.perf_counter()
 for _ in range(a.runs): model.greedy_cached(src,max_len=meta['config']['max_len'])
 elapsed=(time.perf_counter()-start)/a.runs
 result={'checkpoint':a.checkpoint,'params':sum(p.numel() for p in model.parameters()),'file_bytes':Path(a.checkpoint).stat().st_size,'latency_ms':elapsed*1000,'sentences_per_second':1/elapsed,'torch_threads':torch.get_num_threads()}
 print(json.dumps(result,indent=2))
 if a.out:
  Path(a.out).parent.mkdir(parents=True,exist_ok=True)
  Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
