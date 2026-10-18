import argparse, json
from pathlib import Path

def corpus(path):
 rows=[]
 for line in Path(path).read_text(encoding='utf8').splitlines():
  try: rows.append(json.loads(line))
  except json.JSONDecodeError: pass
 return rows

def main():
 p=argparse.ArgumentParser(); p.add_argument('--reference',required=True); p.add_argument('--hypothesis',required=True); p.add_argument('--out'); p.add_argument('--comet-model'); a=p.parse_args()
 refs=[x['tgt'] for x in corpus(a.reference)]; hyps=[x['tgt'] for x in corpus(a.hypothesis)]; result={'examples':min(len(refs),len(hyps))}
 try:
  import sacrebleu
  result['BLEU']=sacrebleu.corpus_bleu(hyps,[refs]).score
  result['chrF']=sacrebleu.corpus_chrf(hyps,[refs]).score
 except ImportError: result['sacrebleu']='unavailable'
 if a.comet_model:
  try:
   from comet import download_model, load_from_checkpoint
   model_path=download_model(a.comet_model); model=load_from_checkpoint(model_path)
   data=[{'src':r.get('src',''),'mt':h.get('tgt',''),'ref':r.get('tgt','')} for r,h in zip(corpus(a.reference),corpus(a.hypothesis))]
   result['COMET']=float(model.predict(data,gpus=0)['system_score'])
  except Exception as exc: result['COMET_error']=str(exc)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if a.out: Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__': main()
