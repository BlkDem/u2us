import argparse, json
from pathlib import Path

def corpus(path):
 rows=[]
 for line in Path(path).read_text(encoding='utf8').splitlines():
  try: rows.append(json.loads(line))
  except ValueError: pass
 return rows

def main():
 p=argparse.ArgumentParser(); p.add_argument('--reference',required=True); p.add_argument('--hypothesis',required=True); a=p.parse_args()
 refs=[x['tgt'] for x in corpus(a.reference)]; hyps=[x['tgt'] for x in corpus(a.hypothesis)]
 try:
  import sacrebleu
  print('BLEU',sacrebleu.corpus_bleu(hyps,[refs]).score)
  print('chrF',sacrebleu.corpus_chrf(hyps,[refs]).score)
 except ImportError: print('Install optional dependency: pip install sacrebleu')
if __name__=='__main__': main()
