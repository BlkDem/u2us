import argparse, json, subprocess, tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--source',default='ru');a=p.parse_args()
 with tempfile.TemporaryDirectory() as d:
  pred=Path(d)/'pred.jsonl'; subprocess.run(['python','-m','micronmt.predict','--checkpoint',a.checkpoint,'--input',a.data,'--output',str(pred),'--source',a.source],check=True)
  rows=[json.loads(x) for x in pred.read_text().splitlines()]; refs=[json.loads(x)['tgt'] for x in Path(a.data).read_text().splitlines()]
  exact=sum(x['tgt'].strip()==r.strip() for x,r in zip(rows,refs))/max(1,len(refs)); print(json.dumps({'examples':len(refs),'exact_match':exact}))
if __name__=='__main__':main()
