import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--max-len',type=int,default=256);a=p.parse_args();lengths=[]
 for l in Path(a.input).read_text(encoding='utf8').splitlines():
  try:r=json.loads(l);lengths.append(min(a.max_len,len(r['src'].split())+2)+min(a.max_len,len(r['tgt'].split())+2))
  except (ValueError,KeyError):pass
 if lengths: print(json.dumps({'examples':len(lengths),'mean_tokens':sum(lengths)/len(lengths),'p95':sorted(lengths)[int(.95*len(lengths))-1],'max':max(lengths)}))
if __name__=='__main__':main()
