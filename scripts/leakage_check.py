import argparse,json,re
from pathlib import Path
def keys(path):
 out=set()
 for l in Path(path).read_text(encoding='utf8').splitlines():
  try:r=json.loads(l);out.add(re.sub(r'\s+',' ',r['src']).strip().lower())
  except (ValueError,KeyError):pass
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--valid',required=True);p.add_argument('--test',required=True);a=p.parse_args(); sets={x:keys(getattr(a,x)) for x in ('train','valid','test')}; result={'train_valid':len(sets['train']&sets['valid']),'train_test':len(sets['train']&sets['test']),'valid_test':len(sets['valid']&sets['test'])};print(json.dumps(result,indent=2));raise SystemExit(1 if any(result.values()) else 0)
if __name__=='__main__':main()
