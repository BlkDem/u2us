import argparse, json
from .api import Translator, load_terminology

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--text',required=True); p.add_argument('--source',default='ru'); p.add_argument('--target'); p.add_argument('--max-len',type=int); p.add_argument('--context'); p.add_argument('--terms'); p.add_argument('--json',action='store_true'); a=p.parse_args()
 tr=Translator(a.checkpoint,max_len=a.max_len,terminology=load_terminology(a.terms) if a.terms else None)
 result=tr.translate(a.text,a.source,a.target,a.context,a.max_len)
 print(json.dumps(result,ensure_ascii=False) if a.json else result['translation'])
if __name__=='__main__': main()
