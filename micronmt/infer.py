import argparse, torch
from .checkpoint import load_checkpoint

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--text',required=True); p.add_argument('--source',default='ru'); p.add_argument('--target',default='en'); p.add_argument('--max-len',type=int,default=256); a=p.parse_args()
 m,tok,_=load_checkpoint(a.checkpoint); tag='<ru2en>' if a.source=='ru' else '<en2ru'; src=torch.tensor([tok.encode(tag+' '+a.text,a.max_len)]); print(tok.decode(m.greedy(src,max_len=a.max_len)[0]))
if __name__=='__main__': main()
