import argparse, torch
from .checkpoint import load_checkpoint
from .lora import inject_lora, load_adapter

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--adapter',required=True);p.add_argument('--text',required=True);p.add_argument('--source',default='ru');a=p.parse_args()
 m,tok,_=load_checkpoint(a.checkpoint); q=torch.load(a.adapter,map_location='cpu'); inject_lora(m,q['rank']); load_adapter(m,q['adapter']); m.eval(); tag='<ru2en>' if a.source=='ru' else '<en2ru'; src=torch.tensor([tok.encode(tag+' '+a.text)]); print(tok.decode(m.greedy(src)[0]))
if __name__=='__main__':main()
