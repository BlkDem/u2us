import argparse, torch, yaml
from .model import MicroTransformer
from .tokenizer import SimpleTokenizer

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--text',required=True); p.add_argument('--source',default='ru'); p.add_argument('--target',default='en'); a=p.parse_args()
 q=torch.load(a.checkpoint,map_location='cpu'); tok=SimpleTokenizer(q.get('tokenizer')); c=q['config']; m=MicroTransformer(tok.vocab_size,**{k:c[k] for k in ('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')}); m.load_state_dict(q['model']); m.eval()
 src=torch.tensor([tok.encode(('<ru2en>' if a.source=='ru' else '<en2ru>')+' '+a.text,c['max_len'])]); print(tok.decode(m.greedy(src)[0]))
if __name__=='__main__': main()
