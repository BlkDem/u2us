import argparse, torch
from .model import MicroTransformer
from .tokenizer import SimpleTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 q=torch.load(a.checkpoint,map_location='cpu'); t=SimpleTokenizer(q['tokenizer']); c=q['config']; m=MicroTransformer(t.vocab_size,**{k:c[k] for k in ('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')});m.load_state_dict(q['model']);m.eval()
 src=torch.ones(1,8,dtype=torch.long); tgt=torch.ones(1,8,dtype=torch.long)
 torch.onnx.export(m,(src,tgt),a.out,input_names=['src','tgt'],output_names=['logits'],dynamic_axes={'src':{1:'src_len'},'tgt':{1:'tgt_len'},'logits':{1:'tgt_len'}},opset_version=17)
if __name__=='__main__':main()
