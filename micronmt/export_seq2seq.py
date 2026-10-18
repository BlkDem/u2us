import argparse, json
from pathlib import Path
import torch
from torch import nn
from .checkpoint import load_checkpoint

class EncoderWrapper(nn.Module):
 def __init__(self,m): super().__init__(); self.m=m
 def forward(self,src):
  mask=src.eq(0); x=self.m.add_pos(self.m.embedding(src)*self.m.d_model**.5)
  return self.m.encoder(x,src_key_padding_mask=mask)

class DecoderWrapper(nn.Module):
 def __init__(self,m): super().__init__(); self.m=m
 def forward(self,memory,tgt,src):
  src_key=src.eq(0); tgt_key=tgt.eq(0); causal=nn.Transformer.generate_square_subsequent_mask(tgt.size(1),device=tgt.device)
  x=self.m.add_pos(self.m.embedding(tgt)*self.m.d_model**.5)
  out=self.m.decoder(x,memory,tgt_mask=causal,tgt_key_padding_mask=tgt_key,memory_key_padding_mask=src_key)
  return self.m.output(self.m.norm(out))

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--sequence-len',type=int,default=128);a=p.parse_args()
 m,tok,q=load_checkpoint(a.checkpoint); out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True); n=a.sequence_len
 src=torch.ones(1,n,dtype=torch.long); tgt=torch.ones(1,n,dtype=torch.long); memory=EncoderWrapper(m)(src)
 torch.onnx.export(EncoderWrapper(m),(src,),out/'encoder.onnx',input_names=['src'],output_names=['memory'],opset_version=17,dynamo=False)
 torch.onnx.export(DecoderWrapper(m),(memory,tgt,src),out/'decoder.onnx',input_names=['memory','tgt','src'],output_names=['logits'],opset_version=17,dynamo=False)
 (out/'metadata.json').write_text(json.dumps({'format':'split-seq2seq','sequence_len':n,'checkpoint':str(a.checkpoint),'tokenizer':q.get('tokenizer')},ensure_ascii=False,indent=2),encoding='utf8')
 print(out)
if __name__=='__main__':main()
