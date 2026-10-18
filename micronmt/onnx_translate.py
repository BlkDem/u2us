import argparse, numpy as np, onnxruntime as ort
from .checkpoint import load_checkpoint

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--text',required=True);p.add_argument('--source',default='ru');p.add_argument('--sequence-len',type=int,default=128);a=p.parse_args()
 _,tok,_=load_checkpoint(a.checkpoint); n=a.sequence_len; tag='<ru2en>' if a.source=='ru' else '<en2ru'; ids=tok.encode(tag+' '+a.text,n); ids=ids[:n]; src=np.full((1,n),tok.pad_id,dtype=np.int64); src[0,:len(ids)]=ids
 enc=ort.InferenceSession(a.model_dir+'/encoder.onnx',providers=['CPUExecutionProvider']); dec=ort.InferenceSession(a.model_dir+'/decoder.onnx',providers=['CPUExecutionProvider']); memory=enc.run(['memory'],{'src':src})[0]
 tgt=np.full((1,n),tok.pad_id,dtype=np.int64);tgt[0,0]=tok.bos_id
 for i in range(1,n):
  logits=dec.run(['logits'],{'memory':memory,'tgt':tgt,'src':src})[0]; nxt=int(logits[0,i-1].argmax());tgt[0,i]=nxt
  if nxt==tok.eos_id:break
 print(tok.decode(tgt[0]))
if __name__=='__main__':main()
