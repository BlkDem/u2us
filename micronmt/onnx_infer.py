import argparse, numpy as np, onnxruntime as ort
from .checkpoint import load_checkpoint

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--text',required=True);p.add_argument('--source',default='ru');p.add_argument('--sequence-len',type=int,default=128);a=p.parse_args()
 _,tok,_=load_checkpoint(a.checkpoint); tag='<ru2en>' if a.source=='ru' else '<en2ru'; ids=tok.encode(tag+' '+a.text,a.sequence_len); ids += [tok.pad_id]*(a.sequence_len-len(ids)); src=np.array([ids],dtype=np.int64); tgt=np.full((1,a.sequence_len),tok.pad_id,dtype=np.int64); tgt[0,0]=tok.bos_id
 session=ort.InferenceSession(a.model,providers=['CPUExecutionProvider']); print(session.run(['logits'],{'src':src,'tgt':tgt})[0].shape)
if __name__=='__main__':main()
