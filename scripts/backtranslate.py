import json, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from micronmt.model import MicroTransformer
from micronmt.checkpoint import load_tokenizer_metadata

ckpt=Path('runs/realbaseline_128x2_50c_enru/best.pt')
saved=torch.load(ckpt,map_location='cpu',weights_only=True)
config=saved['config']
tok=load_tokenizer_metadata(saved['tokenizer'],'x')
model=MicroTransformer(tok.vocab_size, **{k:config[k] for k in ('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')}, init_mode=config.get('init_mode','controlled'))
model.load_state_dict(saved['model']); model.eval()
max_len=config['max_len']

rows=[json.loads(l) for l in Path('data/real_baseline/ru_en/train.jsonl').read_text().splitlines()]
# use English side (tgt) as source for EN->RU
texts=[r['tgt'] for r in rows[:4000]]
out=[]
for i,t in enumerate(texts):
    ids=torch.tensor([tok.encode('<en2ru> '+t,max_len)])
    with torch.no_grad():
        o=model.greedy(ids,max_len=max_len).tolist()[0]
    seq=o[1:]
    if 3 in seq: seq=seq[:seq.index(3)+1]
    hyp=tok.decode(seq)
    if hyp.strip() and len(hyp.split())>=2:
        out.append({'src':hyp,'tgt':t,'source':'backtranslation'})
    if (i+1)%500==0:
        print(i+1,len(out),flush=True)
Path('data/backtranslated_ru_en.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
print('done',len(out))
