import json, sys, torch
from pathlib import Path
sys.path.insert(0,'.')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok=AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
model=AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-ru-en')
model.eval()

rows=[json.loads(l) for l in Path('data/real_baseline/ru_en/train.jsonl').read_text().splitlines()]
out=[]
for i,r in enumerate(rows):
    src=r['src']
    inputs=tok(src, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        gen=model.generate(**inputs, max_length=128, num_beams=4)
    hyp=tok.decode(gen[0], skip_special_tokens=True)
    if hyp.strip():
        out.append({'src':src,'tgt':hyp,'source':'teacher_ru_en'})
    if (i+1)%500==0:
        print(i+1,len(out),flush=True)
Path('data/teacher_ru_en.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
print('done',len(out))
