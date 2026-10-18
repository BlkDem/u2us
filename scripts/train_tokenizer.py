import argparse, json, tempfile
from pathlib import Path
from micronmt.sp_tokenizer import train_sentencepiece

p=argparse.ArgumentParser(); p.add_argument('--input', nargs='+', required=True)
p.add_argument('--prefix', required=True); p.add_argument('--vocab-size', type=int, default=16000)
a=p.parse_args(); Path(a.prefix).parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as d:
    text_files=[]
    for i, source in enumerate(a.input):
        out=Path(d)/f'{i}.txt'
        with out.open('w', encoding='utf-8') as dst:
            for line in Path(source).read_text(encoding='utf-8').splitlines():
                try:
                    row=json.loads(line)
                    for key in ('src','tgt'):
                        if row.get(key): dst.write(str(row[key]).replace('\n',' ')+'\n')
                except json.JSONDecodeError:
                    if line.strip(): dst.write(line.strip()+'\n')
        text_files.append(out)
    train_sentencepiece(text_files, a.prefix, a.vocab_size)
print(a.prefix + '.model')
