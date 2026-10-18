import argparse
from pathlib import Path
from micronmt.sp_tokenizer import train_sentencepiece

p=argparse.ArgumentParser(); p.add_argument('--input', nargs='+', required=True)
p.add_argument('--prefix', required=True); p.add_argument('--vocab-size', type=int, default=16000)
a=p.parse_args(); Path(a.prefix).parent.mkdir(parents=True, exist_ok=True)
train_sentencepiece(a.input, a.prefix, a.vocab_size)
print(a.prefix + '.model')
