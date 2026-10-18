# Real-data baseline

The reproducible real-data path is:

```bash
python -m pip install --user -e ".[data,train]"
python scripts/download_parallel.py \
  --dataset Helsinki-NLP/opus-100 --config en-ru --split train \
  --source ru --target en --limit 100000 \
  --out data/raw/opus100_ru_en.jsonl
python scripts/quality_filter.py \
  --input data/raw/opus100_ru_en.jsonl \
  --output data/clean/opus100.jsonl \
  --report data/clean/opus100.json \
  --source ru --threshold .85
python scripts/advanced_filter.py \
  --input data/clean/opus100.jsonl \
  --output data/filtered/opus100.jsonl
python scripts/prepare_data.py \
  --input data/filtered/opus100.jsonl --out data/opus100
python scripts/train_tokenizer.py \
  --input data/opus100/train.jsonl \
  --prefix tokenizer/opus100_16k --vocab-size 16000
python -m micronmt.train \
  --config configs/tiny.yaml \
  --data data/opus100/train.jsonl \
  --valid data/opus100/valid.jsonl \
  --sp-model tokenizer/opus100_16k.model \
  --grad-accum 4 --amp
```

The download script uses `Helsinki-NLP/opus-100`, not the ambiguous short name `opus100`. The source dataset and license must be recorded in the resulting manifest before production use.

A smoke baseline was trained locally on a 5,000-example sample. After filtering it contained 4,583 pairs: 3,667 train, 458 validation and 458 test. A 16k SentencePiece tokenizer and 3-epoch tiny student trained successfully on CPU with AMP and gradient accumulation. Its validation loss reached 13.6331. This is an engineering smoke result, not a quality claim; a real benchmark requires a much larger corpus and held-out evaluation.
