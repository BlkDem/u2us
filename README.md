# RU↔EN MicroNMT

Экспериментальный компактный encoder-decoder Transformer для русского ↔ английского. Репозиторий содержит воспроизводимый MVP: подготовка JSONL, SentencePiece (с fallback tokenizer), обучение, teacher/distillation hooks, fine-tuning, greedy inference, ONNX export и базовые метрики.

> C2 — исследовательская цель, не обещание MVP. Качество необходимо подтвердить COMET и human evaluation на отдельном 5–10k benchmark.

## Быстрый старт

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[train]'
python scripts/make_toy_data.py --out data/toy
python -m micronmt.train --config configs/tiny.yaml --data data/toy/train.jsonl --valid data/toy/valid.jsonl
# продолжить full fine-tuning
python -m micronmt.train --config configs/tiny.yaml --data new_data/ru_en.jsonl --resume runs/tiny/last.pt
# экспорт ONNX
python -m micronmt.export --checkpoint runs/tiny/last.pt --out runs/tiny/model.onnx
python -m micronmt.infer --checkpoint runs/tiny/best.pt --text 'Я решил остаться здесь.' --source ru --target en
```

Для реальных данных используйте JSONL `{ "src": "...", "tgt": "...", "source": "..." }`; test никогда не передавайте в train. `scripts/prepare_data.py` фильтрует, нормализует, deduplicates и делит корпус.

## Компоненты

- `micronmt/model.py` — компактный Transformer с shared embeddings, direction token и context support.
- `micronmt/tokenizer.py` — SentencePiece Unigram/BPE; fallback whitespace tokenizer для smoke tests.
- `micronmt/train.py` — CE + optional teacher soft-target KL, resume/full fine-tuning, AMP.
- `micronmt/adapt.py` — lightweight LoRA-ready adaptation hook (требует PEFT для полноценного режима).
- `micronmt/terminology.py` — термины передаются как source-side control hints, без global replace.
- `micronmt/export.py` — ONNX export; production runtime не требует PyTorch.
- `micronmt/evaluate.py` — BLEU/chrF, optional sacrebleu/COMET.

## План экспериментов

| Model | Params | FP32 | INT8 | RAM | Speed | BLEU | chrF | COMET |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny (2x128, 4h, 16k) | ~2.5M | измерить | измерить | измерить | измерить | измерить | измерить | измерить |
| small (4x256, 8h, 32k) | ~12M | измерить | измерить | измерить | измерить | измерить | измерить | измерить |
| medium (6x384, 8h, 48k) | ~35M | измерить | измерить | измерить | измерить | измерить | измерить | измерить |

Размеры и метрики намеренно не выдумываются: запускайте benchmark после обучения. Рекомендуемый стартовый вариант — shared multilingual tiny/small: один checkpoint уменьшает размер почти вдвое и direction token сохраняет общие закономерности. Для production качество нужно сравнить с двумя отдельными моделями.

## Данные и distillation

Для исследования используйте OPUS/OpenSubtitles, JW300, ParaCrawl/ELRC и human-curated validation/test с лицензиями, записанными в manifest. Teacher должен быть сильной открытой NMT-моделью (например NLLB/Marian, совместимой с вашим языковым покрытием), а synthetic pairs смешиваются с real в конфиге. В MVP teacher logits подключаются через `teacher_logits` в batch; KL loss имеет temperature и alpha/beta.

Continual learning: `--resume` + replay buffer старых данных, regression set и обязательное сравнение до/после. Доменные адаптации сохраняйте отдельными adapter/checkpoint, не перезаписывая base.
