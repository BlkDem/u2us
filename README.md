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
# bidirectional evaluation of a shared checkpoint
python scripts/evaluate_bidir.py \
  --checkpoint runs/shared/best.pt \
  --ru-en data/ru_en/test.jsonl \
  --en-ru data/en_ru/test.jsonl \
  --out reports/bidirectional.json
# distillation from a compatible teacher checkpoint
python -m micronmt.train --config configs/tiny.yaml --data data/ru_en/train.jsonl \
  --valid data/ru_en/valid.jsonl --teacher runs/teacher.pt
python -m micronmt.infer --checkpoint runs/tiny/best.pt --text 'Я решил остаться здесь.' --source ru --target en
```

Для реальных данных используйте JSONL `{ "src": "...", "tgt": "...", "source": "..." }`; test никогда не передавайте в train. `scripts/prepare_data.py` фильтрует, нормализует, deduplicates и делит корпус.

## OpenAI-совместимый API

`micronmt/serve.py` поднимает HTTP-сервер без дополнительных зависимостей (stdlib) с эндпоинтами `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/health` и SSE-стримингом:

```bash
python -m micronmt.serve --checkpoint runs/opus100_teacher_bt_en_ru/best.pt \
  --source en --beam-size 4 --length-penalty 0.6 --no-repeat-ngram-size 3
# опционально: --host 0.0.0.0 --port 8000 --model-name micronmt --api-key "$MICRONMT_API_KEY"
```

Направление задаётся именем модели (`ru2en`, `en2ru`, `micronmt-en-ru`) или полями `source`/`source_language` в запросе; по умолчанию используется `--source`.

```python
from openai import OpenAI
client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='not-needed')
response = client.chat.completions.create(
    model='en2ru',
    messages=[{'role': 'user', 'content': 'Hello world'}],
)
print(response.choices[0].message.content)
```

## Харнесс для eval

`micronmt/harness.py` прогоняет чекпоинт или запущенный HTTP API по JSONL, перебирает конфиги декодирования и считает BLEU/chrF (опционально COMET):

```bash
# sweep по локальному чекпоинту + сохранение гипотез и отчёта
python -m micronmt.harness --checkpoint runs/opus100_teacher_bt_en_ru/best.pt \
  --data data/opus100_en_ru/test.jsonl --source en \
  --beam-size 1 4 --length-penalty 0.6 --no-repeat-ngram-size 0 3 \
  --out reports/harness_en_ru.json --hyp-dir reports/hyp_en_ru

# тот же sweep через OpenAI-совместимый API
python -m micronmt.harness --api-url http://127.0.0.1:8000/v1 \
  --data data/opus100_en_ru/test.jsonl --source en --beam-size 1 4

# регрессионный гейт: не хуже baseline более чем на 0.5 пункта BLEU/chrF
python -m micronmt.harness --checkpoint runs/tiny/best.pt --data data/opus100/test.jsonl \
  --baseline reports/harness_en_ru.json --max-drop 0.5
```

Конфиги можно задать файлом (`--config decode.json` — JSON-список объектов с ключами `beam_size`, `length_penalty`, `no_repeat_ngram_size`, `repetition_penalty`). Отчёт содержит все конфиги, лучший по BLEU/chrF и результат гейта; код возврата 1 при провале гейта.

## Компоненты

- `micronmt/model.py` — компактный Transformer с shared embeddings, direction token и context support.
- `micronmt/tokenizer.py` — SentencePiece Unigram/BPE; fallback whitespace tokenizer для smoke tests.
- `micronmt/train.py` — CE + teacher soft-target KL, resume/full fine-tuning.
- `micronmt/distill.py` — combined hard-target and temperature-scaled KL loss.
- `micronmt/adapt.py` — lightweight LoRA-ready adaptation hook (требует PEFT для полноценного режима).
- `micronmt/terminology.py` — термины передаются как source-side control hints, без global replace.
- `micronmt/export.py` — ONNX export; production runtime не требует PyTorch.
- `micronmt/serve.py` — OpenAI-совместимый HTTP API поверх `Translator` (stdlib, без FastAPI).
- `micronmt/harness.py` — eval-харнесс: sweep конфигов декодирования для чекпоинта или API, BLEU/chrF/COMET, отчёт и регрессионный гейт.
- `micronmt/evaluate.py` — BLEU/chrF and optional COMET JSON report.
- `micronmt/benchmark.py` — CPU latency, throughput, parameter count and checkpoint size.

## Проверки и ограничения

```bash
python -m unittest discover -s tests -v
# pytest также запускает исходные function-style smoke tests
pip install -e '.[train,dev]'
python -m pytest -q
```

Новые checkpoints содержат tokenizer внутри файла. Для split ONNX используйте
`python -m micronmt.export_seq2seq --checkpoint runs/tiny/best.pt --out-dir artifacts/tiny-onnx`;
затем `python -m micronmt.onnx_translate --model-dir artifacts/tiny-onnx --text 'Hello' --source en`
не требует PyTorch checkpoint. Длина берётся из metadata экспорта.

`--resume` продолжает сохранённый прогресс, а `epochs` задаёт общее число эпох,
не число дополнительных. Для продолжения завершённого обучения увеличьте `epochs`.
`--max-steps` задаёт общий предел обработанных microbatches, включая сохранённые.
Старые LoRA-адаптеры с attention `out_proj` необходимо переобучить.

Regression gate требует явной политики, например:

```bash
python scripts/regression.py --checkpoint runs/tiny/best.pt --data data/regression.jsonl --min-exact-match 0.8
```

Порог выше — пример, не подтверждённая норма качества. Exact match не заменяет
BLEU/chrF/COMET и human evaluation. Текущие ограничения и требования к релизу — в `AUDIT.md`.

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

## Лицензия

Код проекта распространяется по лицензии MIT (см. `LICENSE`).

**Данные, токенизаторы и обученные веса — research-only.** OPUS-100 на зафиксированном revision имеет `license: unknown`, поэтому в манифестах стоит `production_approved: false`: права на распространение и commercial use весов не подтверждены. Backtranslation/synthetic данные дополнительно зависят от лицензии teacher-модели. Лицензия кода не распространяется на данные и веса; до выяснения прав не публикуйте веса и release-артефакты как готовый продукт.
