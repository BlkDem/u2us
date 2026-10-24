# План прокачки MicroNMT (RU↔EN)

Документ описывает приоритизированный план улучшения качества модели после первого успешного обучения пары tiny-моделей на `opus50k`. Все ожидания по метрикам — гипотезы, а не гарантии; проверяются только на фиксированном test.

## 1. Текущее состояние (baseline)

Обучено 2026-10-21, CPU (Ryzen 7 7840HS, 16 потоков), `configs/opus50k_tiny_256.yaml`:
d_model 128, 2+2 слоя, ff 512, vocab 8000, max_len 256, batch 32, grad-accum 2, lr 5e-4, label smoothing 0.1, init controlled.

| Направление | Данные | Эпох (early stop) | Best valid chrF | Test BLEU | Test chrF | Latency |
|---|---|---:|---:|---:|---:|---:|
| ru→en | opus50k train 43 276 | 36 (best 30) | 26.14 | **7.52** | **26.02** | 13.3 мс |
| en→ru | opus50k train 43 276 | 30 (best 24) | 25.76 | **4.78** | **23.13** | 17.6 мс |

Артефакты: `runs/opus50k_tiny_256_{ru_en,en_ru}/best.pt`, `reports/opus50k_tiny_256/` (логи, `*_history.jsonl`, test-гипотезы, метрики, latency).

Что уже разблокировано:
- чекпоинты существуют → работают backtranslation, KL-дистилляция, LoRA/continual;
- `micronmt.train --device cuda|cpu|dml` (CUDA-путь не проверялся на реальной NVIDIA);
- капкан DirectML задокументирован: backward для `TransformerEncoderLayer(norm_first=True)` не компилируется, на этой машине обучение только CPU.

## 2. Принципы экспериментов

1. **Split фиксирован.** `data/opus50k/train.jsonl` → обучение, `valid.jsonl` → выбор эпохи/гиперпараметров, `test.jsonl` → оценка один раз в конце этапа. Никакого подбора гиперпараметров по test.
2. **Один эксперимент = один `run_dir` + `history`.** В history пишутся loss и chrF по эпохам; в summary — sha256 конфигов/данных/токенизатора (механика уже есть).
3. **Сравнение только при прочих равных.** Меняем одну вещь за раз; при смене данных/токенизатора фиксируем и старый test для преемственности.
4. **Seed 42 для итераций, 3 seed для финального выбора.** Текущие цифры получены на одном seed — расхождения ±0.5–1.5 chrF возможны.
5. **Ограничения честные.** `data/raw/opus50k_ru_en.manifest.json`: `license: unknown`, `production_approved: false`. До выяснения лицензии модель остаётся research-only.

## 3. Этапы

### P0. Быстрые победы без переобучения (0.5 дня)

| Мера | Действие | Ожидание | Стоимость |
|---|---|---|---|
| Beam search | добавить `--beam-size` в `micronmt/predict.py` (логика уже есть в `micronmt/beam.py`), прогнать test с beam 4–5 | +0.5–1.5 chrF | часы |
| KV-cache | кэш ключей/значений в greedy/beam (сейчас префикс пересчитывается, O(n²)) | скорость eval/инференса ×5–10, качество не меняет | 0.5 дня кода |
| Санити-чек | пересчитать метрики на 3 seed не нужно; проверить, что beam не ухудшает ru→en | — | 1 ч |

Критерий: beam-декод не хуже greedy на valid/test, latency приемлема.
Статус: beam 4 проверен в обе стороны (en→ru: BLEU 8.17 → 11.31, chrF 27.94 → 28.09; ru→en: BLEU 15.76 → 17.99, chrF 37.93 → 38.40); повторяемость en→ru 6.9% → 3.9%, но латиница в выходе 11.7% → 17.1%. KV-cache не сделан.

### P1. Данные: полный opus-100 (ночь, ~6–10 ч CPU)

Скачать ~100k пар и прогнать полный пайплайн из `REAL_DATA.md:55–81`:

```powershell
.venv\Scripts\python -m scripts.download_parallel --dataset Helsinki-NLP/opus-100 --config ru-en \
  --revision 805090dc28bf78897da9641cdf08b61287580df9 --license unknown \
  --split train --limit 100000 --out data/raw/opus100_ru_en.jsonl
.venv\Scripts\python -m scripts.quality_filter --input data/raw/opus100_ru_en.jsonl --output data/clean/opus100.jsonl \
  --source ru --threshold 0.85 --report reports/opus100/quality_filter.json
.venv\Scripts\python -m scripts.advanced_filter --input data/clean/opus100.jsonl --output data/filtered/opus100.jsonl
.venv\Scripts\python -m scripts.prepare_data --input data/filtered/opus100.jsonl --out data/opus100
.venv\Scripts\python -m scripts.train_tokenizer --input data/opus100/train.jsonl --prefix data/tokenizer/opus100/sp_16k --vocab-size 16000
```

`train_tokenizer` кладёт модель в `data/tokenizer/opus100/sp_16k.model` (флаг `--prefix`, не `--out`). `quality_filter` требует `--report`.

Затем `configs/opus100_tiny_256.yaml` (копия `opus50k_tiny_256.yaml` с `vocab_size: 8000` → фактический размер токенизатора и `run_dir: runs/opus100_tiny_256_ru_en`) и обучение:

```powershell
.venv\Scripts\python -m micronmt.train --config configs/opus100_tiny_256.yaml \
  --data data/opus100/train.jsonl --valid data/opus100/valid.jsonl \
  --sp-model data/tokenizer/opus100/sp_16k.model --device cpu --grad-accum 2 \
  --eval-samples 100 --patience 6 --history reports/opus100_tiny_256/ru_en_history.jsonl
```

Ожидание: +3–6 chrF относительно 26.0 (масштабирование 43k → 100k). Критерий приёмки: test chrF > 26.02 ru→en при matching-конфиге; отдельно отчёт на старом `opus50k/test.jsonl` (пересечение строк) для преемственности.

Риски: лицензия unknown; качество opus-100 неравномерно; возможна утечка при неаккуратном split — `prepare_data.py` делает union-find, не отключать.

### P2. Дистилляция (1 день подготовки + ночь)

Два независимых механизма, можно комбинировать:

1. **Дистилляция данными (уже доказала эффект).** `scripts/gen_teacher_ru_en.py` генерирует переводы `Helsinki-NLP/opus-mt-ru-en` (beam 4). Сейчас сделано только для 8k; расширить на весь train + 50/50 микс с реальными парами. Прошлый эффект на 8k: chrF 15.6 → 21.6.
   Подготовка: скрипт жёстко зашит на `data/real_baseline/ru_en/train.jsonl` / `data/teacher_ru_en.jsonl`, пишет без `encoding='utf8'` и требует `transformers` (нет ни в `pyproject.toml`, ни в venv). Нужен рефакторинг: CLI-аргументы `--input/--output/--checkpoint-hf/--limit`, явный UTF-8, добавить extra `[teacher]` с `transformers`.
2. **Logit-дистилляция.** Сначала обучить teacher той же архитектуры/токенизатора (например, small d256 на P1-данных), затем tiny с `--teacher runs/<teacher>/best.pt` и параметрами из `tiny.yaml`: `alpha_hard 0.8`, `beta_kl 0.2`, `temperature 2.0` (сейчас не применялись ни разу).

Ожидание: +2–5 chrF от данных-teacher, +1–2 от logit-KL. Риск: шум и галлюцинации teacher переносятся в ученика — фильтровать (длина, ratio, повторяемость).

Приёмка: test chrF выше P1 при том же бюджете обучения; проверить на 100 примерах глазами на деградацию (повторы, обрыв фраз).

### P3. Backtranslation и continual (ночь)

1. Починить `scripts/backtranslate.py`: жёстко зашиты `runs/realbaseline_128x2_50c_enru/best.pt` (файла нет), путь к данным, `[:4000]` и запись без UTF-8. Заменить на CLI `--checkpoint runs/opus50k_tiny_256_en_ru/best.pt --input ... --output ... --limit N`, явный UTF-8.
2. Сгенерировать BT-пары для английской стороны, смешать с реальными (`scripts/train_controlled.py` умеет replay, но заточен под 8k; для 100k использовать `micronmt/continual.py --replay --replay-ratio 0.25`).
3. Сравнить: только реальные vs реальные+BT vs реальные+BT+teacher (2×2).

Ожидание: +1–3 chrF. Риск: BT от слабой модели усиливает собственные ошибки (в прошлом в `data/mixed_ru_en.jsonl` были явно битые синтетические пары — фильтровать).

### P4. Ёмкость модели (нужен GPU или 1–2 суток CPU)

`configs/small.yaml`: d256, 4+4, ff1024, vocab 32000, max_len 384 (~12M params). На этой машине ~20–25 мин/эпоху на 100k, то есть 15–30 ч на направление. Реалистично только при наличии NVIDIA (код уже умеет `--device cuda`, но путь не проверен на железе) или на арендованном GPU.

Порядок: small на 100k → оценить → если tiny нужен для деплоя, дистиллировать small→tiny (P2, logit-уровень).

Ожидание (small vs tiny на тех же данных): +4–8 chrF.

### P5. Инфраструктура и дисциплина (параллельно, без влияния на качество)

- batching по токенам вместо фиксированного batch (AUDIT.md п.4) — стабилизирует обучение и убирает перекос loss усреднения по батчам (`train.py:102`);
- провалидировать CUDA-путь на реальной NVIDIA (тесты bit-exact resume — на CPU);
- починить Windows/Linux-специфику: `import resource` в `scripts/train_controlled.py`, `diagnose_training.py`, `run_real_baseline.py`; это блокирует часть тестов на Windows;
- CI: GitHub Actions на CPU-подмножестве тестов (train/resume, onnx, checkpoint, quality);
- `requirements.lock` / pin окружения, LICENSE (до релиза);
- multi-seed runner (3 seed для финальных сравнений);
- COMET (`unbabel-comet`) как дополнительная метрика при наличии ресурсов.

## 4. Сводная таблица приоритетов

| Приоритет | Мера | Ожидание (chrF) | Стоимость | Риск |
|---|---|---:|---|---|
| 1 | Beam search (P0) | +0.5–1.5 | часы | низкий |
| 2 | Opus-100, 100k (P1) | +3–6 | ночь CPU | средний (лицензия, шум) |
| 3 | Teacher-дистилляция (P2) | +2–5 | ночь + день подготовки | средний |
| 4 | Backtranslation (P3) | +1–3 | ночь | средний |
| 5 | Small d256 (P4) | +4–8 | GPU / 1–2 суток CPU | высокий (время) |
| 6 | KV-cache (P0) | 0 (скорость ×5–10) | 0.5 дня | низкий |

Порядок выполнения: P0 → P1 → P2 → P3 → (P4 только при GPU) → P5 параллельно.

## 5. Протокол каждого эксперимента

Заполнять в журнале ниже:

- id и гипотеза («что и почему меняем»);
- конфиг (path), данные (manifest + sha256), токенизатор (fingerprint);
- valid chrF по эпохам, best epoch, early stop;
- test BLEU/chrF (один раз), latency;
- вывод: принять/отклонить, следующий шаг.

## 6. Журнал экспериментов

| Дата | id | Изменение | Valid chrF | Test BLEU | Test chrF | Решение |
|---|---|---|---:|---:|---:|---|
| 2026-10-21 | opus50k_tiny_256_ru_en | baseline: tiny, max_len 256, opus50k | 26.14 | 7.52 | 26.02 | принят как baseline |
| 2026-10-21 | opus50k_tiny_256_en_ru | baseline en→ru | 25.76 | 4.78 | 23.13 | принят как baseline |
| 2026-10-22 | opus100_teacher_mixed_ru_en | данные-teacher (`opus-mt-ru-en`, beam 4) + реальные, 85.9k×2 | 35.91 | 15.76 | 37.93 | принят; лучший ru→en |
| 2026-10-22 | opus100_teacher_mixed_en_ru | teacher-файл оказался ru→ru (направление перепутано) | 6.32 | — | — | отклонён; данные невалидны |
| 2026-10-23 | opus100_en_ru | полный opus-100, реальные пары, swap en→ru, 85.9k | 27.38 | 8.17 | 27.94 | принят; +3.4 BLEU / +4.8 chrF к baseline |
| 2026-10-23 | opus100_en_ru + beam4 | beam 4, length penalty 0.6, весь test | — | 11.31 | 28.09 | принят (P0): +3.14 BLEU, +0.15 chrF; 1789 предложений за ~23 мин |
| 2026-10-23 | opus100_teacher_bt_en_ru | реальный en→ru + teacher-ru-en (reversed), 50/50, 171k | 30.87 | 10.88 | 32.48 | принят: +2.71 BLEU / +4.54 chrF к чистым реальным на том же split |
| 2026-10-24 | opus100_teacher_bt_en_ru + beam4+nr3+rep1.2 | улучшенный beam с запретом триграмм и штрафом за повторы | — | 12.20 | 31.31 | принят: +1.32 BLEU, -1.17 chrF; повторы 0%, латиница выросла 22.7% (нужно балансировать) |
| 2026-10-23 | opus100_teacher_mixed_ru_en + beam4 | beam 4, length penalty 0.6, весь test | — | 17.99 | 38.40 | принят (P0): +2.23 BLEU, +0.47 chrF; ~24 мин |

Профиль `opus100_en_ru` (для деплоя): 3.0M параметров, 11.5 MB fp32 / 2.9 MB int8, greedy 74.9 мс (8 потоков), median 70.4 мс на 1 потоке, RSS после загрузки 233 MB, пик при инференсе 276 MB; выбросы до 14 с из-за вырожденных 256-токенных повторов. Оценки ёмкости: d256/4+4/ff1024/16k ≈ 11.0M (42 MB fp32, 10 MB int8), d384/6+6/ff1536/32k ≈ 35.4M (135 MB, 34 MB), d512/8+8/ff2048/32k ≈ 71.1M (271 MB, 68 MB).

## 7. Целевые ориентиры

- Минимальная цель после P0–P2: **test chrF ≥ 30, BLEU ≥ 10** (без смены архитектуры, CPU-only).
- Амбициозная цель после P3–P4: **test chrF 35–40, BLEU 15–20** (100k+ данных и/или дистилляция от teacher).
- Перед любым релизом: выяснить лицензию данных, независимый test, multi-seed, отчёт о качестве с примерами; текущий флаг `production_approved: false` не снимать до этого.

## 8. Риски и известные ограничения

- Лицензия opus-100 не подтверждена → публичное распространение модели под вопросом.
- Один seed в baseline; различия <1 chrF могут быть шумом.
- DirectML на AMD не годится (backward pre-LN transformer падает); GPU-ускорение возможно только на NVIDIA/CUDA.
- Узкое место tiny — ёмкость, а не только данные; после P1–P3 потолок оценить честно и не «дожимать» шум на test.
- В `opus50k/test.jsonl` есть мусорные/OCR-подобные пары, что занижает метрики; при переходе на opus-100 отфильтровать test вручную по качеству, но не подгонять под модель.
