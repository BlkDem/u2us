# Real-data baseline

## Completed short-pair initialization comparison (no test)

Run `PYTHONPATH=. /usr/bin/python scripts/diagnose_training.py` only with fresh `reports/diagnostics` and `runs/diagnostics` paths; existing outputs are refused. Completed artifacts are already present. No download or corpus expansion occurred.

3,000 train and100 valid RU→EN pairs were selected in existing file order with both encoded sides fully <=64 including direction tag and BOS/EOS. Two matched seed42 arms each completed1500 updates, batch32, d64, encoder2/decoder2, FF128, dropout.1, smoothing.1, constant AdamW LR.001; evaluation every250. B copies A and only replaces shared embedding/tied-output weights with normal std=.125 and zero initial PAD, independent initialization RNG. Production model code was not altered.

| Validation-selected result | A unchanged | B controlled |
|---|---:|---:|
| Selected step | 1500 | 1250 |
| chrF | 2.236089811 | 9.895955377 |
| BLEU | .095216551 | .481661758 |
| Non-PAD CE | 5.322493175 | 4.698798698 |
| Content-token accuracy | 3.528742% | 7.911212% |
| EOS / premature EOS | 72% / 35% | 86% / 6% |
| Text repetition | 42% | 40% |
| Arm elapsed seconds | 313.448355487 | 322.060477994 |

Total635.966000140s. B wins validation chrF; test was never read and no third run was launched. Initial unfiltered valid100 CE53.484260606→8.098430276 and logit std8.150353916→1.005041287 support an initialization-scale contribution. This does not establish usable translation or isolate input versus tied-output scale. Both arms use identical short filtering, so filtering effects relative to previous experiments cannot be isolated. Single-seed small-validation selection, low BLEU and high repetition limit conclusions.

First100 train/valid automated audits, future-human-review samples (not human reviewed), full histories and exact data/tokenizer/batch/checkpoint hashes are in `reports/diagnostics/`; selected/initial/last checkpoints are in `runs/diagnostics/{A,B}/`. Detailed report: `reports/diagnostics/REPORT.md`. Full unittest discovery passed47 tests in141.793s. License remains unknown and production not approved.

## Completed controlled training on the existing corpus

Reproduce into fresh output directories with system Python (the local `.venv` does not supply Torch):

```bash
PYTHONPATH=. /usr/bin/python scripts/train_controlled.py
```

The script refuses existing controlled outputs. It reads the archived baseline data/tokenizer without rewriting them and stores independent models in `runs/controlled_ru_en/` and `runs/controlled_en_ru/`. The existing 8,000 train / 1,916 valid / 1,935 test pairs per direction were not expanded. Data, tokenizer and config hashes are retained in `reports/controlled/summary.json`.

The fixed budget was 12 epochs / maximum 3,000 steps per direction, two CPU threads, batch 32, lr .001, dropout .1, label smoothing .1; architecture 64-dimensional, two encoder/decoder layers, FF128, max_len64. Patience=4 used the same first 100 validation rows every epoch. Validation loss covers the full valid split. No test-driven configuration or checkpoint selection was performed.

| Direction | Completed epochs / steps | Selected step | Best valid chrF | Selected valid loss | Test100 BLEU | Test100 chrF |
|---|---:|---:|---:|---:|---:|---:|
| RU→EN | 5 / 1250 | 250 | 0.378121 | 6.489181 | 0.003919 | 0.353189 |
| EN→RU | 6 / 1500 | 500 | 0.653326 | 6.287805 | 0 | 0.607416 |

Both stopped by the predeclared patience rule. Baseline test100 BLEU/chrF were 0/1.390384 and 0.016263/0.156344 respectively. Cohorts match the archived reference files exactly. Neither result constitutes useful translation: samples include `.` / `,.` for RU→EN and `- .` / `Я` for EN→RU. Nonempty rates were 100% / 96%; heuristic repetition rates 1% / 10%. Per-epoch hypotheses, losses, chrF and repetition rates are retained in `reports/controlled/*_history.jsonl`, with final test hypotheses and references alongside them.

Truncated source/target fractions (including BOS/EOS and source direction tag):

| Direction | Train source | Train target | Valid source | Valid target |
|---|---:|---:|---:|---:|
| RU→EN | 830/8000 (10.375%) | 658/8000 (8.225%) | 367/1916 (19.1545%) | 281/1916 (14.6660%) |
| EN→RU | 700/8000 (8.750%) | 797/8000 (9.9625%) | 304/1916 (15.8664%) | 351/1916 (18.3194%) |

Training wall time was 276.576s / 255.163s; final test decoding 1.409s / 3.456s; total 537.509s. Peak child RSS 608,920 KiB, driver RSS 605,860 KiB (not a combined peak). Tests ran concurrently, so timings are not isolated benchmarks. `python -m unittest discover -s tests -v` passed 43 tests in 162.741s; the count includes repeated inherited/imported training tests. The unknown license is unchanged, `production_approved: false`. Train-only overfit success is not evidence of generalization.


The reproducible real-data path is:

```bash
python -m pip install --user -e ".[data,train]"
python scripts/download_parallel.py \
  --dataset Helsinki-NLP/opus-100 --config en-ru --split train \
  --revision 805090dc28bf78897da9641cdf08b61287580df9 --license unknown \
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

The download script uses `Helsinki-NLP/opus-100`, not the ambiguous short name `opus100`, and requires a pinned revision and an explicit license provenance field. At revision `805090dc28bf78897da9641cdf08b61287580df9`, the dataset card reports `license: unknown`. This is NOT verified permission for production or redistribution. Manifests mark artifacts `production_approved: false`; underlying corpus licenses require separate review. The `--license` argument records the operator's provenance statement, not an automated legal validation.

An earlier version of this document reported a 5,000-example smoke run with validation loss 13.6331, but its artifacts were not present during the readiness audit. Treat that historical number as unverified, not a current reproducible quality result. Current runs must retain their revision, data hashes, configuration, checkpoint and measured reports.
