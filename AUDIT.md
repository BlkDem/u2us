# Implementation audit

## Readiness

Research prototype; not a production translation release. Local real-corpus baseline and controlled-training artifacts are now recorded in `reports/real_baseline/` and `reports/controlled/`; they demonstrate very poor translation quality, not release readiness. Passing toy tests does not demonstrate translation quality.

## Completed controlled real-corpus run

`python scripts/train_controlled.py` completed using `/usr/bin/python`, two CPU threads, the unchanged 8,000-pair train split and existing tokenizer, without expanding data or replacing baseline/overfit artifacts. Two independent models used d_model=64, two encoder/decoder layers, FF=128, max_len=64, batch=32, lr=.001, dropout=.1 and label smoothing=.1. The predefined budget was 12 epochs / 3,000 batches per direction with patience=4 on fixed valid100 chrF; early stopping ended RU→EN at 1,250 steps (5 epochs) and EN→RU at 1,500 (6 epochs). Selected best checkpoints are steps 250 and 500 respectively. Test100 was checked against the original baseline references and used only for final evaluation.

| Direction | Test BLEU (baseline → controlled) | Test chrF (baseline → controlled) | Nonempty | Repetition |
|---|---:|---:|---:|---:|
| RU→EN | 0 → 0.003919 | 1.390384 → 0.353189 | 100% | 1% |
| EN→RU | 0.016263 → 0 | 0.156344 → 0.607416 | 96% | 10% |

Outputs remain degenerate (e.g. `.` / `,.` and `- .` / `Я`); nonempty does not mean a usable translation. Validation loss decreased while chrF remained near zero. Full per-epoch losses, chrF, fixed-cohort hypotheses and repetition rates are in `reports/controlled/*_history.jsonl`. Repetition means eight identical consecutive characters or a consecutive word bigram repeated three times; it is a heuristic, not a quality metric.

Training took 276.576s / 255.163s; total wall time was 537.509s. Peak child RSS was 608,920 KiB and driver RSS 605,860 KiB (separate process maxima, not a combined peak). Concurrent test execution may affect timings. Full test discovery completed successfully: **43 tests, 162.741s, OK**, including the history CLI test; inherited/imported training tests account for repeated cases in this count.

Truncation at 64 tokens affects RU→EN train source/target 10.375% / 8.225%, valid 19.1545% / 14.6660%; EN→RU train 8.750% / 9.9625%, valid 15.8664% / 18.3194%. Counts include BOS/EOS and the source direction tag. No generalization claim is inferred from the separate train-only overfit check. License remains unknown and `production_approved` is false.

## Completed initialization diagnosis (RU→EN, no test)

`scripts/diagnose_training.py` completed both matched 1,500-update arms on identical 3,000 short train pairs / 100 short validation pairs (both sides fully <=64 including tag and BOS/EOS). A preserves initialization; B copies A and changes only the tied embedding/output to normal std=.125, zero initial PAD, independent RNG. Other model parameters, batches, seed42, dropout, optimizer and constant LR=.001 are identical. Production `micronmt/model.py` is unchanged.

Validation-selected A/B chrF: **2.236089811 / 9.895955377**, BLEU .095216551 / .481661758; selected steps1500/1250, both complete1500. Non-PAD CE5.322493175/4.698798698; content-token accuracy3.528742%/7.911212%. Fixed unfiltered valid100 initial CE53.484260606/8.098430276 and logit std8.150353916/1.005041287 support embedding/tied-output scale as a contributing optimization issue, not the only cause. EOS72%/86%, premature EOS35%/6%, repetition42%/40% indicate quality remains poor.

Automated audit of first100 train/valid found zero schema failures, 3/10 heuristic alignment/language flags, low UNK rates, and source/target truncations9/5 and16/14. Samples are saved for future human review; no human review was performed. Both experimental arms use the same short filtering, so initialization is isolated between arms but filtering versus previous experiments is not. No test accessed and no third experiment launched. License unknown, production not approved.

Exact arm timings313.448355487s/322.060477994s; total635.966000140s. Full discovery passed47 tests in141.793s, including four new diagnostic tests. Detailed metrics, definitions, limitations, hashes, samples and timings: `reports/diagnostics/REPORT.md`, `reports/diagnostics/summary.json`; checkpoints: `runs/diagnostics/{A,B}/`. Previous artifacts preserved.

## Correctness fixes

- Training and validation use the requested direction token. Resume inherits the saved direction unless explicitly overridden.
- New checkpoints embed SentencePiece model bytes and do not depend on an external tokenizer path. Legacy external metadata remains supported.
- Resume restores progress, optimizer, scheduler, scaler, best score, early stopping and Python/Torch RNG state.
- Step limits terminate training and flush partial gradient accumulation windows with their actual divisor. Resume after such a flush need not match uninterrupted optimizer updates bit-for-bit.
- Distillation checks tokenizer compatibility rather than vocabulary size alone.
- LoRA freezes the full base model and saves all trainable adapter parameters. Attention out_proj is intentionally excluded because MultiheadAttention bypasses its module forward. Old adapters containing those keys must be retrained.
- Data preparation splits connected components sharing normalized source or target text, including cross-side matches. This prevents exact-text leakage but not semantic or near-duplicate leakage.
- Evaluation validates schema, nonempty corpora and row alignment; metric failures are not successful reports.
- Regression CLI requires an explicit threshold or baseline and exits nonzero on failure. Exact match alone is not an adequate translation-quality release gate.
- Packaging checks tokenizer compatibility. Split ONNX exports include a tokenizer and sequence length metadata.

## Verification

Run built-in integration tests without pytest:

```bash
python -m unittest discover -s tests -v
```

Optional pytest installation runs both these tests and the original function-style smoke tests:

```bash
pip install -e '.[train,dev]'
python -m pytest -q
```

Tests cover checkpoint/tokenizer round trips, directions, malformed data, LoRA updates and round trips, training stop/resume, partial accumulation, teacher compatibility, split leakage and regression thresholds. ONNX parity tests require the train extras.

## Remaining release requirements

1. Train on licensed real corpora and establish an independent permanent RU↔EN benchmark.
2. Measure BLEU/chrF, optional COMET and human adequacy/naturalness before claiming translation quality.
3. Gate releases on quality metrics and regression thresholds, latency, memory and artifact size.
4. Training is CPU-only; GPU/device support and scalable token-based batching remain future work.
5. Decoder generation recomputes prefixes; no KV cache is implemented.
6. Context and terminology are hints, not hard constraints; long context can displace source text under truncation.
7. Near-duplicate and semantic filtering require further validation.
8. Shared bidirectional training needs an explicit mixed-direction sampling design; a single-direction training run does not establish competence in both directions.
