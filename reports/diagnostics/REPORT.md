# Completed offline initialization diagnosis

Only existing RU→EN train/valid and the existing SP2000 tokenizer were read; no test data or network access. Production `micronmt/model.py` is unchanged. Old experiments are preserved.

## Data/tokenization audit (unchanged first100 per split)

All 200 rows satisfy src/tgt string schema. Automated language/alignment heuristics flag 3 train and 10 valid rows; this does not establish semantic alignment. Saved 100 rows **per split** for future human review, not a completed human review.

| Fixed cohort | Source UNK | Target UNK | Source/target normalized roundtrip discrepancies | Source/target truncations |
|---|---:|---:|---:|---:|
| train100 | .054054% | .062500% | 1 / 1 | 9 / 5 |
| valid100 | 0% | .033761% | 0 / 1 | 16 / 14 |

All encodings retain EOS, but encode forcibly appends it after truncating: EOS retention alone cannot demonstrate intact sentences. Lengths include BOS/EOS and source direction tag. Detailed raw and normalized roundtrips, script fractions, lengths and alignment flags are retained in JSONL. Low UNK rates do not support widespread unknown-token failure; substantial truncation remains a real confound in previous runs. Heuristics cannot rule out incorrect translations or subtle direction/alignment errors.

## Matched experiment

First 3,000 eligible train pairs and first 100 eligible validation pairs in original order, each side fully <=64 tokens, including source tag and BOS/EOS; no truncation. Batch32, seed42, d64, four heads, encoder2/decoder2, FF128, dropout .1, smoothing .1, AdamW defaults (weight decay .01), constant LR .001, clip1. Both arms complete 1,500 updates (~16 epochs), evaluating every250 without early stopping. Exact same shuffled batches and training dropout RNG. Validation chrF selects among trained evaluation checkpoints; step0 is diagnostic, not a selection candidate.

A uses unchanged initialization. B copies all A tensors, then reinitializes only the tied shared embedding/output with normal std=.125 (=64^-0.5), pad row zero, independent seed424242. This brings shared output logits closer to unit scale while accounting for sqrt(d) input scaling. Only `embedding.weight` and its `output.weight` alias differ; tying is preserved. Initial zero pad does not imply it stays zero: tied output gradients can update that row during training.

Initial **unfiltered fixed valid100**: A CE=53.484260606, logit std=8.150353916; B CE=8.098430276, std=1.005041287. A/B overall teacher-forced accuracy=.574919%/.395257%, content=.604752%/.431965%. Both initially produce 63 generated tokens with zero EOS; A decodes empty because special tokens are stripped, B has 99% text-repetition heuristic rate. Text repetition ignores repeated special IDs, so zero text repetition is not evidence of healthy generation.

## Selected short-valid100 results

| Metric | A unchanged | B controlled embedding |
|---|---:|---:|
| Selected step | 1500 | 1250 |
| Completed updates | 1500 | 1500 |
| chrF | 2.236089811 | **9.895955377** |
| BLEU | .095216551 | .481661758 |
| Unsmoothed non-PAD CE | 5.322493175 | 4.698798698 |
| Label-smoothed CE | 5.703064634 | 5.219578004 |
| Teacher-forced accuracy | 13.111010% | 15.851987% |
| Content-token accuracy | 3.528742% | 7.911212% |
| Punctuation-token accuracy | 40.662651% | 34.939759% |
| Mean generated/reference tokens | 25.63 / 20.89 | 27.13 / 20.89 |
| EOS rate | 72% | 86% |
| Premature EOS (<half reference tokens) | 35% | 6% |
| Immediate EOS | 0% | 0% |
| Nonempty | 100% | 99% |
| Text repetition heuristic | 42% | 40% |

B final step1500 chrF=9.674010757, CE=4.745066971: best is correctly step1250, not last. Hypotheses for all 100 validation examples at every evaluation (including initialization) are in arm histories.

Timings using system Python, two CPU threads: A 313.448355487s, B 322.060477994s, total 635.966000140s (includes auditing and serialization; arm timings include initial/fixed-cohort evaluations). Full unittest discovery subsequently passed **47 tests in 141.793s**; diagnostic-only tests passed four tests in .068s. Logs: `tests.log`.

## Interpretation and limitations

Controlled embedding initialization improves validation chrF by 7.659865566 points and more than doubles content-token accuracy. The large initial CE/logit-scale reduction and matched training support initialization scale as a contributing optimization problem. They do not prove it is the only cause or distinguish changes in input embedding scale from tied output scale. Both arms are filtered identically: initialization is isolated between these arms, but filtering versus prior runs is not. Truncation, limited capacity/data, incorrect alignments and tokenizer granularity remain possible contributors. A single seed and selected small validation cohort are not an independent generalization estimate. BLEU remains below .5, repetition high and translation quality poor. No production initialization change, no larger third run, no test evaluation. License unknown; production not approved.

`summary.json` records complete configurations, exact counts, tokenizer/data/selection/batch hashes and selected checkpoint hashes. `batch_indices.json` records every matched batch. Selected checkpoints: `runs/diagnostics/A/best.pt`, `runs/diagnostics/B/best.pt`; each arm also has initial and last states.
