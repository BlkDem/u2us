# Implementation audit

## Current architecture

Compact PyTorch `nn.TransformerEncoder` + `nn.TransformerDecoder`, shared source/target embedding and tied output projection. A direction token (`<ru2en>`/`<en2ru>`) is prepended to the source. Tiny config is 2+2 layers, d_model=128, 4 heads.

## Current quality

No valid real-corpus quality result exists yet. Toy results are smoke tests only and must not be interpreted as translation quality. BLEU/chrF infrastructure exists; COMET is optional.

## Current bottlenecks

- Word-level fallback tokenizer remains available and can silently produce poor models.
- SentencePiece metadata references an external file rather than packaging it with checkpoints.
- Teacher distillation currently requires compatible student/teacher vocabulary and architecture.
- Greedy/beam inference repeatedly recomputes the full decoder prefix.
- ONNX currently exports one decoder forward pass, not a complete translation runtime.
- No calibrated confidence score is exposed, correctly.

## Dataset problems

- JSONL preparation has basic cleaning, ratio filtering and exact deduplication.
- Language identification, near-duplicate detection, semantic filtering and source/license manifests are incomplete.
- Tiny toy data is not a quality benchmark.
- C2 and permanent regression suites are not yet populated.

## Training problems

- No scheduler, curriculum sampler, mixed precision policy or token-based batching.
- Validation is loss-only and does not select by translation metrics.
- Distillation is implemented but not yet tested on a strong external teacher.
- Continual training has replay but does not gate release on regression thresholds.

## Inference problems

- PyTorch CPU inference works; beam search is available.
- Context and terminology are input hints, not hard constraints.
- ONNX autoregressive runtime is incomplete.
- No streaming is intentionally provided because quality has priority.

## Compression opportunities

- ONNX dynamic INT8 and weight-only quantization.
- Shared embeddings/tied output already reduce parameters.
- LoRA adapters avoid copying the base model.
- Encoder/decoder export enables mobile/browser deployment.
- Pruning and layer dropping should only be accepted after regression evaluation.

## Recommended next steps

1. Export and validate separate encoder/decoder ONNX graphs.
2. Package tokenizer and metadata atomically with model artifacts.
3. Add language/semantic filtering and near-deduplication.
4. Train on licensed real corpora and create permanent 5k–10k C2 regression sets.
5. Compare CE, distillation, curriculum and 16k/32k/48k SentencePiece variants.
6. Add release gating: regression suite, BLEU/chrF/COMET, latency, size and RAM.
7. Implement protected identifiers, terminology and translation-memory signals.
