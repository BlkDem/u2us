import torch


def beam_search(model, src, beam_size=4, bos_id=2, eos_id=3, max_len=256, pad_id=0, length_penalty=0.6, no_repeat_ngram_size=0, repetition_penalty=1.0):
    """Single-example autoregressive beam search; returns token ids."""
    device = src.device
    beams = [(torch.tensor([[bos_id]], device=device), 0.0)]
    for _ in range(max_len - 1):
        candidates = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_id:
                candidates.append((seq, score))
                continue
            logits = model(src, seq, pad_id)[:, -1, :]
            if repetition_penalty != 1.0:
                for token_id in seq[0].tolist():
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty
            if no_repeat_ngram_size > 0 and seq.size(1) >= no_repeat_ngram_size:
                prefix = seq[0, -(no_repeat_ngram_size - 1):].tolist() if no_repeat_ngram_size > 1 else []
                # find all previous occurrences
                for i in range(seq.size(1) - no_repeat_ngram_size + 1):
                    if seq[0, i:i + no_repeat_ngram_size - 1].tolist() == prefix:
                        banned = seq[0, i + no_repeat_ngram_size - 1].item()
                        logits[0, banned] = -float('inf')
            logp = torch.log_softmax(logits, dim=-1)
            vals, ids = torch.topk(logp, beam_size, dim=-1)
            for v, i in zip(vals[0], ids[0]):
                candidates.append((torch.cat([seq, i.view(1, 1)], 1), score + v.item()))
        beams = sorted(candidates, key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)[:beam_size]
        if all(s[0, -1].item() == eos_id for s, _ in beams):
            break
    return beams[0][0]

