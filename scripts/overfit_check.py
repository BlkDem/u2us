"""Controlled train-only memorization check; never measures generalization."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import sacrebleu
from micronmt.model import MicroTransformer
from micronmt.sp_tokenizer import SentencePieceTokenizer
from micronmt.checkpoint import tokenizer_metadata, tokenizer_fingerprint


def select_rows(path, tok, count=16, max_len=64, direction='ru2en'):
    selected = []
    for index, line in enumerate(Path(path).read_text(encoding='utf8').splitlines()):
        row = json.loads(line)
        texts = (f'<{direction}> ' + row['src'], row['tgt'])
        ids = [[tok.bos_id, *tok.processor.encode(text, out_type=int), tok.eos_id] for text in texts]
        if max(map(len, ids)) > max_len or max(map(len, ids)) > 24:
            continue
        if tok.unk_id in ids[0] or tok.unk_id in ids[1] or tok.decode(ids[1]) != row['tgt']:
            continue
        selected.append(dict(row, train_line=index + 1, src_ids=ids[0], tgt_ids=ids[1]))
        if len(selected) == count:
            return selected
    raise ValueError('not enough short, losslessly encoded train pairs')


def batch(rows, pad_id=0):
    def padded(key):
        result = torch.full((len(rows), max(len(r[key]) for r in rows)), pad_id, dtype=torch.long)
        for i, row in enumerate(rows):
            result[i, :len(row[key])] = torch.tensor(row[key])
        return result
    return padded('src_ids'), padded('tgt_ids')


def token_stats(logits, targets, pad_id=0):
    mask = targets.ne(pad_id)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id)
    accuracy = ((logits.argmax(-1) == targets) & mask).sum().item() / mask.sum().item()
    return loss, accuracy


@torch.inference_mode()
def evaluate(model, tok, rows, step):
    model.eval()
    src, tgt = batch(rows, tok.pad_id)
    loss, accuracy = token_stats(model(src, tgt[:, :-1]), tgt[:, 1:], tok.pad_id)
    hypotheses = [tok.decode(ids) for ids in model.greedy(src, max_len=64)]
    refs = [r['tgt'] for r in rows]
    return {'step': step, 'loss': loss.item(), 'token_accuracy': accuracy,
            'exact_match': sum(h == r for h, r in zip(hypotheses, refs)) / len(rows),
            'chrF': sacrebleu.corpus_chrf(hypotheses, [refs]).score,
            'samples': [{'source': r['src'], 'reference': r['tgt'], 'hypothesis': h} for r, h in zip(rows, hypotheses)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', choices=['ru2en', 'en2ru'], default='ru2en')
    p.add_argument('--pairs', type=int, default=16)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--report-dir', type=Path, default=ROOT / 'reports/overfit')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/overfit')
    a = p.parse_args()
    if not 16 <= a.pairs <= 32 or not 500 <= a.steps <= 1500 or not 100 <= a.eval_every <= 200 or not .001 <= a.lr <= .003:
        p.error('pairs 16..32, steps 500..1500, eval-every 100..200, lr .001...003 required')
    if a.report_dir.exists() or a.run_dir.exists():
        p.error('output directory already exists; use new directories to preserve earlier experiments')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    started = time.perf_counter()
    train = ROOT / 'data/real_baseline' / ('ru_en' if a.direction == 'ru2en' else 'en_ru') / 'train.jsonl'
    tok = SentencePieceTokenizer(ROOT / 'data/tokenizer/real_baseline/shared.model')
    rows = select_rows(train, tok, a.pairs, direction=a.direction)
    config = dict(d_model=64, nhead=4, encoder_layers=2, decoder_layers=2,
                  dim_feedforward=128, dropout=0., max_len=64, batch_size=16,
                  lr=a.lr, label_smoothing=0., seed=a.seed)
    model = MicroTransformer(tok.vocab_size, **{k: config[k] for k in ('d_model', 'nhead', 'encoder_layers', 'decoder_layers', 'dim_feedforward', 'dropout', 'max_len')})
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    history = [evaluate(model, tok, rows, 0)]
    a.report_dir.mkdir(parents=True)
    a.run_dir.mkdir(parents=True)
    def save(step):
        report = {'status': 'converged' if history[-1]['exact_match'] >= .95 else 'running' if step < a.steps else 'not_converged',
                  'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  'config': config, 'threads': torch.get_num_threads(), 'device': 'cpu',
                  'train_path': str(train), 'train_sha256': hashlib.sha256(train.read_bytes()).hexdigest(),
                  'tokenizer_sha256': tokenizer_fingerprint(tok), 'selected_pairs': rows,
                  'encoded_length_max': {key: max(len(r[key]) for r in rows) for key in ('src_ids', 'tgt_ids')},
                  'truncation': False, 'history': history, 'updates': step,
                  'elapsed_seconds': time.perf_counter() - started,
                  'limitations': ['Train-only memorization, not generalization or translation quality evidence.', 'Existing baseline data and checkpoints are read-only.', 'Constant learning rate; AdamW default weight decay .01; gradient clipping 1.0.']}
        (a.report_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        torch.save({'model': model.state_dict(), 'config': config, 'tokenizer': tokenizer_metadata(tok),
                    'global_step': step, 'direction': a.direction, 'train_loss': history[-1]['loss'],
                    'optimizer': optimizer.state_dict(), 'torch_rng_state': torch.get_rng_state()}, a.run_dir / 'model.pt')
        print(json.dumps({k: v for k, v in history[-1].items() if k != 'samples'}), flush=True)
    save(0)
    order = []
    for step in range(1, a.steps + 1):
        if not order:
            order = torch.randperm(len(rows)).tolist()
        chosen, order = order[:16], order[16:]
        src, tgt = batch([rows[i] for i in chosen])
        model.train()
        optimizer.zero_grad()
        loss, _ = token_stats(model(src, tgt[:, :-1]), tgt[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % a.eval_every == 0 or step == a.steps:
            history.append(evaluate(model, tok, rows, step))
            save(step)
            if history[-1]['exact_match'] >= .95:
                break


if __name__ == '__main__':
    main()
