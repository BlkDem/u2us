"""Fixed-budget CPU training on existing data; no test-driven selection."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_real_baseline import dump, rows_write, sha


def repetition(text):
    words = text.split()
    char_run = any(len(set(text[i:i+8])) == 1 for i in range(max(0, len(text)-7)))
    return char_run or any(words[i:i+2] == words[i+2:i+4] == words[i+4:i+6] for i in range(max(0, len(words)-5)))


def output_stats(texts):
    return {'nonempty_rate': sum(bool(t.strip()) for t in texts)/len(texts),
            'repetition_rate': sum(repetition(t) for t in texts)/len(texts)}


def truncation(rows, tok, direction):
    result = {'examples': len(rows)}
    for side in ('src', 'tgt'):
        lengths = [len(tok.processor.encode((f'<{direction}> ' if side == 'src' else '') + r[side]))+2 for r in rows]
        count = sum(n > 64 for n in lengths)
        result[side] = {'truncated': count, 'fraction': count/len(rows), 'max_untruncated_tokens': max(lengths)}
    return result


def main():
    os.chdir(ROOT)
    report = Path('reports/controlled')
    if any(p.exists() for p in (report, Path('runs/controlled_ru_en'), Path('runs/controlled_en_ru'))):
        raise SystemExit('Controlled outputs already exist; refusing to overwrite')
    import torch
    import sacrebleu
    from micronmt.sp_tokenizer import SentencePieceTokenizer
    from micronmt.checkpoint import load_checkpoint
    torch.set_num_threads(2)
    report.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    tok_path = Path('data/tokenizer/real_baseline/shared.model')
    tok = SentencePieceTokenizer(tok_path)
    baseline = json.loads(Path('reports/real_baseline/summary.json').read_text())
    summary = {'status': 'running', 'production_approved': False, 'license': 'unknown',
               'budget': {'epochs': 12, 'max_steps_per_direction': 3000, 'patience': 4, 'selection': 'best fixed valid100 chrF', 'threads': 2, 'seed': 42},
               'repetition_definition': '8 identical consecutive characters OR a consecutive word bigram repeated three times',
               'tokenizer_sha256': sha(tok_path), 'results': {},
               'limitations': ['Unknown source rights; not cleared for production.', 'Test is the first fixed 100 baseline examples, not full test.', 'Overfit memorization is not generalization evidence.', 'Greedy max_len64; validation loss is mean batch label-smoothed loss.']}
    started = time.perf_counter()
    for direction, tag in [('ru_en', 'ru2en'), ('en_ru', 'en2ru')]:
        base = Path('data/real_baseline') / direction
        paths = {s: base / (s+'.jsonl') for s in ('train', 'valid', 'test')}
        rows = {s: [json.loads(line) for line in p.read_text().splitlines()] for s,p in paths.items()}
        result = {'data': {s: {'sha256': sha(p), 'rows': len(rows[s])} for s,p in paths.items()},
                  'truncation': {s: truncation(rows[s], tok, tag) for s in ('train', 'valid')},
                  'validation_cohort': {'first_rows': 100, 'selection': 'unchanged file order'},
                  'config_sha256': sha(Path(f'configs/controlled_{direction}.yaml'))}
        t = time.perf_counter()
        with (report / f'train_{direction}.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'micronmt.train', '--config', f'configs/controlled_{direction}.yaml', '--data', str(paths['train']), '--valid', str(paths['valid']), '--sp-model', str(tok_path), '--direction', tag, '--max-steps', '3000', '--patience', '4', '--eval-samples', '100', '--history', str(report / f'{direction}_history.jsonl')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result['train_seconds'] = time.perf_counter()-t
        history = [json.loads(l) for l in (report / f'{direction}_history.jsonl').read_text().splitlines()]
        for entry in history:
            entry.update(output_stats([s['hypothesis'] for s in entry['samples']]))
        rows_write(report / f'{direction}_history.jsonl', history)
        checkpoint = Path(f'runs/controlled_{direction}/best.pt')
        model, tokenizer, meta = load_checkpoint(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        last = torch.load(checkpoint.parent/'last.pt', map_location='cpu', weights_only=True)
        result.update(selected_checkpoint=str(checkpoint), selected_step=saved['global_step'], completed_steps=last['global_step'], epochs_evaluated=len(history), valid_loss=saved['valid_loss'], selected_valid_chrf=max(h['valid_chrf'] for h in history))
        refs = rows['test'][:100]
        baseline_refs = [json.loads(l) for l in (Path('reports/real_baseline') / f'{direction}_references.jsonl').read_text().splitlines()]
        assert refs == baseline_refs, 'test cohort differs from baseline'
        t = time.perf_counter()
        with torch.inference_mode():
            hyps = [tokenizer.decode(model.greedy(torch.tensor([tokenizer.encode(f'<{tag}> '+r['src'],64)]), max_len=64)[0]) for r in refs]
        result['decode_seconds'] = time.perf_counter()-t
        result['metrics'] = {'examples':100, 'BLEU':sacrebleu.corpus_bleu(hyps, [[r['tgt'] for r in refs]]).score, 'chrF':sacrebleu.corpus_chrf(hyps, [[r['tgt'] for r in refs]]).score, **output_stats(hyps)}
        result['baseline_metrics'] = baseline['results'][direction]['metrics']
        result['samples'] = [{'source':r['src'], 'reference':r['tgt'], 'hypothesis':h} for r,h in zip(refs[:10],hyps[:10])]
        rows_write(report / f'{direction}_references.jsonl',refs)
        rows_write(report / f'{direction}_hypotheses.jsonl',[{'src':r['src'],'tgt':h} for r,h in zip(refs,hyps)])
        summary['results'][direction] = result
        dump(report/'summary.json',summary)
    summary.update(status='completed', elapsed_seconds=time.perf_counter()-started,
                   peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                   peak_driver_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   python=sys.executable)
    dump(report/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
