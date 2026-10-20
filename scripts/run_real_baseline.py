"""Pinned OPUS-100 CPU smoke baseline; unknown license is not a legal clearance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import read_rows, leakage_key

REVISION = '805090dc28bf78897da9641cdf08b61287580df9'
DATASET = 'Helsinki-NLP/opus-100'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def rows_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(raw_paths, train_limit):
    seen = set()
    parts, report = {}, {}
    for name in ('test', 'valid', 'train'):
        clean = read_rows([raw_paths[name]], 2000, 3.0)
        kept = []
        rejected = 0
        for row in clean:
            keys = {leakage_key(row[k]) for k in ('src', 'tgt')}
            if keys & seen:
                rejected += 1
                continue
            seen.update(keys)
            kept.append(row)
        parts[name] = kept[:train_limit] if name == 'train' else kept
        report[name] = {'clean_unique_pairs': len(clean), 'normalized_side_overlap_removed': rejected,
                        'retained': len(parts[name])}
    return parts, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-limit', type=int, default=8000)
    p.add_argument('--max-steps', type=int, default=150)
    p.add_argument('--eval-limit', type=int, default=100)
    p.add_argument('--resume', action='store_true', help='reuse downloaded/prepared artifacts and resume last checkpoints')
    a = p.parse_args()
    if not 5000 <= a.train_limit <= 10000 or not 1 <= a.max_steps <= 150 or not 1 <= a.eval_limit <= 2000:
        p.error('train-limit 5000..10000, max-steps 1..150, eval-limit 1..2000 required')
    os.chdir(ROOT)
    data = Path('data/real_baseline')
    report = Path('reports/real_baseline')
    tok = Path('data/tokenizer/real_baseline/shared')
    outputs = [data, report, tok.parent, Path('runs/baseline_ru_en'), Path('runs/baseline_en_ru')]
    if not a.resume and any(x.exists() for x in outputs):
        p.error('outputs already exist; explicit --resume required')
    report.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')

    def run(label, args):
        t = time.perf_counter()
        with (report / (label + '.log')).open('a' if a.resume else 'w') as log:
            subprocess.run([sys.executable] + args, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        timings[label] = time.perf_counter() - t

    provenance_path = data / 'provenance.json'
    if not provenance_path.exists():
        import requests
        import pyarrow.parquet as pq
        t = time.perf_counter()
        api = f'https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}'
        response = requests.get(api, timeout=120); response.raise_for_status()
        metadata = response.json()
        license_value = metadata.get('cardData', {}).get('license', 'unknown')
        dump(report / 'dataset_metadata.json', metadata)
        license_text = json.dumps(license_value).lower()
        if any(x in license_text for x in ('no-training', 'no_training', 'non-commercial-no-training')):
            raise RuntimeError('dataset license prohibits training')
        provenance = {'dataset': DATASET, 'revision': REVISION, 'license': license_value,
                      'license_caution': 'Unknown/mixed source rights; no legal guarantee or redistribution clearance.', 'files': {}}
        print('cardData license:', license_value, flush=True)
        for split, n in [('train', 10000), ('validation', 2000), ('test', 2000)]:
            filename = f'en-ru/{split}-00000-of-00001.parquet'
            url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}'
            target = data / 'downloads' / Path(filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with target.open('wb') as f:
                    for block in r.iter_content(1024 * 1024): f.write(block)
            rows = []
            for batch in pq.ParquetFile(target).iter_batches(batch_size=1000):
                for item in batch.to_pylist():
                    tr = item['translation']; rows.append({'src': tr['ru'], 'tgt': tr['en']})
                    if len(rows) == n: break
                if len(rows) == n: break
            raw = data / 'raw' / ('valid.jsonl' if split == 'validation' else split + '.jsonl')
            rows_write(raw, rows)
            provenance['files'][split] = {'url': url, 'sha256': sha(target), 'download_bytes': target.stat().st_size,
                                         'selected_rows': len(rows), 'raw_sha256': sha(raw)}
        dump(provenance_path, provenance)
        timings['download'] = time.perf_counter() - t
    provenance = json.loads(provenance_path.read_text())
    parts, leakage = prepare({s: data / 'raw' / (s + '.jsonl') for s in ('test', 'valid', 'train')}, a.train_limit)
    if len(parts['train']) < 5000 or not parts['valid'] or not parts['test']:
        raise RuntimeError('insufficient clean rows')
    dump(report / 'leakage.json', {'priority': ['test', 'valid', 'train'], 'splits': leakage,
                                 'remaining_normalized_side_overlap': 0})
    manifest = {}
    for direction in ('ru_en', 'en_ru'):
        for split, rows in parts.items():
            dest = data / direction / (split + '.jsonl')
            rows_write(dest, rows if direction == 'ru_en' else [{'src': r['tgt'], 'tgt': r['src']} for r in rows])
            manifest[str(dest)] = {'rows': len(rows), 'sha256': sha(dest)}
    dump(data / 'manifest.json', manifest)
    if not tok.with_suffix('.model').exists():
        run('tokenizer', ['scripts/train_tokenizer.py', '--input', str(data / 'ru_en/train.jsonl'), '--prefix', str(tok), '--vocab-size', '2000'])
    import torch
    from micronmt.checkpoint import load_checkpoint
    torch.set_num_threads(2)
    results = {}
    for direction, tag in [('ru_en', 'ru2en'), ('en_ru', 'en2ru')]:
        checkpoint = Path('runs') / ('baseline_' + direction) / 'last.pt'
        args = ['-m', 'micronmt.train', '--config', f'configs/baseline_{direction}.yaml', '--data', str(data / direction / 'train.jsonl'), '--valid', str(data / direction / 'valid.jsonl'), '--sp-model', str(tok.with_suffix('.model')), '--direction', tag, '--max-steps', str(a.max_steps)]
        if a.resume and checkpoint.exists(): args += ['--resume', str(checkpoint)]
        run('train_' + direction, args)
        t = time.perf_counter()
        model, tokenizer, meta = load_checkpoint(checkpoint)
        refs = parts['test'][:a.eval_limit]
        refs = refs if direction == 'ru_en' else [{'src': r['tgt'], 'tgt': r['src']} for r in refs]
        hyps = []
        with torch.inference_mode():
            for row in refs:
                ids = torch.tensor([tokenizer.encode('<' + tag + '> ' + row['src'], 64)])
                hyps.append({'src': row['src'], 'tgt': tokenizer.decode(model.greedy(ids, max_len=64)[0])})
        timings['decode_' + direction] = time.perf_counter() - t
        ref_path, hyp_path = report / (direction + '_references.jsonl'), report / (direction + '_hypotheses.jsonl')
        rows_write(ref_path, refs); rows_write(hyp_path, hyps)
        metric_path = report / (direction + '_metrics.json')
        run('evaluate_' + direction, ['-m', 'micronmt.evaluate', '--reference', str(ref_path), '--hypothesis', str(hyp_path), '--out', str(metric_path)])
        bench_path = report / (direction + '_latency.json')
        run('benchmark_' + direction, ['-m', 'micronmt.benchmark', '--checkpoint', str(checkpoint), '--text', refs[0]['src'], '--source', 'ru' if direction == 'ru_en' else 'en', '--runs', '3', '--out', str(bench_path)])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        results[direction] = {'metrics': json.loads(metric_path.read_text()), 'benchmark': json.loads(bench_path.read_text()), 'global_step': saved['global_step'], 'train_loss': saved['train_loss'], 'valid_loss': saved['valid_loss'], 'samples': [{'source': r['src'], 'reference': r['tgt'], 'hypothesis': h['tgt']} for r,h in zip(refs[:5], hyps[:5])]}
    timings['total'] = time.perf_counter() - started
    dump(report / 'summary.json', {'status': 'completed', 'arguments': vars(a), 'provenance': provenance, 'counts': {s: len(r) for s,r in parts.items()}, 'tokenizer': {'vocab_size': 2000, 'training_split': 'train only', 'sha256': sha(tok.with_suffix('.model'))}, 'timings_seconds': timings, 'results': results, 'limitations': ['Unknown license: no legal guarantee; upstream source-specific rights require review.', 'Tiny CPU smoke baseline, at most 150 batches, not a production translation system.', 'First fixed test examples, not full test; max length 64 truncates sentences.', 'Greedy decoding; no COMET, GPU, ONNX or full quality tuning.', 'On resume timings cover this invocation only; earlier download/tokenizer times are not included.', 'Parquet shards downloaded fully, only first 10000 train rows selected.']})
    print((report / 'summary.json').read_text())


if __name__ == '__main__':
    main()
