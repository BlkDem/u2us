import argparse
import json
from pathlib import Path


def corpus(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf8').splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{number}: invalid JSON') from exc
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ('src', 'tgt')):
            raise ValueError(f'{path}:{number}: expected string src and tgt')
        if not row['src'].strip():
            raise ValueError(f'{path}:{number}: empty src')
        rows.append(row)
    if not rows:
        raise ValueError(f'{path}: empty corpus')
    return rows


def aligned(reference, hypothesis):
    refs, hyps = corpus(reference), corpus(hypothesis)
    if len(refs) != len(hyps):
        raise ValueError(f'length mismatch: {len(refs)} references, {len(hyps)} hypotheses')
    for number, (ref, hyp) in enumerate(zip(refs, hyps), 1):
        if not ref['tgt'].strip():
            raise ValueError(f'{reference}:{number}: empty reference tgt')
        if ref['src'] != hyp['src']:
            raise ValueError(f'source alignment mismatch at row {number}')
    return refs, hyps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--hypothesis', required=True)
    p.add_argument('--out')
    p.add_argument('--comet-model')
    a = p.parse_args()
    reference, hypothesis = aligned(a.reference, a.hypothesis)
    refs = [x['tgt'] for x in reference]
    hyps = [x['tgt'] for x in hypothesis]
    result = {'examples': len(refs)}
    import sacrebleu
    result['BLEU'] = sacrebleu.corpus_bleu(hyps, [refs]).score
    result['chrF'] = sacrebleu.corpus_chrf(hyps, [refs]).score
    if a.comet_model:
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model(a.comet_model))
        data = [{'src': r['src'], 'mt': h['tgt'], 'ref': r['tgt']}
                for r, h in zip(reference, hypothesis)]
        result['COMET'] = float(model.predict(data, gpus=0)['system_score'])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding='utf8')


if __name__ == '__main__':
    main()
