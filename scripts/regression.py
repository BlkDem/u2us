import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

# Keep direct script invocation working without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from micronmt.evaluate import aligned, corpus


def exact_match(reference, hypothesis):
    refs, hyps = aligned(reference, hypothesis)
    return sum(h['tgt'].strip() == r['tgt'].strip() for r, h in zip(refs, hyps)) / len(refs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--source', default='ru', choices=['ru', 'en'])
    p.add_argument('--min-exact-match', type=float)
    p.add_argument('--baseline', help='Previous regression JSON report')
    p.add_argument('--max-drop', type=float, default=0.0)
    a = p.parse_args()
    if a.min_exact_match is None and not a.baseline:
        p.error('Specify --min-exact-match or --baseline to define a regression gate')
    for value in (a.min_exact_match, a.max_drop):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            p.error('thresholds must be finite and between 0 and 1')
    refs = corpus(a.data)
    baseline = None
    if a.baseline:
        baseline = json.loads(Path(a.baseline).read_text(encoding='utf8'))
        if (not isinstance(baseline, dict) or
                type(baseline.get('exact_match')) not in (int, float) or
                not math.isfinite(baseline['exact_match']) or
                not 0 <= baseline['exact_match'] <= 1 or
                type(baseline.get('examples')) is not int or baseline['examples'] != len(refs)):
            raise ValueError('baseline must contain a valid exact_match and matching examples count')
    with tempfile.TemporaryDirectory() as d:
        pred = Path(d) / 'pred.jsonl'
        subprocess.run([sys.executable, '-m', 'micronmt.predict', '--checkpoint', a.checkpoint,
                        '--input', a.data, '--output', str(pred), '--source', a.source], check=True)
        exact = exact_match(a.data, pred)
    passed = a.min_exact_match is None or exact >= a.min_exact_match
    if baseline is not None:
        passed = passed and exact + a.max_drop >= baseline['exact_match']
    print(json.dumps({'examples': len(refs), 'exact_match': exact, 'passed': passed}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
