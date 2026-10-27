import argparse
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from .api import Translator, load_terminology
from .evaluate import corpus

DEFAULT_DECODING = {'beam_size': 1, 'length_penalty': 0.6,
                    'no_repeat_ngram_size': 0, 'repetition_penalty': 1.0}


def normalize_config(config):
    merged = dict(DEFAULT_DECODING)
    if isinstance(config, dict):
        merged.update({k: config[k] for k in DEFAULT_DECODING if k in config})
    beam_size = merged['beam_size']
    length_penalty = merged['length_penalty']
    no_repeat_ngram_size = merged['no_repeat_ngram_size']
    repetition_penalty = merged['repetition_penalty']
    if not isinstance(beam_size, int) or isinstance(beam_size, bool) or beam_size < 1:
        raise ValueError('beam_size must be an integer >= 1')
    if not isinstance(no_repeat_ngram_size, int) or isinstance(no_repeat_ngram_size, bool) or no_repeat_ngram_size < 0:
        raise ValueError('no_repeat_ngram_size must be an integer >= 0')
    if not isinstance(length_penalty, (int, float)) or isinstance(length_penalty, bool) or length_penalty < 0:
        raise ValueError('length_penalty must be a number >= 0')
    if not isinstance(repetition_penalty, (int, float)) or isinstance(repetition_penalty, bool) or repetition_penalty < 1:
        raise ValueError('repetition_penalty must be a number >= 1')
    return {'beam_size': beam_size, 'length_penalty': float(length_penalty),
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'repetition_penalty': float(repetition_penalty)}


def expand_configs(beam_sizes=None, length_penalties=None,
                   no_repeat_ngram_sizes=None, repetition_penalties=None):
    beam_sizes = beam_sizes or [DEFAULT_DECODING['beam_size']]
    length_penalties = length_penalties or [DEFAULT_DECODING['length_penalty']]
    no_repeat_ngram_sizes = no_repeat_ngram_sizes or [DEFAULT_DECODING['no_repeat_ngram_size']]
    repetition_penalties = repetition_penalties or [DEFAULT_DECODING['repetition_penalty']]
    configs, seen = [], set()
    for beam, penalty, ngram, repetition in itertools.product(
            beam_sizes, length_penalties, no_repeat_ngram_sizes, repetition_penalties):
        config = normalize_config({'beam_size': beam, 'length_penalty': penalty,
                                   'no_repeat_ngram_size': ngram,
                                   'repetition_penalty': repetition})
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def config_id(config):
    return 'b%s-lp%s-nr%s-rp%s' % (config['beam_size'], config['length_penalty'],
                                   config['no_repeat_ngram_size'], config['repetition_penalty'])


class LocalBackend:
    type = 'local'

    def __init__(self, checkpoint, device='cpu', max_len=None, terminology=None):
        self.checkpoint = checkpoint
        self.translator = Translator(checkpoint, device=device, max_len=max_len, terminology=terminology)

    def describe(self):
        return {'type': self.type, 'checkpoint': str(self.checkpoint),
                'max_len': self.translator.max_len}

    def translate(self, text, source, config):
        return self.translator.translate(text, source=source, **config)['translation']


class ApiBackend:
    type = 'api'

    def __init__(self, base_url, api_key=None, model=None, timeout=120):
        url = base_url.rstrip('/')
        if not url.endswith('/v1'):
            url += '/v1'
        self.base_url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def describe(self):
        return {'type': self.type, 'base_url': self.base_url, 'model': self.model}

    def translate(self, text, source, config):
        payload = {'prompt': text, 'source': source}
        payload['model'] = self.model or ('ru2en' if source == 'ru' else 'en2ru')
        payload.update(config)
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        data = json.dumps(payload).encode('utf-8')
        req = request.Request(self.base_url + '/completions', data=data, headers=headers, method='POST')
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')
            raise RuntimeError(f'API request failed ({exc.code}): {detail}') from exc
        except error.URLError as exc:
            raise RuntimeError(f'API request failed: {exc.reason}') from exc
        return body['choices'][0]['text']


def compute_metrics(refs, hyps):
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('sacrebleu is required for evaluation metrics') from exc
    return {'BLEU': sacrebleu.corpus_bleu(hyps, [refs]).score,
            'chrF': sacrebleu.corpus_chrf(hyps, [refs]).score}


def compute_comet(rows, hyps, model_name):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    data = [{'src': row['src'], 'mt': hyp, 'ref': row['tgt']} for row, hyp in zip(rows, hyps)]
    return float(model.predict(data, gpus=0)['system_score'])


def evaluate_configs(backend, rows, source, configs, hyp_dir=None, stem='pred'):
    refs = [row['tgt'] for row in rows]
    results = []
    for config in configs:
        started = time.time()
        hyps = [backend.translate(row['src'], source, config) for row in rows]
        entry = {'decoding': config, 'examples': len(rows),
                 'seconds': round(time.time() - started, 3)}
        entry.update(compute_metrics(refs, hyps))
        entry['_hypotheses'] = hyps
        if hyp_dir:
            Path(hyp_dir).mkdir(parents=True, exist_ok=True)
            path = Path(hyp_dir) / f'{stem}_{config_id(config)}.jsonl'
            lines = [json.dumps({'src': row['src'], 'tgt': hyp, 'decoding': config}, ensure_ascii=False)
                     for row, hyp in zip(rows, hyps)]
            path.write_text('\n'.join(lines) + '\n', encoding='utf8')
            entry['hypothesis'] = str(path)
        results.append(entry)
    best = max(results, key=lambda item: (item['BLEU'], item['chrF']))
    return results, best


def _baseline_values(baseline):
    if not isinstance(baseline, dict):
        raise ValueError('baseline must be a JSON object')
    candidate = baseline.get('best') if isinstance(baseline.get('best'), dict) else baseline
    bleu, chrf = candidate.get('BLEU'), candidate.get('chrF')
    if not isinstance(bleu, (int, float)) or not isinstance(chrf, (int, float)):
        raise ValueError('baseline must contain numeric BLEU and chrF')
    return float(bleu), float(chrf)


def check_gate(best, baseline=None, min_bleu=None, min_chrf=None, max_drop=0.0):
    if not isinstance(max_drop, (int, float)) or not math.isfinite(max_drop) or max_drop < 0:
        raise ValueError('max_drop must be a finite number >= 0')
    for name, value in (('min_bleu', min_bleu), ('min_chrf', min_chrf)):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f'{name} must be a finite number')
    checks = {}
    passed = True
    if min_bleu is not None:
        ok = best['BLEU'] >= min_bleu
        checks['min_bleu'] = {'threshold': min_bleu, 'value': best['BLEU'], 'passed': ok}
        passed = passed and ok
    if min_chrf is not None:
        ok = best['chrF'] >= min_chrf
        checks['min_chrf'] = {'threshold': min_chrf, 'value': best['chrF'], 'passed': ok}
        passed = passed and ok
    if baseline is not None:
        base_bleu, base_chrf = _baseline_values(baseline)
        ok_bleu = best['BLEU'] + max_drop >= base_bleu
        ok_chrf = best['chrF'] + max_drop >= base_chrf
        checks['baseline'] = {'BLEU': {'baseline': base_bleu, 'value': best['BLEU'], 'passed': ok_bleu},
                              'chrF': {'baseline': base_chrf, 'value': best['chrF'], 'passed': ok_chrf},
                              'max_drop': max_drop}
        passed = passed and ok_bleu and ok_chrf
    return {'passed': bool(passed), 'checks': checks}


def _load_configs(path):
    data = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(data, list):
        raise ValueError('config file must contain a JSON list of decoding objects')
    return [normalize_config(item) for item in data]


def _merge_configs(configs):
    merged, seen = [], set()
    for config in configs:
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            merged.append(config)
    return merged


def _print_table(results, best):
    print('%-32s %8s %8s %10s' % ('config', 'BLEU', 'chrF', 'seconds'))
    for item in results:
        print('%-32s %8.2f %8.2f %10.1f' % (config_id(item['decoding']), item['BLEU'],
                                            item['chrF'], item['seconds']))
    print('best: %s (BLEU %.2f, chrF %.2f)' % (config_id(best['decoding']), best['BLEU'], best['chrF']))


def main():
    parser = argparse.ArgumentParser(description='Evaluation harness for translation checkpoints and the HTTP API')
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument('--checkpoint')
    backend.add_argument('--api-url')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--api-model')
    parser.add_argument('--data', required=True)
    parser.add_argument('--source', choices=['ru', 'en'], default='ru')
    parser.add_argument('--target', choices=['ru', 'en'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--terms')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', help='JSON file with a list of decoding configs')
    parser.add_argument('--beam-size', type=int, nargs='+')
    parser.add_argument('--length-penalty', type=float, nargs='+')
    parser.add_argument('--no-repeat-ngram-size', type=int, nargs='+')
    parser.add_argument('--repetition-penalty', type=float, nargs='+')
    parser.add_argument('--hyp-dir')
    parser.add_argument('--out')
    parser.add_argument('--comet-model')
    parser.add_argument('--baseline')
    parser.add_argument('--min-bleu', type=float)
    parser.add_argument('--min-chrf', type=float)
    parser.add_argument('--max-drop', type=float, default=0.0)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    target = args.target or ('en' if args.source == 'ru' else 'ru')
    if target == args.source:
        parser.error('--target must differ from --source')
    if args.limit < 0:
        parser.error('--limit must be >= 0')

    if args.checkpoint:
        terminology = load_terminology(args.terms) if args.terms else None
        engine = LocalBackend(args.checkpoint, device=args.device, max_len=args.max_len,
                              terminology=terminology)
    else:
        engine = ApiBackend(args.api_url, api_key=args.api_key, model=args.api_model)

    rows = corpus(args.data)
    if args.limit:
        rows = rows[:args.limit]

    configs = _load_configs(args.config) if args.config else []
    sweep = expand_configs(args.beam_size, args.length_penalty,
                           args.no_repeat_ngram_size, args.repetition_penalty)
    if not configs:
        configs = sweep
    else:
        configs = _merge_configs(configs + sweep)

    stem = Path(args.data).stem
    results, best = evaluate_configs(engine, rows, args.source, configs,
                                     hyp_dir=args.hyp_dir, stem=stem)
    hyps = best.pop('_hypotheses')
    for item in results:
        if item is not best:
            item.pop('_hypotheses', None)

    if args.comet_model:
        best['COMET'] = compute_comet(rows, hyps, args.comet_model)

    report = {
        'created': datetime.now(timezone.utc).isoformat(),
        'backend': engine.describe(),
        'data': str(args.data),
        'source': args.source,
        'target': target,
        'examples': len(rows),
        'configs': results,
        'best': best,
    }
    gate_requested = any(value is not None for value in (args.min_bleu, args.min_chrf, args.baseline))
    if gate_requested:
        baseline = json.loads(Path(args.baseline).read_text(encoding='utf8')) if args.baseline else None
        report['gate'] = check_gate(best, baseline, args.min_bleu, args.min_chrf, args.max_drop)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    if not args.quiet:
        _print_table(results, best)
    else:
        print('best: %s (BLEU %.2f, chrF %.2f)' % (config_id(best['decoding']), best['BLEU'], best['chrF']))
    if 'gate' in report:
        raise SystemExit(0 if report['gate']['passed'] else 1)


if __name__ == '__main__':
    main()
