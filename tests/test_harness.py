import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from micronmt.checkpoint import tokenizer_metadata
from micronmt.harness import (ApiBackend, LocalBackend, check_gate, config_id,
                              evaluate_configs, expand_configs, normalize_config)
from micronmt.model import MicroTransformer
from micronmt.serve import build_server
from micronmt.tokenizer import SimpleTokenizer


def _checkpoint(directory):
    tokenizer = SimpleTokenizer.from_texts(['hello world привет мир'])
    config = dict(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                  dim_feedforward=16, dropout=0.0, max_len=12)
    model = MicroTransformer(tokenizer.vocab_size, **config).eval()
    path = Path(directory) / 'model.pt'
    torch.save(dict(model=model.state_dict(), config=config,
                    tokenizer=tokenizer_metadata(tokenizer)), path)
    return path


def _data(directory):
    path = Path(directory) / 'test.jsonl'
    rows = [{'src': 'привет', 'tgt': 'hello'}, {'src': 'мир', 'tgt': 'world'}]
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf8')
    return path, rows


class HarnessTests(unittest.TestCase):
    def test_expand_configs_default_and_dedupe(self):
        configs = expand_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]['beam_size'], 1)
        swept = expand_configs([1, 4], [0.6, 0.8], [0, 3], [1.0])
        self.assertEqual(len(swept), 8)
        keys = {tuple(sorted(config.items())) for config in swept}
        self.assertEqual(len(keys), 8)
        self.assertEqual(config_id(swept[0]), 'b1-lp0.6-nr0-rp1.0')

    def test_normalize_config(self):
        self.assertEqual(normalize_config({})['beam_size'], 1)
        self.assertEqual(normalize_config({'beam_size': 4, 'length_penalty': 1})['length_penalty'], 1.0)
        for bad in ({'beam_size': 0}, {'beam_size': 1.5}, {'length_penalty': -1},
                    {'no_repeat_ngram_size': -1}, {'repetition_penalty': 0.5},
                    {'beam_size': True}):
            with self.assertRaises(ValueError):
                normalize_config(bad)

    def test_local_backend_and_hypotheses(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _checkpoint(directory)
            data, rows = _data(directory)
            backend = LocalBackend(str(checkpoint))
            configs = expand_configs([1])
            results, best = evaluate_configs(backend, rows, 'ru', configs,
                                             hyp_dir=Path(directory) / 'hyp', stem='test')
            self.assertEqual(len(results), 1)
            self.assertIn('BLEU', best)
            self.assertIn('chrF', best)
            self.assertEqual(len(best['_hypotheses']), len(rows))
            hypothesis = Path(best['hypothesis'])
            self.assertTrue(hypothesis.exists())
            lines = hypothesis.read_text(encoding='utf8').splitlines()
            self.assertEqual(len(lines), len(rows))
            self.assertEqual(json.loads(lines[0])['src'], rows[0]['src'])

    def test_api_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _checkpoint(directory)
            server = build_server(str(checkpoint), host='127.0.0.1', port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = 'http://127.0.0.1:%d' % server.server_address[1]
                backend = ApiBackend(base, model='ru2en')
                _, rows = _data(directory)
                results, best = evaluate_configs(backend, rows, 'ru', expand_configs([1]))
                self.assertEqual(len(results), 1)
                self.assertIsInstance(best['_hypotheses'][0], str)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_check_gate(self):
        best = {'BLEU': 12.0, 'chrF': 30.0}
        passed = check_gate(best, min_bleu=10, min_chrf=25)
        self.assertTrue(passed['passed'])
        failed = check_gate(best, min_bleu=15)
        self.assertFalse(failed['passed'])
        baseline = {'best': {'BLEU': 12.5, 'chrF': 30.5}}
        strict = check_gate(best, baseline=baseline, max_drop=0.0)
        self.assertFalse(strict['passed'])
        tolerant = check_gate(best, baseline=baseline, max_drop=1.0)
        self.assertTrue(tolerant['passed'])
        with self.assertRaises(ValueError):
            check_gate(best, baseline={'BLEU': 'x', 'chrF': 1})
        with self.assertRaises(ValueError):
            check_gate(best, min_bleu=float('nan'))

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _checkpoint(directory)
            data, _ = _data(directory)
            out = Path(directory) / 'report.json'
            root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [sys.executable, '-m', 'micronmt.harness', '--checkpoint', str(checkpoint),
                 '--data', str(data), '--source', 'ru', '--out', str(out), '--quiet'],
                capture_output=True, text=True, cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text(encoding='utf8'))
            self.assertEqual(report['examples'], 2)
            self.assertEqual(report['source'], 'ru')
            self.assertEqual(report['target'], 'en')
            self.assertEqual(len(report['configs']), 1)
            self.assertNotIn('_hypotheses', report['best'])
            self.assertIn('BLEU', report['best'])


if __name__ == '__main__':
    unittest.main()
