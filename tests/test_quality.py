import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from micronmt.evaluate import aligned, corpus
from scripts import leakage_check, prepare_data, quality_filter, regression


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def file(self, name, rows):
        path = self.root / name
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
        return path

    def test_schema(self):
        for row in ([], None, {}, {'src': 1, 'tgt': 'hello'}, {'src': 'hi', 'tgt': None}):
            path = self.file('bad', [row])
            for reader in (corpus, leakage_check.keys):
                with self.assertRaises(ValueError):
                    reader(path)
            with self.assertRaises(ValueError):
                prepare_data.read_rows([path], 2000, 3)

    def test_malformed_and_empty_evaluation(self):
        for text in ('', '\n', '{bad}\n'):
            path = self.root / 'bad'
            path.write_text(text)
            with self.assertRaises(ValueError):
                corpus(path)

    def test_alignment(self):
        ref = self.file('ref', [{'src': 'a', 'tgt': 'b'}])
        for rows in ([{'src': 'c', 'tgt': 'b'}],
                     [{'src': 'a', 'tgt': 'b'}] * 2,
                     [{'src': 'a', 'tgt': 4}]):
            with self.assertRaises(ValueError):
                aligned(ref, self.file('hyp', rows))
        self.assertEqual(regression.exact_match(ref, self.file('hyp', [{'src': 'a', 'tgt': ''}])), 0)

    def test_connected_split(self):
        rows = [{'src': 'A', 'tgt': 'one'}, {'src': 'B', 'tgt': 'one'},
                {'src': 'b', 'tgt': 'two'}, {'src': 'two', 'tgt': 'three'}]
        rows += [{'src': f's{i}', 'tgt': f't{i}'} for i in range(30)]
        splits = prepare_data.split_rows(rows, .2, .2, 42)
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        sets = {name: leakage_check.keys(self.file(name, part)) for name, part in splits.items()}
        for a, b in [('train', 'test'), ('train', 'valid'), ('valid', 'test')]:
            self.assertFalse(sets[a] & sets[b])
        self.assertTrue(any(all(row in part for row in rows[:4]) for part in splits.values()))
        self.assertEqual(splits, prepare_data.split_rows(rows, .2, .2, 42))
        for ratios in ((-.1, .1), (.5, .5), (float('nan'), 0)):
            with self.assertRaises(ValueError):
                prepare_data.split_rows(rows, *ratios, 42)

    def test_quality_schema(self):
        path = self.file('input', [{'src': [], 'tgt': 'hello'}])
        args = ['quality_filter', '--input', str(path), '--output', str(self.root / 'out'),
                '--report', str(self.root / 'report')]
        with patch.object(sys, 'argv', args), self.assertRaises(ValueError):
            quality_filter.main()

    def test_regression_exit_and_baseline(self):
        data = self.file('data', [{'src': 'hello', 'tgt': 'world'}])
        baseline = self.root / 'baseline'
        baseline.write_text(json.dumps({'examples': 1, 'exact_match': 1.0}))

        def predict(args, **kwargs):
            Path(args[args.index('--output') + 1]).write_text(json.dumps({'src': 'hello', 'tgt': 'wrong'}) + '\n')

        base = ['regression', '--checkpoint', 'unused', '--data', str(data)]
        for extra, status in [([], 2), (['--min-exact-match', '.5'], 1),
                              (['--baseline', str(baseline)], 1),
                              (['--baseline', str(baseline), '--max-drop', '1'], 0)]:
            with patch.object(sys, 'argv', base + extra), patch.object(regression.subprocess, 'run', side_effect=predict), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    regression.main()
                self.assertEqual(exc.exception.code, status)


if __name__ == '__main__':
    unittest.main()
