import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class DownloadTests(unittest.TestCase):
    def test_pinned_streaming_and_manifest(self):
        spec = importlib.util.spec_from_file_location('download_parallel', Path(__file__).resolve().parents[1] / 'scripts/download_parallel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []

        def load_dataset(**kwargs):
            calls.append(kwargs)
            return [{'translation': {'ru': None, 'en': 'bad'}},
                    {'translation': {'ru': 'Привет', 'en': 'Hello'}},
                    {'translation': {'ru': 'Мир', 'en': 'World'}}]

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'pairs.jsonl'
            args = ['download_parallel', '--dataset', 'example/corpus', '--revision', 'abc123',
                    '--license', 'unknown', '--out', str(out), '--limit', '1']
            with patch.object(sys, 'argv', args), patch.dict(sys.modules, {'datasets': types.SimpleNamespace(load_dataset=load_dataset)}), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            manifest = json.loads(out.with_suffix('.manifest.json').read_text())
            self.assertEqual(manifest['count'], 1)
            self.assertEqual(manifest['revision'], 'abc123')
            self.assertFalse(manifest['production_approved'])
            self.assertEqual(len(manifest['sha256']), 64)
            self.assertTrue(calls[0]['streaming'])
            self.assertEqual(calls[0]['revision'], 'abc123')


if __name__ == '__main__':
    unittest.main()
