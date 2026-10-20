import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from micronmt.checkpoint import load_checkpoint, tokenizer_metadata
from micronmt.model import MicroTransformer
from micronmt.tokenizer import SimpleTokenizer


class PackageTests(unittest.TestCase):
    def test_simple_package_roundtrip(self):
        tok = SimpleTokenizer.from_texts(['hello привет'])
        config = dict(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                      dim_feedforward=16, dropout=0.0, max_len=12)
        model = MicroTransformer(tok.vocab_size, **config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'source.pt'
            torch.save(dict(model=model.state_dict(), config=config, tokenizer=tokenizer_metadata(tok)), checkpoint)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run([sys.executable, 'scripts/package_model.py', '--checkpoint', str(checkpoint),
                                     '--out', str(root / 'package')], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoint.unlink()
            _, restored, _ = load_checkpoint(root / 'package/model.pt')
            self.assertEqual(restored.vocab, tok.vocab)
            metadata = json.loads((root / 'package/metadata.json').read_text())
            self.assertEqual(metadata['files'], ['model.pt'])


if __name__ == '__main__':
    unittest.main()
