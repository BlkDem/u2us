import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from micronmt import train
from micronmt.data import collate as real_collate


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data.jsonl'
        self.data.write_text('\n'.join(json.dumps({'src': f'hello {i}', 'tgt': f'world {i}'}) for i in range(3)))
        self.config = dict(seed=7, d_model=8, nhead=2, encoder_layers=1,
                           decoder_layers=1, dim_feedforward=16, dropout=0.0,
                           max_len=8, batch_size=1, epochs=2, lr=0.001,
                           label_smoothing=0.0, run_dir=str(self.root / 'run'))
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))

    def cli(self, *extra, success=True):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-m', 'micronmt.train', '--config',
                                 str(self.config_path), '--data', str(self.data), *map(str, extra)],
                                capture_output=True, text=True, env=env)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def checkpoint(self):
        return torch.load(self.root / 'run' / 'last.pt', weights_only=False)

    def test_cli_stop_resume_and_partial_accumulation(self):
        self.cli('--grad-accum', 2, '--max-steps', 1)
        first = self.checkpoint()
        self.assertEqual((first['epoch'], first['batch_offset'], first['global_step']), (0, 1, 1))
        self.assertEqual(first['scheduler']['T_max'], 4)
        self.assertEqual(first['scheduler']['last_epoch'], 1)
        self.cli('--grad-accum', 2, '--resume', self.root / 'run' / 'last.pt')
        resumed = self.checkpoint()
        self.assertEqual((resumed['epoch'], resumed['global_step']), (2, 6))
        self.assertIn('best', resumed)
        self.assertIn('bad_epochs', resumed)
        self.assertIn('torch_rng_state', resumed)
        self.assertEqual(resumed['scheduler']['last_epoch'], 4)

    def test_epoch_resume_matches_uninterrupted(self):
        self.cli()
        expected = self.checkpoint()
        self.cli('--max-steps', 3)
        self.cli('--resume', self.root / 'run' / 'last.pt')
        actual = self.checkpoint()
        for key in expected['model']:
            self.assertTrue(torch.equal(expected['model'][key], actual['model'][key]), key)
        self.assertEqual(expected['best'], actual['best'])
        self.assertEqual(expected['bad_epochs'], actual['bad_epochs'])

    def test_direction_in_train_and_validation(self):
        directions = []
        def capture(*args, **kwargs):
            directions.append(kwargs.get('direction'))
            return real_collate(*args, **kwargs)
        args = ['train', '--config', str(self.config_path), '--data', str(self.data),
                '--valid', str(self.data), '--direction', 'en2ru', '--max-steps', '1']
        with patch.object(sys, 'argv', args), patch.object(train, 'collate', side_effect=capture):
            train.main()
        self.assertGreaterEqual(len(directions), 4)
        self.assertEqual(set(directions), {'en2ru'})

    def test_reject_invalid_arguments_and_empty_data(self):
        for args in [('--grad-accum', '0'), ('--max-steps', '-1'), ('--patience', '-1')]:
            self.cli(*args, success=False)
        self.data.write_text('')
        self.cli(success=False)

    def test_teacher_rejects_same_size_different_vocabulary(self):
        self.cli('--max-steps', 1)
        teacher = self.checkpoint()
        vocab = teacher['tokenizer']['vocab']
        vocab['different'] = vocab.pop('hello')
        path = self.root / 'teacher.pt'
        torch.save(teacher, path)
        result = self.cli('--teacher', path, success=False)
        self.assertIn('teacher tokenizer is incompatible', result.stderr)


if __name__ == '__main__':
    unittest.main()
