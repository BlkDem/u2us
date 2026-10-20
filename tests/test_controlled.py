import json
from pathlib import Path
import unittest

import yaml
from scripts.train_controlled import output_stats, repetition, truncation
from tests.test_training import TrainingTests


class ControlledTests(unittest.TestCase):
    def test_configs(self):
        for direction in ('ru_en', 'en_ru'):
            c = yaml.safe_load(Path(f'configs/controlled_{direction}.yaml').read_text())
            self.assertEqual(c['run_dir'], f'runs/controlled_{direction}')
            self.assertEqual((c['epochs'], c['lr'], c['batch_size']), (12, .001, 32))

    def test_repetition(self):
        self.assertTrue(repetition('oooooooo'))
        self.assertTrue(repetition('a b a b a b'))
        self.assertFalse(repetition('a normal sentence'))
        self.assertEqual(output_stats(['', 'oooooooo']), {'nonempty_rate': .5, 'repetition_rate': .5})

    def test_truncation(self):
        class Processor:
            def encode(self, text):
                return text.split()
        class Tok:
            processor = Processor()
        result = truncation([{'src': 'x '*62, 'tgt': 'x '*62}], Tok(), 'ru2en')
        self.assertEqual(result['src']['fraction'], 1.)
        self.assertEqual(result['tgt']['fraction'], 0.)


class HistoryTests(TrainingTests):
    def test_history_cli(self):
        history = self.root / 'history.jsonl'
        self.cli('--valid', self.data, '--eval-samples', 2, '--history', history)
        records = [json.loads(l) for l in history.read_text().splitlines()]
        self.assertEqual([r['epoch'] for r in records], [1, 2])
        self.assertEqual([r['global_step'] for r in records], [3, 6])
        for record in records:
            self.assertIsInstance(record['valid_chrf'], float)
            self.assertIsInstance(record['valid_loss'], float)
            self.assertEqual(len(record['samples']), 2)
        self.assertEqual(records[0]['samples'][0]['source'], records[1]['samples'][0]['source'])
