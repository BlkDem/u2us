import json
from pathlib import Path
import tempfile
import unittest

import torch

from micronmt.checkpoint import load_checkpoint
from micronmt.sp_tokenizer import SentencePieceTokenizer
from scripts.overfit_check import ROOT, batch, evaluate, select_rows, token_stats


class OverfitTest(unittest.TestCase):
    def test_padding_ignored(self):
        logits = torch.tensor([[[0., 10., 0.], [10., 0., 0.]]])
        loss, accuracy = token_stats(logits, torch.tensor([[1, 0]]))
        self.assertEqual(accuracy, 1.)
        self.assertLess(loss.item(), .001)
        src, tgt = batch([{'src_ids': [2, 3], 'tgt_ids': [2, 4, 3]},
                          {'src_ids': [2, 5, 3], 'tgt_ids': [2, 3]}])
        self.assertEqual(src.tolist(), [[2, 3, 0], [2, 5, 3]])
        self.assertEqual(tgt.tolist(), [[2, 4, 3], [2, 3, 0]])

    def test_selection_never_truncates(self):
        tok = SentencePieceTokenizer(ROOT / 'data/tokenizer/real_baseline/shared.model')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.jsonl'
            rows = [{'src': 'Привет ' * 100, 'tgt': 'Hello'},
                    {'src': 'Привет', 'tgt': 'Hello'}]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            chosen = select_rows(path, tok, count=1)
            self.assertEqual(chosen[0]['train_line'], 2)
            for key, text in [('src_ids', '<ru2en> Привет'), ('tgt_ids', 'Hello')]:
                self.assertEqual(chosen[0][key], [2, *tok.processor.encode(text), 3])
            with self.assertRaises(ValueError):
                select_rows(path, tok, count=2)

    def test_saved_checkpoint_embeds_tokenizer_and_reproduces_metrics(self):
        checkpoint = ROOT / 'runs/overfit/model.pt'
        if not checkpoint.exists():
            self.skipTest('run scripts/overfit_check.py first')
        torch.set_num_threads(2)
        report = json.loads((ROOT / 'reports/overfit/summary.json').read_text())
        model, tok, saved = load_checkpoint(checkpoint)
        self.assertIn('model_proto', saved['tokenizer'])
        measured = evaluate(model, tok, report['selected_pairs'], saved['global_step'])
        for key in ('loss', 'token_accuracy', 'exact_match', 'chrF'):
            self.assertAlmostEqual(measured[key], report['history'][-1][key], places=6)
        self.assertGreaterEqual(measured['exact_match'], .95)


if __name__ == '__main__':
    unittest.main()
