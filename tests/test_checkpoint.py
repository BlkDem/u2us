import tempfile
import unittest
from pathlib import Path
import torch
from micronmt.checkpoint import load_checkpoint, tokenizer_metadata, load_tokenizer_metadata, tokenizer_fingerprint
from micronmt.model import MicroTransformer
from micronmt.tokenizer import SimpleTokenizer
from micronmt.data import collate, ParallelDataset


class CheckpointTests(unittest.TestCase):
    def test_simple_roundtrip(self):
        tok = SimpleTokenizer.from_texts(['hello привет'])
        config = dict(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                      dim_feedforward=16, dropout=0.0, max_len=12)
        model = MicroTransformer(tok.vocab_size, **config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(model=model.state_dict(), config=config, tokenizer=tokenizer_metadata(tok)), path)
            loaded, restored, _ = load_checkpoint(path)
            self.assertEqual(tok.vocab, restored.vocab)
            for a, b in zip(model.parameters(), loaded.parameters()):
                torch.testing.assert_close(a, b)

    def test_direction(self):
        tok = SimpleTokenizer.from_texts(['hello привет'])
        for direction, tag in [('ru2en', '<ru2en>'), ('en2ru', '<en2ru>')]:
            src, _ = collate([('hello', 'привет')], tok, direction=direction)
            self.assertEqual(src[0, 1].item(), tok.vocab[tag])
        with self.assertRaises(ValueError):
            collate([('hello', 'привет')], tok, direction='invalid')

    def test_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('null\n[]\n{"src": null, "tgt":"x"}\n{"src":"hello","tgt":"привет"}\n')
            self.assertEqual(len(ParallelDataset(path, SimpleTokenizer())), 1)

    def test_invalid_vocabulary(self):
        for vocab in [{'x': 99}, {'<pad>': 4}, {'x': 5}]:
            with self.assertRaises(ValueError):
                SimpleTokenizer(vocab)

    def test_sentencepiece_embedded(self):
        try:
            from micronmt.sp_tokenizer import SentencePieceTokenizer, train_sentencepiece
        except ImportError:
            self.skipTest('sentencepiece not installed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            corpus = path / 'corpus.txt'
            corpus.write_text('hello world привет мир\nhello friend привет друг\n', encoding='utf8')
            train_sentencepiece([corpus], path / 'tokenizer', vocab_size=128)
            tok = SentencePieceTokenizer(path / 'tokenizer.model')
            meta = tokenizer_metadata(tok)
            (path / 'tokenizer.model').unlink()
            restored = load_tokenizer_metadata(meta)
            self.assertEqual(tok.encode('<en2ru> hello'), restored.encode('<en2ru> hello'))
            self.assertEqual(tokenizer_fingerprint(tok), tokenizer_fingerprint(restored))


if __name__ == '__main__':
    unittest.main()
