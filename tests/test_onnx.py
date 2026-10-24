import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DEPENDENCIES = ('torch', 'numpy', 'onnx', 'onnxruntime')
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)


@unittest.skipUnless(AVAILABLE, 'requires torch, numpy, onnx and onnxruntime')
class OnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import onnxruntime as ort
        import torch
        from micronmt.checkpoint import tokenizer_metadata
        from micronmt.model import MicroTransformer
        from micronmt.tokenizer import SimpleTokenizer

        cls.torch, cls.np = torch, np
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        cls.temp = tempfile.TemporaryDirectory(prefix='onnx-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.root = Path(__file__).resolve().parents[1]
        cls.env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        cls.env['PYTHONPATH'] = str(cls.root) + os.pathsep + cls.env.get('PYTHONPATH', '')
        torch.manual_seed(123)
        cls.tok = SimpleTokenizer.from_texts(['hello world привет мир'])
        config = dict(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                      dim_feedforward=16, dropout=0.0, max_len=8)
        cls.model = MicroTransformer(cls.tok.vocab_size, **config).eval()
        assert cls.model.output.weight is cls.model.embedding.weight
        assert cls.model.embedding.weight[0].abs().sum().item() == 0
        checkpoint = cls.directory / 'tiny.pt'
        torch.save(dict(model=cls.model.state_dict(), config=config,
                        tokenizer=tokenizer_metadata(cls.tok)), checkpoint)
        cls.export = cls.directory / 'export'
        subprocess.run([sys.executable, '-m', 'micronmt.export_seq2seq',
                        '--checkpoint', str(checkpoint), '--out-dir', str(cls.export),
                        '--sequence-len', '8'], cwd=cls.root, env=cls.env,
                       check=True, capture_output=True, text=True)
        checkpoint.unlink()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cls.encoder = ort.InferenceSession(str(cls.export / 'encoder.onnx'), options,
                                          providers=['CPUExecutionProvider'])
        cls.decoder = ort.InferenceSession(str(cls.export / 'decoder.onnx'), options,
                                          providers=['CPUExecutionProvider'])

    def source(self, text='привет'):
        src = self.np.zeros((1, 8), dtype=self.np.int64)
        ids = self.tok.encode('<ru2en> ' + text, 8)
        src[0, :len(ids)] = ids
        return src

    def test_padded_logits(self):
        from micronmt.export_seq2seq import EncoderWrapper
        for text in ('привет', 'привет мир hello'):
            with self.subTest(text=text):
                src = self.source(text)
                tgt = self.np.array([[2, 6, 7, 3, 0, 0, 0, 0]], dtype=self.np.int64)
                memory = self.encoder.run(['memory'], {'src': src})[0]
                actual = self.decoder.run(['logits'], {'memory': memory, 'tgt': tgt, 'src': src})[0]
                with self.torch.no_grad():
                    expected_memory = EncoderWrapper(self.model)(self.torch.from_numpy(src))
                    expected = self.model(self.torch.from_numpy(src), self.torch.from_numpy(tgt))
                self.np.testing.assert_allclose(memory, expected_memory.numpy(), atol=2e-5, rtol=2e-5)
                self.np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-5)

    def generated(self):
        src = self.source()
        memory = self.encoder.run(['memory'], {'src': src})[0]
        tgt = self.np.zeros((1, 8), dtype=self.np.int64)
        tgt[0, 0] = self.tok.bos_id
        for i in range(1, 8):
            logits = self.decoder.run(['logits'], {'memory': memory, 'tgt': tgt, 'src': src})[0]
            tgt[0, i] = logits[0, i - 1].argmax()
            if tgt[0, i] == self.tok.eos_id:
                break
        return tgt[:, :i + 1]

    def test_autoregressive_tokens(self):
        expected = self.model.greedy(self.torch.from_numpy(self.source()), max_len=8)
        self.np.testing.assert_array_equal(self.generated(), expected.numpy())

    @unittest.skipUnless(importlib.util.find_spec('sentencepiece'), 'requires sentencepiece')
    def test_embedded_sentencepiece_standalone(self):
        from micronmt.checkpoint import tokenizer_metadata
        from micronmt.sp_tokenizer import SentencePieceTokenizer, train_sentencepiece
        from micronmt.model import MicroTransformer

        directory = self.directory / 'sentencepiece'
        directory.mkdir()
        corpus = directory / 'corpus.txt'
        corpus.write_text('hello world привет мир\nhello friend привет друг\n', encoding='utf8')
        train_sentencepiece([corpus], directory / 'tokenizer', vocab_size=128)
        tok = SentencePieceTokenizer(directory / 'tokenizer.model')
        config = dict(d_model=8, nhead=2, encoder_layers=1, decoder_layers=1,
                      dim_feedforward=16, dropout=0.0, max_len=8)
        model = MicroTransformer(tok.vocab_size, **config).eval()
        checkpoint = directory / 'tiny.pt'
        self.torch.save(dict(model=model.state_dict(), config=config,
                             tokenizer=tokenizer_metadata(tok)), checkpoint)
        (directory / 'tokenizer.model').unlink()
        export = directory / 'export'
        subprocess.run([sys.executable, '-m', 'micronmt.export_seq2seq',
                        '--checkpoint', str(checkpoint), '--out-dir', str(export),
                        '--sequence-len', '8'], cwd=self.root, env=self.env,
                       check=True, capture_output=True, text=True)
        checkpoint.unlink()
        metadata = json.loads((export / 'metadata.json').read_text())
        self.assertNotIn('model_proto', metadata['tokenizer'])
        self.assertEqual((export / 'tokenizer.model').read_bytes(),
                         tok.processor.serialized_model_proto())
        src = self.torch.tensor([tok.encode('<ru2en> привет', 8)])
        expected = tok.decode(model.greedy(src, max_len=8)[0])
        result = subprocess.run([sys.executable, '-m', 'micronmt.onnx_translate',
                                 '--model-dir', str(export), '--text', 'привет'],
                                cwd=self.directory, env=self.env, check=True,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), expected.strip())

    def test_standalone_cli_without_checkpoint(self):
        metadata = json.loads((self.export / 'metadata.json').read_text())
        self.assertFalse(Path(metadata['checkpoint']).exists())
        result = subprocess.run([sys.executable, '-m', 'micronmt.onnx_translate',
                                 '--model-dir', str(self.export), '--text', 'привет'],
                                cwd=self.directory, env=self.env, check=True,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), self.tok.decode(self.generated()[0]))


if __name__ == '__main__':
    unittest.main()
