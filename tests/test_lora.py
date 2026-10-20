import copy
import io
import unittest

import torch
from torch import nn

from micronmt.lora import LoRALinear, adapter_state, inject_lora, load_adapter
from micronmt.model import MicroTransformer


class LoRATests(unittest.TestCase):
    def make_model(self):
        return MicroTransformer(24, d_model=8, nhead=2, encoder_layers=1,
                                decoder_layers=1, dim_feedforward=16,
                                dropout=0, max_len=8)

    def test_trainable_parameters_and_update(self):
        model = self.make_model()
        original = {k: v.detach().clone() for k, v in model.named_parameters()}
        self.assertEqual(inject_lora(model, rank=2, dropout=0), 5)
        trainable = {k for k, v in model.named_parameters() if v.requires_grad}
        self.assertEqual(trainable, set(adapter_state(model)))
        self.assertEqual(len(trainable), 10)
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                self.assertIsInstance(module.out_proj, nn.Linear)
        self.assertIsInstance(model.output, LoRALinear)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        src = torch.tensor([[2, 4, 3]])
        tgt = torch.tensor([[2, 5, 3]])
        model(src, tgt).square().mean().backward()
        for name, p in model.named_parameters():
            if name in trainable:
                self.assertIsNotNone(p.grad, name)
            else:
                self.assertIsNone(p.grad, name)
        optimizer.step()
        for name, value in original.items():
            parts = name.rsplit('.', 1)
            wrapped = parts[0] + '.base.' + parts[1]
            current = dict(model.named_parameters())
            # The tied output/embedding weight is listed once.
            torch.testing.assert_close(current.get(wrapped, current.get(name)), value)

    def test_adapter_roundtrip_train_and_eval(self):
        torch.manual_seed(7)
        model = self.make_model()
        restored = copy.deepcopy(model)
        inject_lora(model, rank=2, dropout=0)
        inject_lora(restored, rank=2, dropout=0)
        src = torch.tensor([[2, 4, 3]])
        tgt = torch.tensor([[2, 5, 3]])
        model.eval()
        with torch.no_grad():
            baseline = model(src, tgt).clone()
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.B.normal_(std=.2)
        buffer = io.BytesIO()
        torch.save(adapter_state(model), buffer)
        buffer.seek(0)
        load_adapter(restored, torch.load(buffer, weights_only=True))
        for training in (True, False):
            model.train(training)
            restored.train(training)
            with torch.no_grad():
                actual = model(src, tgt)
                torch.testing.assert_close(actual, restored(src, tgt))
                self.assertFalse(torch.allclose(actual, baseline))
        with self.assertRaises(ValueError):
            load_adapter(restored, {})

    def test_root_linear_state_and_dtype(self):
        module = LoRALinear(nn.Linear(3, 4).double(), rank=2)
        self.assertEqual(module.A.dtype, torch.float64)
        self.assertEqual(set(adapter_state(module)), {'A', 'B'})
        load_adapter(module, adapter_state(module))


if __name__ == '__main__':
    unittest.main()
