import tempfile
import unittest
from pathlib import Path
import torch
from micronmt.model import MicroTransformer
from micronmt.train import build_model
from micronmt.checkpoint import load_checkpoint, tokenizer_metadata
from micronmt.tokenizer import SimpleTokenizer
from micronmt.lora import inject_lora


class InitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config=dict(d_model=32,nhead=4,encoder_layers=1,decoder_layers=1,
                         dim_feedforward=64,dropout=0.,max_len=12)

    def test_stats_pad_tying_and_seed_rng_parity(self):
        torch.manual_seed(42)
        legacy=MicroTransformer(2000,**self.config,init_mode='legacy')
        state=torch.get_rng_state()
        torch.manual_seed(42)
        model=MicroTransformer(2000,**self.config)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertIs(model.embedding.weight,model.output.weight)
        self.assertEqual(model.embedding.weight[0].abs().sum().item(),0)
        self.assertAlmostEqual(model.embedding.weight[1:].std().item(),32**-.5,places=3)
        with torch.random.fork_rng():
            torch.manual_seed(424242)
            expected=torch.empty_like(model.embedding.weight).normal_(0,32**-.5)
            expected[0].zero_()
        torch.testing.assert_close(model.embedding.weight,expected,rtol=0,atol=0)
        for k,v in legacy.state_dict().items():
            if k not in ('embedding.weight','output.weight'):
                torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        with self.assertRaises(ValueError): MicroTransformer(20,init_mode='invalid')

    def test_old_checkpoint_exact_logits_and_optional_config(self):
        tok=SimpleTokenizer.from_texts(['hello привет'])
        old=MicroTransformer(tok.vocab_size,**self.config,init_mode='legacy').eval()
        src=torch.tensor([[2,4,6,3,0]]); tgt=torch.tensor([[2,6,3]])
        expected=old(src,tgt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'old.pt'
            torch.save(dict(model=old.state_dict(),config=self.config,tokenizer=tokenizer_metadata(tok)),path)
            loaded,_,_=load_checkpoint(path)
            torch.testing.assert_close(loaded(src,tgt),expected,rtol=0,atol=0)
            self.assertIs(loaded.embedding.weight,loaded.output.weight)
        model=build_model(tok.vocab_size,dict(self.config,init_mode='legacy'))
        model.load_state_dict(old.state_dict()); model.eval()
        torch.testing.assert_close(model(src,tgt),expected,rtol=0,atol=0)

    def test_controlled_lora_preserves_base_logits_and_tying(self):
        model=MicroTransformer(24,**self.config).eval()
        src=torch.tensor([[2,4,3]]); tgt=torch.tensor([[2,5]])
        expected=model(src,tgt)
        inject_lora(model,rank=2,dropout=0)
        self.assertIs(model.embedding.weight,model.output.base.weight)
        torch.testing.assert_close(model(src,tgt),expected,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
