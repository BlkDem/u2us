import copy
import unittest
import torch
from scripts.diagnose_training import (CONFIG, controlled_init, token_masks, select_short,
                                       lengths, normalize, evaluate)
from micronmt.model import MicroTransformer
from micronmt.sp_tokenizer import SentencePieceTokenizer


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tok=SentencePieceTokenizer('data/tokenizer/real_baseline/shared.model')

    def test_masks_exclude_pad_separate_content(self):
        targets=torch.tensor([[0,3,6,7]])
        table=torch.zeros(8,dtype=torch.bool); table[6]=True
        valid,content=token_masks(targets,table)
        self.assertEqual(valid.tolist(),[[False,True,True,True]])
        self.assertEqual(content.tolist(),[[False,False,True,False]])

    def test_controlled_init_tying_pad_other_weights_rng(self):
        a=MicroTransformer(self.tok.vocab_size,**CONFIG); b=copy.deepcopy(a)
        before=torch.get_rng_state().clone(); controlled_init(b)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertIs(b.embedding.weight,b.output.weight)
        self.assertEqual(b.embedding.weight[0].abs().sum().item(),0)
        self.assertAlmostEqual(b.embedding.weight[1:].std().item(),.125,places=2)
        for k,v in a.state_dict().items():
            if k not in ('embedding.weight','output.weight'):
                self.assertTrue(torch.equal(v,b.state_dict()[k]),k)

    def test_selection_no_truncation(self):
        rows=[dict(src='Привет',tgt='Hello'),dict(src='длинный '*200,tgt='long '*200)]
        picked=select_short(rows,self.tok,100)
        self.assertEqual(len(picked),1)
        self.assertEqual(picked[0]['original_index'],0)
        self.assertLessEqual(max(lengths(picked[0],self.tok)),64)
        for k,prefix in [('src','<ru2en> '),('tgt','')]:
            ids=self.tok.encode(prefix+picked[0][k],64)
            self.assertEqual(ids[1:-1],self.tok.processor.encode(prefix+picked[0][k]))
            self.assertEqual(ids[-1],3)

    def test_metrics_immediate_eos(self):
        tok=self.tok
        class EosModel:
            def eval(self): return self
            def __call__(self,src,tgt):
                logits=torch.zeros((*tgt.shape,tok.vocab_size)); logits[:,:,3]=10
                return logits
            def greedy(self,src,max_len): return torch.tensor([[2,3]]*len(src))
        metrics=evaluate(EosModel(),[dict(src='Привет',tgt='Hello world.')],tok)
        self.assertEqual(metrics['nonempty_rate'],0)
        self.assertEqual(metrics['immediate_eos_rate'],1)
        self.assertEqual(metrics['premature_eos_rate'],1)
        self.assertEqual(metrics['chrF'],0)
        self.assertEqual(metrics['content_token_accuracy'],0)
        self.assertGreater(metrics['ce'],0)
        self.assertEqual(normalize('Ａ  B'),'A B')


if __name__=='__main__': unittest.main()
