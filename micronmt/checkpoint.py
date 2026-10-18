import torch
from .model import MicroTransformer
from .tokenizer import SimpleTokenizer
from .sp_tokenizer import SentencePieceTokenizer

KEYS=('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')
def load_checkpoint(path):
    q=torch.load(path,map_location='cpu')
    meta=q.get('tokenizer', {})
    if meta.get('type') == 'sentencepiece': tok=SentencePieceTokenizer(meta['model_file'])
    else: tok=SimpleTokenizer(meta.get('vocab', meta))
    m=MicroTransformer(tok.vocab_size, **{k:q['config'][k] for k in KEYS})
    m.load_state_dict(q['model']); m.eval()
    return m, tok, q
