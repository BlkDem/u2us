import torch
from .model import MicroTransformer
from .tokenizer import SimpleTokenizer
from .sp_tokenizer import SentencePieceTokenizer

KEYS=('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')
def load_checkpoint(path):
    q=torch.load(path,map_location='cpu')
    meta=q.get('tokenizer', {})
    if meta.get('type') == 'sentencepiece':
        model_file=meta['model_file']; model_file=str((__import__('pathlib').Path(path).parent/model_file)) if not __import__('pathlib').Path(model_file).is_absolute() else model_file
        tok=SentencePieceTokenizer(model_file)
    else: tok=SimpleTokenizer(meta.get('vocab', meta))
    m=MicroTransformer(tok.vocab_size, **{k:q['config'][k] for k in KEYS})
    m.load_state_dict(q['model']); m.eval()
    return m, tok, q
