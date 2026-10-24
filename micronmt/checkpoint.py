import hashlib
from pathlib import Path
import torch
from .model import MicroTransformer
from .tokenizer import SimpleTokenizer

try:
    from _codecs import encode as _codecs_encode
    torch.serialization.add_safe_globals([_codecs_encode])
except (ImportError, AttributeError):
    pass

KEYS=('d_model','nhead','encoder_layers','decoder_layers','dim_feedforward','dropout','max_len')


def load_tokenizer_metadata(meta, checkpoint_path=None):
    if meta.get('type') == 'sentencepiece':
        from .sp_tokenizer import SentencePieceTokenizer
        if 'model_proto' in meta:
            return SentencePieceTokenizer(model_proto=meta['model_proto'])
        model_file = Path(meta['model_file'])
        if not model_file.is_absolute() and checkpoint_path is not None:
            packaged = Path(checkpoint_path).resolve().parent / model_file
            if packaged.exists():
                model_file = packaged
            elif not model_file.exists():
                raise FileNotFoundError(f'SentencePiece model not found: {packaged}')
        return SentencePieceTokenizer(model_file)
    return SimpleTokenizer(meta.get('vocab', meta))


def tokenizer_metadata(tokenizer):
    if isinstance(tokenizer, SimpleTokenizer):
        return {'type': 'simple', 'vocab': dict(tokenizer.vocab)}
    return {'type': 'sentencepiece', 'model_proto': tokenizer.processor.serialized_model_proto()}


def tokenizer_fingerprint(tokenizer):
    if isinstance(tokenizer, SimpleTokenizer):
        import json
        content = json.dumps(tokenizer.vocab, sort_keys=True, ensure_ascii=False).encode('utf8')
    else:
        content = tokenizer.processor.serialized_model_proto()
    return hashlib.sha256(content).hexdigest()


def load_checkpoint(path):
    q=torch.load(path,map_location='cpu',weights_only=True)
    tok=load_tokenizer_metadata(q.get('tokenizer', {}), path)
    m=MicroTransformer(tok.vocab_size, **{k:q['config'][k] for k in KEYS},
                       init_mode=q['config'].get('init_mode', 'controlled'))
    m.load_state_dict(q['model']); m.eval()
    return m, tok, q
