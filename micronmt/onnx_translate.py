import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--checkpoint', help='Legacy fallback for exports without tokenizer metadata')
    p.add_argument('--text', required=True)
    p.add_argument('--source', choices=['ru', 'en'], default='ru')
    p.add_argument('--sequence-len', type=int)
    a = p.parse_args()
    directory = Path(a.model_dir)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf8'))
    n = metadata['sequence_len']
    if a.sequence_len is not None and a.sequence_len != n:
        p.error(f'sequence-len must match exported length {n}')
    meta = metadata.get('tokenizer', {})
    if meta.get('type') == 'simple':
        from .tokenizer import SimpleTokenizer
        tok = SimpleTokenizer(meta['vocab'])
    elif meta.get('type') == 'sentencepiece' and (directory / meta.get('model_file', 'tokenizer.model')).exists():
        from .sp_tokenizer import SentencePieceTokenizer
        tok = SentencePieceTokenizer(directory / meta.get('model_file', 'tokenizer.model'))
    elif a.checkpoint:
        from .checkpoint import load_checkpoint
        _, tok, _ = load_checkpoint(a.checkpoint)
    else:
        p.error('Export has no packaged tokenizer; re-export or provide --checkpoint')
    tag = '<ru2en>' if a.source == 'ru' else '<en2ru>'
    ids = tok.encode(tag + ' ' + a.text, n)
    src = np.full((1, n), tok.pad_id, dtype=np.int64)
    src[0, :len(ids)] = ids
    enc = ort.InferenceSession(str(directory / 'encoder.onnx'), providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession(str(directory / 'decoder.onnx'), providers=['CPUExecutionProvider'])
    memory = enc.run(['memory'], {'src': src})[0]
    tgt = np.full((1, n), tok.pad_id, dtype=np.int64)
    tgt[0, 0] = tok.bos_id
    for i in range(1, n):
        logits = dec.run(['logits'], {'memory': memory, 'tgt': tgt, 'src': src})[0]
        nxt = int(logits[0, i - 1].argmax())
        tgt[0, i] = nxt
        if nxt == tok.eos_id:
            break
    print(tok.decode(tgt[0]))


if __name__ == '__main__':
    main()
