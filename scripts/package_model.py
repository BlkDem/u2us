import argparse, hashlib, json, shutil, torch
from pathlib import Path
from micronmt.checkpoint import load_tokenizer_metadata, tokenizer_metadata, tokenizer_fingerprint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', help='Optional matching SentencePiece model')
    p.add_argument('--out', required=True)
    p.add_argument('--model-name', default='micronmt')
    a = p.parse_args()
    cp = Path(a.checkpoint)
    q = torch.load(cp, map_location='cpu', weights_only=True)
    tok = load_tokenizer_metadata(q.get('tokenizer', {}), cp)
    if a.tokenizer:
        from micronmt.sp_tokenizer import SentencePieceTokenizer
        supplied = SentencePieceTokenizer(a.tokenizer)
        if tokenizer_fingerprint(tok) != tokenizer_fingerprint(supplied):
            p.error('Supplied tokenizer does not match the checkpoint tokenizer')
    out = Path(a.out)
    if (out / 'model.pt').resolve() == cp.resolve():
        p.error('Output model must not overwrite the source checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    q['tokenizer'] = tokenizer_metadata(tok)
    torch.save(q, out / 'model.pt')
    files = ['model.pt']
    if q['tokenizer']['type'] == 'sentencepiece':
        (out / 'tokenizer.model').write_bytes(q['tokenizer']['model_proto'])
        files.append('tokenizer.model')
    info = {'model_name': a.model_name, 'checkpoint_sha256': sha(cp),
            'packaged_model_sha256': sha(out / 'model.pt'),
            'tokenizer_fingerprint': tokenizer_fingerprint(tok),
            'files': files, 'format': 'pytorch-checkpoint'}
    (out / 'metadata.json').write_text(json.dumps(info, indent=2), encoding='utf8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
