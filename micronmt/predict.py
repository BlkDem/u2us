import argparse, json, torch
from pathlib import Path
from .checkpoint import load_checkpoint
from .beam import beam_search


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source', default='ru')
    p.add_argument('--max-len', type=int, default=256)
    p.add_argument('--beam-size', type=int, default=1)
    p.add_argument('--length-penalty', type=float, default=0.6)
    p.add_argument('--no-repeat-ngram-size', type=int, default=0)
    p.add_argument('--repetition-penalty', type=float, default=1.0)
    a = p.parse_args()
    if a.beam_size < 1:
        p.error('--beam-size must be >= 1')
    if a.length_penalty < 0:
        p.error('--length-penalty must be >= 0')
    if a.no_repeat_ngram_size < 0:
        p.error('--no-repeat-ngram-size must be >= 0')
    if a.repetition_penalty < 1.0:
        p.error('--repetition-penalty must be >= 1.0')
    m, tok, _ = load_checkpoint(a.checkpoint)
    tag = '<ru2en>' if a.source == 'ru' else '<en2ru>'
    out = []
    for line in Path(a.input).read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        ids = torch.tensor([tok.encode(tag + ' ' + row['src'], a.max_len)])
        if a.beam_size > 1:
            pred = beam_search(m, ids, beam_size=a.beam_size, max_len=a.max_len,
                               length_penalty=a.length_penalty,
                               no_repeat_ngram_size=a.no_repeat_ngram_size,
                               repetition_penalty=a.repetition_penalty)[0]
        else:
            pred = m.greedy_cached(ids, max_len=a.max_len)[0]
        out.append({'src': row['src'], 'tgt': tok.decode(pred)})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n', encoding='utf8')
    print(f'wrote {len(out)} predictions to {a.output}')


if __name__ == '__main__':
    main()

