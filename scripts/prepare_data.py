import argparse, hashlib, json, random, re
from pathlib import Path

CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

def clean(text):
    return re.sub(r'\s+', ' ', CONTROL.sub(' ', text)).strip()

def good(src, tgt, max_chars, max_ratio):
    if not src or not tgt or len(src) > max_chars or len(tgt) > max_chars:
        return False
    if not re.search(r'[A-Za-zА-Яа-я]', src) or not re.search(r'[A-Za-zА-Яа-я]', tgt):
        return False
    a, b = max(len(src), 1), max(len(tgt), 1)
    return max(a / b, b / a) <= max_ratio

def read_rows(paths, max_chars, max_ratio):
    seen, rows = set(), []
    for path in paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            src, tgt = clean(str(row.get('src', ''))), clean(str(row.get('tgt', '')))
            if not good(src, tgt, max_chars, max_ratio): continue
            key = hashlib.sha1((src + '\0' + tgt).encode()).hexdigest()
            if key not in seen:
                seen.add(key); rows.append({'src': src, 'tgt': tgt})
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--valid-ratio', type=float, default=.02)
    p.add_argument('--test-ratio', type=float, default=.02)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-chars', type=int, default=2000)
    p.add_argument('--max-ratio', type=float, default=3.0)
    a = p.parse_args()
    rows = read_rows(a.input, a.max_chars, a.max_ratio)
    random.Random(a.seed).shuffle(rows)
    nv, nt = int(len(rows)*a.valid_ratio), int(len(rows)*a.test_ratio)
    splits = {'test': rows[:nt], 'valid': rows[nt:nt+nv], 'train': rows[nt+nv:]}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in part: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'seed': a.seed, 'counts': {k: len(v) for k,v in splits.items()}}, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(rows), 'counts': {k: len(v) for k,v in splits.items()}}, ensure_ascii=False))

if __name__ == '__main__': main()
