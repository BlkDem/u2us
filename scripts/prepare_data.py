import argparse, hashlib, json, math, random, re, unicodedata
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
        for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
            try: row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ('src', 'tgt')):
                raise ValueError(f'{path}:{number}: expected string src and tgt')
            src, tgt = clean(row['src']), clean(row['tgt'])
            if not good(src, tgt, max_chars, max_ratio): continue
            key = hashlib.sha1((src + '\0' + tgt).encode()).hexdigest()
            if key not in seen:
                seen.add(key); rows.append({**row, 'src': src, 'tgt': tgt})
    return rows

def leakage_key(text):
    return clean(unicodedata.normalize('NFKC', text)).casefold()


def split_rows(rows, valid_ratio, test_ratio, seed):
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (valid_ratio, test_ratio)) or valid_ratio + test_ratio >= 1:
        raise ValueError('split ratios must be finite, nonnegative and sum to less than one')
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for side in ('src', 'tgt'):
            key = leakage_key(row[side])
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    splits = {'test': [], 'valid': [], 'train': []}
    targets = {'test': int(len(rows) * test_ratio), 'valid': int(len(rows) * valid_ratio)}
    for group in groups:
        destination = next((s for s in targets if len(splits[s]) < targets[s]), 'train')
        splits[destination].extend(group)
    return splits


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
    if a.max_chars <= 0 or not math.isfinite(a.max_ratio) or a.max_ratio < 1:
        p.error('max-chars must be positive and max-ratio finite and >= 1')
    rows = read_rows(a.input, a.max_chars, a.max_ratio)
    splits = split_rows(rows, a.valid_ratio, a.test_ratio, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in part: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'seed': a.seed, 'counts': {k: len(v) for k,v in splits.items()}}, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(rows), 'counts': {k: len(v) for k,v in splits.items()}}, ensure_ascii=False))

if __name__ == '__main__': main()
