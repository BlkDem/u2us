import argparse
import json
import re
import unicodedata
from pathlib import Path


def keys(path):
    out = set()
    for number, line in enumerate(Path(path).read_text(encoding='utf8').splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{number}: invalid JSON') from exc
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('src', 'tgt')):
            raise ValueError(f'{path}:{number}: expected nonempty string src and tgt')
        for side in ('src', 'tgt'):
            text = unicodedata.normalize('NFKC', row[side])
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)
            out.add(re.sub(r'\s+', ' ', text).strip().casefold())
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ('train', 'valid', 'test'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    sets = {name: keys(getattr(a, name)) for name in ('train', 'valid', 'test')}
    result = {x + '_' + y: len(sets[x] & sets[y])
              for x, y in (('train', 'valid'), ('train', 'test'), ('valid', 'test'))}
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if any(result.values()) else 0)


if __name__ == '__main__':
    main()
