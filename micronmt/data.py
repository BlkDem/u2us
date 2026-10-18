import json, re
from pathlib import Path
import torch
from torch.utils.data import Dataset

class ParallelDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256, direction='ru2en'):
        self.rows=[]; self.tok=tokenizer; self.max_len=max_len
        seen=set()
        for line in Path(path).read_text(encoding='utf8').splitlines():
            try: r=json.loads(line); s,t=r['src'].strip(),r['tgt'].strip()
            except (ValueError, KeyError): continue
            key=(s,t)
            if not s or not t or key in seen or len(s)>4000 or len(t)>4000: continue
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', s+t): continue
            seen.add(key); self.rows.append((s,t))
        self.direction=direction
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def collate(batch, tokenizer, max_len=256, direction='ru2en'):
    src, tgt=zip(*batch); tag='<ru2en>' if direction=='ru2en' else '<en2ru>'
    a=[tokenizer.encode(tag+' '+x, max_len) for x in src]
    b=[tokenizer.encode(x, max_len) for x in tgt]
    return pad(a,tokenizer.pad_id), pad(b,tokenizer.pad_id)

def pad(xs, p):
    n=max(map(len,xs)); return torch.tensor([x+[p]*(n-len(x)) for x in xs])
