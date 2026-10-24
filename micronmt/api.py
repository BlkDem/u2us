import json
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .terminology import control_hint
from .beam import beam_search

class Translator:
    def __init__(self, checkpoint, device='cpu', max_len=None, terminology=None):
        self.device=torch.device(device)
        self.model,self.tokenizer,self.metadata=load_checkpoint(checkpoint)
        self.model.to(self.device).eval()
        self.max_len=max_len or self.metadata['config'].get('max_len',256)
        self.terminology=terminology or {}

    @torch.inference_mode()
    def translate(self, text, source='ru', target=None, context=None, max_len=None, beam_size=1):
        if source not in ('ru','en'): raise ValueError('source must be ru or en')
        target=target or ('en' if source=='ru' else 'ru')
        if target not in ('ru','en') or target==source: raise ValueError('target must be the opposite language')
        prefix='<ru2en>' if source=='ru' else '<en2ru>'
        if context: text=' <context> '+context+' </context> '+text
        if self.terminology: text=control_hint(text,self.terminology,source,target)
        ids=torch.tensor([self.tokenizer.encode(prefix+' '+text,max_len or self.max_len)],device=self.device)
        out=beam_search(self.model,ids,max_len=max_len or self.max_len,beam_size=beam_size) if beam_size > 1 else self.model.greedy_cached(ids,max_len=max_len or self.max_len)
        if out.dim()==1: out=out.unsqueeze(0)
        return {'translation':self.tokenizer.decode(out[0]),'source_language':source,'target_language':target}

def load_terminology(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
