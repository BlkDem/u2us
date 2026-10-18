import json
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .terminology import control_hint

class Translator:
    def __init__(self, checkpoint, device='cpu', max_len=None, terminology=None):
        self.device=torch.device(device)
        self.model,self.tokenizer,self.metadata=load_checkpoint(checkpoint)
        self.model.to(self.device).eval()
        self.max_len=max_len or self.metadata['config'].get('max_len',256)
        self.terminology=terminology or {}

    @torch.inference_mode()
    def translate(self, text, source='ru', target=None, context=None, max_len=None):
        if source not in ('ru','en'): raise ValueError('source must be ru or en')
        target=target or ('en' if source=='ru' else 'ru')
        if target not in ('ru','en') or target==source: raise ValueError('target must be the opposite language')
        prefix='<ru2en>' if source=='ru' else '<en2ru>'
        if context: text=' <context> '+context+' </context> '+text
        if self.terminology: text=control_hint(text,self.terminology,source,target)
        ids=torch.tensor([self.tokenizer.encode(prefix+' '+text,max_len or self.max_len)],device=self.device)
        out=self.model.greedy(ids,max_len=max_len or self.max_len)
        return {'translation':self.tokenizer.decode(out[0]),'source_language':source,'target_language':target}

def load_terminology(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
