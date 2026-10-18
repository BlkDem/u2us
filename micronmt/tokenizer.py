from pathlib import Path
class SimpleTokenizer:
    """Deterministic smoke-test tokenizer; train SentencePiece for real experiments."""
    pad_id,bos_id,eos_id,unk_id=0,2,3,1
    def __init__(self, vocab=None):
        base={'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3,'<ru2en>':4,'<en2ru>':5}
        if vocab:
            # Checkpoints may contain either token->id or an iterable of tokens.
            if isinstance(vocab, dict):
                base.update({str(k): int(v) for k,v in vocab.items()})
            else:
                for x in vocab:
                    if x not in base: base[x]=len(base)
        self.vocab=base
        self.rev={v:k for k,v in self.vocab.items()}

    @classmethod
    def from_texts(cls, texts):
        tok=cls()
        for text in texts:
            for word in text.split():
                if word not in tok.vocab: tok.vocab[word]=len(tok.vocab)
        tok.rev={v:k for k,v in tok.vocab.items()}
        return tok
    def encode(self,text,max_len=256):
        ids=[self.vocab.get(x,self.unk_id) for x in text.split()][:max_len-2]
        return [self.bos_id]+ids+[self.eos_id]
    def decode(self,ids): return ' '.join(self.rev.get(int(i),'<unk>') for i in ids if int(i) not in (0,2,3))
    @property
    def vocab_size(self): return len(self.vocab)
