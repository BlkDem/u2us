from pathlib import Path
import sentencepiece as spm

class SentencePieceTokenizer:
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def __init__(self, model_file=None, model_proto=None):
        self.processor = (spm.SentencePieceProcessor(model_proto=model_proto) if model_proto is not None
                          else spm.SentencePieceProcessor(model_file=str(model_file)))
        actual = (self.processor.pad_id(), self.processor.unk_id(), self.processor.bos_id(), self.processor.eos_id())
        if actual != (0, 1, 2, 3):
            raise ValueError('SentencePiece requires pad/unk/bos/eos IDs 0/1/2/3')
        for tag in ('<ru2en>', '<en2ru>'):
            if self.processor.piece_to_id(tag) == self.unk_id:
                raise ValueError(f'SentencePiece is missing direction token {tag}')

    @property
    def vocab_size(self):
        return self.processor.get_piece_size()

    def encode(self, text, max_len=256):
        ids = self.processor.encode(text, out_type=int)[:max_len - 2]
        return [self.bos_id, *ids, self.eos_id]

    def decode(self, ids):
        ids = [int(i) for i in ids if int(i) not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.processor.decode(ids)

def train_sentencepiece(input_files, model_prefix, vocab_size=16000, model_type='unigram'):
    spm.SentencePieceTrainer.train(
        input=','.join(map(str, input_files)), model_prefix=str(model_prefix),
        vocab_size=max(vocab_size, 128), model_type=model_type,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=['<ru2en>', '<en2ru>'],
        character_coverage=0.9995, hard_vocab_limit=False,
    )
