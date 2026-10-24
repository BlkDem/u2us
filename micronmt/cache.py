import torch


class KVCache:
    def __init__(self):
        self.k = []
        self.v = []

    def append(self, k, v):
        self.k.append(k)
        self.v.append(v)

    def get(self):
        if not self.k:
            return None, None
        return torch.cat(self.k, dim=2), torch.cat(self.v, dim=2)
