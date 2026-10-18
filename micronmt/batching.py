import random
from torch.utils.data import Sampler

class LengthBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=512, seed=42):
        self.batch_size=batch_size; self.seed=seed
        lengths=[len(row[0].split())+len(row[1].split()) for row in dataset.rows]
        order=sorted(range(len(lengths)),key=lambda i:lengths[i])
        rng=random.Random(seed); buckets=[order[i:i+bucket_size] for i in range(0,len(order),bucket_size)]
        for b in buckets:rng.shuffle(b)
        rng.shuffle(buckets); self.batches=[b[i:i+batch_size] for b in buckets for i in range(0,len(b),batch_size)]
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)
