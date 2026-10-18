import random
from torch.utils.data import Sampler
class CurriculumSampler(Sampler):
 def __init__(self,dataset,stage=4,seed=42):
  self.indices=[]
  for i,row in enumerate(dataset.rows):
   meta=row[2] if len(row)>2 else {}; level=int(meta.get('difficulty',{}).get('level',1)) if isinstance(meta,dict) else 1
   if level<=stage:self.indices.append(i)
  self.seed=seed
 def __iter__(self):
  x=self.indices.copy();random.Random(self.seed).shuffle(x);return iter(x)
 def __len__(self):return len(self.indices)
