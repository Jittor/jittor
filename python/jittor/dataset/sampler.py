import jittor as jt
from .dataset import Dataset
import numpy as np
from PIL import Image

class Sampler():
    def __init__(self,data_source):
        self.data_source = data_source
    
    def __iter__(self):
        pass
    
    def __len__(self):
        pass

class SequentialSampler(Sampler):
    def __init__(self,data_source):
        self.data_source = data_source
    
    def __iter__(self):
        return iter(range(len(self.data_source)))
    
    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self,data_source,replacement=False, num_samples=None):
        self.data_source = data_source
        self.rep = replacement
        self._num_samples = num_samples
    
    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __iter__(self):
        n = len(self.data_source)
        if self.rep:
            return iter(np.random.randint(low=0, high=n, size=(self.num_samples,), dtype=np.int64).tolist()) 
        return iter(jt.randperm(n).numpy().tolist())


class SubsetRandomSampler(Sampler):
    def __init__(self,indices):
        self.indices = indices
    
    def __iter__(self):        
        return (self.indices[jt.to_int(i)] for i in jt.randperm((len(self.indices))))
    
    def __len__(self):
        return len(self.indices)


class BatchSampler(Sampler):
    def __init__(self,sampler,batch_size,drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
            



  