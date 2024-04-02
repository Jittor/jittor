import jittor as jt
import jittor.dataset
from jittor.dataset import Dataset as JDataset

from collections import namedtuple
from typing import Any, Callable, Iterable, Optional, Sequence, Union


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

class IterableDataset:
    def __iter__(self):
        raise NotImplementedError


class DataLoader(JDataset):
    def __init__(self, dataset,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False, 
                 sampler = None,
                 batch_sampler = None,
                 num_workers: int = 0, 
                 collate_fn = None,
                 pin_memory: bool = False, 
                 drop_last: bool = False,
                 timeout: float = 0, 
                 worker_init_fn = None,
                 multiprocessing_context=None, 
                 generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 pin_memory_device: str = "") -> None:
        super().__init__(batch_size=batch_size, 
                         shuffle=shuffle,
                         num_workers=num_workers,
                         drop_last=drop_last)
        
        unsupported_kwargs = {
            "batch_sampler": batch_sampler, 
            "pin_memory": pin_memory, 
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context, 
            "generator": generator, 
            "persistent_workers": persistent_workers, 
            "pin_memory_device": pin_memory_device
        }
        for kwarg, value in unsupported_kwargs.items():
            if value:
                jt.LOG.w(f"Not implemented Dataloader kwarg: {kwarg}")

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sampler = sampler

        if not isinstance(dataset, IterableDataset):
            self.total_len = len(dataset)
        else:
            # TODO: support multiple worker for iterable dataset
            assert(num_workers == 0)

    def collate_batch(self, batch):
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        else:
            return super().collate_batch(batch)

    def __getitem__(self, i):
        return self.dataset[i]
    
    def __iter__(self):
        if isinstance(self.dataset, IterableDataset):
            return self.inner_iter()
        else:
            return super().__iter__()

    def inner_iter(self):
        current_batch = []

        if jt.world_size > 1:
            assert self.batch_size % jt.world_size == 0, \
                f"IterableDataset does not support a batch size ({self.batch_size}) that is not evenly divisible by the number of processes f{jt.world_size}"
            real_batch_size = int(self.batch_size / jt.world_size)
        else:
            real_batch_size = self.batch_size

        for element in self.dataset:
            current_batch.append(element)

            if len(current_batch) == real_batch_size:
                current_batch = self.collate_batch(current_batch)
                current_batch = self.to_jittor(current_batch)
                yield current_batch
                current_batch = []
        
        if not self.drop_last and len(current_batch) > 0:
            current_batch = self.collate_batch(current_batch)
            yield self.to_jittor(current_batch)

def get_worker_info():
    # always return the fake worker info
    return namedtuple('WorkerInfo', 'id num_workers')(0, 1)

class RandomSampler(jt.dataset.RandomSampler):
    def __init__(self, dataset, generator=None, **kwargs):
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        if getattr(self.dataset, "support_random_access", True):
            return super().__iter__()
        else:
            self.dataset.shuffle()
            return iter(range(self.dataset.__real_len__() if hasattr(self.dataset,"__real_len__") else self.dataset.__len__()))

class DistributedSampler(jt.dataset.Sampler):
    def __init__(self, sampler: RandomSampler):
        assert(isinstance(sampler, RandomSampler))
        self.sampler = sampler

    def set_epoch(self, epoch: int):
        ### do nothing, let jittor's inner dataset handle 
        pass

    def __iter__(self):
        return self.sampler.__iter__()
    
    def __len__(self):
        return self.sampler.__len__()

BatchSampler = jt.dataset.BatchSampler
Sampler = jt.dataset.Sampler
SequentialSampler = jt.dataset.SequentialSampler
SubsetRandomSampler = jt.dataset.SubsetRandomSampler

TensorDataset = Dataset
