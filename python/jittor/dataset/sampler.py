# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Hao-Yang Peng
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from .dataset import Dataset
import numpy as np
from PIL import Image


class Sampler():
    def __init__(self, dataset):
        self.dataset = dataset
        # MUST set sampler here
        dataset.sampler = self

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset):
        # MUST set sampler here
        dataset.sampler = self
        self.dataset = dataset

    def __iter__(self):
        return iter(range(self.dataset.__real_len__()))

    def __len__(self):
        return self.dataset.__real_len__()


class RandomSampler(Sampler):
    def __init__(self, dataset, replacement=False, num_samples=None):
        # MUST set sampler here
        dataset.sampler = self
        self.dataset = dataset
        self.rep = replacement
        self._num_samples = num_samples

    @property
    def num_samples(self):
        if self._num_samples is None:
            return self.dataset.__real_len__()
        return self._num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = self.dataset.__real_len__()
        if self.rep:
            return iter(np.random.randint(low=0, high=n, size=(self.num_samples,), dtype=np.int64).tolist())
        return iter(np.random.permutation(n).tolist())


class SubsetRandomSampler(Sampler):
    def __init__(self, dataset, indice):
        '''
        testdataset = TestSamplerDataset()
        subsetsampler = SubsetRandomSampler(testdataset, (20, 30))

        for i, data in enumerate(testdataset):
            # data between 20 ~ 29
            ......
            
        '''
        # MUST set sampler here
        dataset.sampler = self
        self.dataset = dataset
        self.indices = indice
        assert indice[0] >= 0 and indice[1] < dataset.__real_len__() and indice[0] < indice[1]

    def __iter__(self):
        return (int(i) + self.indices[0] for i in np.random.permutation(self.indices[1] - self.indices[0]))

    def __len__(self):
        return self.indices[1] - self.indices[0]


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
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
