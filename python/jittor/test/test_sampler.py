import jittor as jt
from jittor.dataset import *
from PIL import Image
import numpy as np
import unittest



class TestSamplerDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=40, batch_size=1)

    def __getitem__(self, idx):
        return idx**2


class TestSampler(unittest.TestCase):
    def test_sequential_sampler(self):
        testdataset = TestSamplerDataset()
        seqsampler = SequentialSampler(testdataset)
        assert len(seqsampler) == 40
        for idx, batch in enumerate(seqsampler):
            assert idx == batch
        for i, data in enumerate(testdataset):
            assert data.item() == i**2

    def test_random_sampler(self):
        testdataset = TestSamplerDataset()
        randomsampler = RandomSampler(testdataset)
        assert len(randomsampler) == 40
        diff = 0
        for i, data in enumerate(testdataset):
            diff += data.item() == i**2
        assert diff < 10

    def test_subset_random_sampler(self):
        testdataset = TestSamplerDataset()
        subsetsampler = SubsetRandomSampler(testdataset, (20, 30))
        assert len(subsetsampler) == 10
        s = 0
        for i, data in enumerate(testdataset):
            s += data.item()
        s2 = 0 
        for i in range(20,30):
            s2 += i**2
        assert s == s2, (s, s2)

    def test_batch_sampler(self):
        testdataset = TestSamplerDataset()
        seqforbatch = SequentialSampler(testdataset)
        batchsampler = BatchSampler(seqforbatch, 4, drop_last=False)
        assert len(batchsampler) == 10
        for batch in batchsampler:
            assert len(batch) == 4


if __name__ == "__main__":
    unittest.main()
