import jittor as jt
from jittor.dataset import *
from PIL import Image
import numpy as np
import unittest

test_img = np.random.normal(size=(40, 1, 2, 2))


class TestSamplerDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=40, batch_size=1)

    def __getitem__(self, idx):
        return test_img[idx:(idx + 1), ...]


testdataset = TestSamplerDataset()


class TestSampler(unittest.TestCase):
    def test_sequential_sampler(self):
        seqsampler = SequentialSampler(testdataset)
        assert len(seqsampler) == 40
        for idx, batch in enumerate(seqsampler):
            assert idx == batch

    def test_random_sampler(self):
        randomsampler = RandomSampler(testdataset)
        assert len(randomsampler) == 40

    def test_subset_random_sampler(self):
        subsetsampler = SubsetRandomSampler(testdataset, (20, 30))
        assert len(subsetsampler) == 10

    def test_batch_sampler(self):
        seqforbatch = SequentialSampler(testdataset)
        batchsampler = BatchSampler(seqforbatch, 4, drop_last=False)
        assert len(batchsampler) == 10
        for batch in batchsampler:
            assert len(batch) == 4


if __name__ == "__main__":
    unittest.main()
