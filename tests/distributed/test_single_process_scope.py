# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.distributed import run_mpi_test

from jittor.dataset import Dataset


class SyntheticDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=128)

    def __getitem__(self, index):
        return np.array([index], dtype=np.float32), index


def val1():
    dataloader = SyntheticDataset().set_attrs(batch_size=16)
    for i, (imgs, labels) in enumerate(dataloader):
        assert(imgs.shape[0]==8)
        if i == 5:
            break


@jt.single_process_scope(rank=0)
def val2():
    dataloader = SyntheticDataset().set_attrs(batch_size=16)
    for i, (imgs, labels) in enumerate(dataloader):
        assert(imgs.shape[0]==16)
        if i == 5:
            break


@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestSingleProcessScope(unittest.TestCase):
    def test_single_process_scope(self):
        val1()
        val2()


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestSingleProcessScopeEntry(unittest.TestCase):
    def test_entry(self):
        run_mpi_test(2, "test_single_process_scope")


if __name__ == "__main__":
    unittest.main()
