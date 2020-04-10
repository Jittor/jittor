# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os, sys
import jittor as jt
import numpy as np
mpi = jt.compile_extern.mpi

@unittest.skipIf(mpi is None, "no inside mpirun")
class TestMpi(unittest.TestCase):
    def test_mpi_test_op(self):
        assert jt.compile_extern.mpi_ops.mpi_test("").data == 123

    @unittest.skipIf(jt.compile_extern.nccl_ops is None, "no inccl")
    @jt.flag_scope(use_cuda=1)
    def test_nccl_with_mpi(self):
        assert jt.compile_extern.nccl_ops.nccl_test("test_with_mpi").data == 123

    def test_mpi_broadcast(self):
        a = np.zeros(100) + mpi.world_rank()
        mpi.broadcast(a, 0)
        assert (a == 0).all()

    def test_mpi_dataset(self):
        from jittor.dataset.dataset import Dataset
        class ToyDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.set_attrs(total_len=1024)

            def __getitem__(self, index):
                return index, index*index
        
        toy = ToyDataset()
        offset = ((toy.total_len-1) // mpi.world_size() + 1) * mpi.world_rank()

        for _ in range(2):
            for i,(a,b) in enumerate(toy):
                assert (a.data*a.data == b.data).all()
                c = np.array(range(offset+i*toy.batch_size, offset+(i+1)*toy.batch_size))
                assert (c==a.data).all()

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiEntry(unittest.TestCase):
    def test_entry(self):
        if not jt.compile_extern.inside_mpi():
            mpirun_path = jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")
            cmd = f"{mpirun_path} -np 2 {sys.executable} -m jittor.test.test_mpi"
            print("run cmd:", cmd)
            assert os.system(cmd)==0, "run cmd failed: "+cmd

if __name__ == "__main__":
    unittest.main()