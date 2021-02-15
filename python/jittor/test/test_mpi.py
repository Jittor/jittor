# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import os, sys
import jittor as jt
import numpy as np
mpi = jt.compile_extern.mpi

@unittest.skipIf(not jt.in_mpi, "no inside mpirun")
class TestMpi(unittest.TestCase):
    def test_mpi_test_op(self):
        assert jt.compile_extern.mpi_ops.mpi_test("").data == 123

    @unittest.skipIf(jt.compile_extern.nccl_ops is None, "no nccl")
    @jt.flag_scope(use_cuda=1)
    def test_nccl_with_mpi(self):
        assert jt.compile_extern.nccl_ops.nccl_test("test_with_mpi").data == 123

    def test_mpi_broadcast(self):
        for i in range(mpi.world_size()):
            a = np.zeros(100) + mpi.world_rank()
            mpi.broadcast(a, i)
            assert (a == i).all()

    def test_mpi_dataset(self):
        from jittor.dataset.dataset import Dataset
        class ToyDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.set_attrs(batch_size=21, total_len=211)

            def __getitem__(self, index):
                return index, index*index
        
        toy = ToyDataset()
        offset = ((toy.batch_size-1) // mpi.world_size() + 1) * mpi.world_rank()

        for _ in range(2):
            for i,(a,b) in enumerate(toy):
                assert (a.data*a.data == b.data).all()
                if mpi.world_rank() == 0:
                    if i == len(toy)-1:
                        assert a.shape[0] == 1
                        c = np.array([210])
                    else:
                        assert toy.real_batch_size == 11
                        c = np.array(range(offset+i*toy.batch_size, offset+i*toy.batch_size + toy.real_batch_size))
                else:
                    if i == len(toy)-1:
                        assert a.shape[0] == 1
                        c = np.array([210])
                    else:
                        assert toy.real_batch_size == 10
                        c = np.array(range(offset+i*toy.batch_size, offset+i*toy.batch_size + toy.real_batch_size))

                assert (c==a.data).all(), (c, a.data)

def run_mpi_test(num_procs, name):
    if not jt.compile_extern.inside_mpi():
        mpirun_path = jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")
        cmd = f"{mpirun_path} -np {num_procs} {sys.executable} -m jittor.test.{name} -v"
        print("run cmd:", cmd)
        assert os.system(cmd)==0, "run cmd failed: "+cmd

@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiEntry(unittest.TestCase):
    def test_entry(self):
        run_mpi_test(2, "test_mpi")
        
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    def test_mpi_resnet_entry(self):
        run_mpi_test(2, "test_resnet")

if __name__ == "__main__":
    unittest.main()