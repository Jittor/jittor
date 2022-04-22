# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor.dataset.dataset import ImageFolder, Dataset
import jittor.transform as transform

import jittor as jt
import unittest
import os
import numpy as np
import random

pass_this_test = False
msg = ""
mid = 0
if hasattr(os, "uname") and os.uname()[1] == "jittor-ce":
    mid = 1
try:
    traindir = ["/data1/cjld/imagenet/train/","/home/cjld/imagenet/train/"][mid]
    assert os.path.isdir(traindir)
except Exception as e:
    pass_this_test = True
    msg = str(e)

@unittest.skipIf(pass_this_test, f"can not run imagenet dataset test: {msg}")
class TestDataset(unittest.TestCase):
    def test_multi_workers(self):
        check_num_batch = 10
        tc_data = []

        def get_dataset():
            dataset = ImageFolder(traindir).set_attrs(batch_size=256, shuffle=False)
            dataset.set_attrs(transform = transform.Compose([
                transform.Resize(224),
                transform.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ]), num_workers=0)
            return dataset

        dataset = get_dataset()

        for i, data in enumerate(dataset):
            print("get batch", i)
            tc_data.append(data)
            if i==check_num_batch: break

        def check(num_workers, epoch=1):
            dataset = get_dataset().set_attrs(num_workers=num_workers)

            random.seed(0)

            for _ in range(epoch):
                for i, (images, labels) in enumerate(dataset):
                    print("compare", i)
                    assert np.allclose(images.data, tc_data[i][0].data), \
                         (images.sum(), tc_data[i][0].sum(), images.shape, 
                         tc_data[i][0].shape)
                    assert np.allclose(labels.data, tc_data[i][1].data)
                    if i==check_num_batch: break
            # dataset.terminate()
        check(1)
        check(2)
        check(4,2)

    def test_collate_batch(self):
        from jittor.dataset.utils import collate_batch
        batch = collate_batch([(1,1),(1,2),(1,3)])
        assert isinstance(batch[0], np.ndarray)
        assert isinstance(batch[1], np.ndarray)


class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=10240)

    def __getitem__(self, k):
        self.tmp = None
        x = jt.array(k)
        y = x
        for i in range(10):
            for j in range(i+2):
                y = y + j - j
            y.stop_fuse()
        return x, y

        
class YourDataset2(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=16)

    def __getitem__(self, k):
        return np.random.rand(2)


class YourDataset3(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=16)

    def __getitem__(self, k):
        return random.randint(0,1000)


class YourDataset4(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        return jt.rand(2)


class YourDataset5(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        return { "a":np.array([1,2,3]) }

class TestDataset2(unittest.TestCase):
    def test_dataset_use_jittor(self):
        dataset = YourDataset().set_attrs(batch_size=256, shuffle=True, num_workers=4)
        dataset.tmp = jt.array([1,2,3,4,5])
        dataset.tmp.sync()
        for x, y in dataset:
            # dataset.display_worker_status()
            pass


    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_dataset_use_jittor_cuda(self):
        self.test_dataset_use_jittor()

class TestDatasetSeed(unittest.TestCase):
    def test_np(self):

        dataset = YourDataset2().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                dd.append(d.numpy())
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    assert not np.allclose(dd[i], dd[j])

    def test_py_native(self):
        import random

        jt.set_global_seed(0)
        dataset = YourDataset3().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                dd.append(d.numpy())
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    assert not np.allclose(dd[i], dd[j])

    def test_jtrand(self):
        import random

        jt.set_global_seed(0)
        dataset = YourDataset4().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                dd.append(d.numpy())
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    assert not np.allclose(dd[i], dd[j])

    def test_dict(self):
        import random

        jt.set_global_seed(0)
        dataset = YourDataset5().set_attrs(batch_size=1, shuffle=True, num_workers=4)
        for _ in range(10):
            dd = []
            for d in dataset:
                # breakpoint()
                assert isinstance(d, dict)
                assert isinstance(d['a'], jt.Var)
                np.testing.assert_allclose(d['a'].numpy(), [[1,2,3]])

    def test_cifar(self):
        from jittor.dataset.cifar import CIFAR10
        a = CIFAR10()
        a.set_attrs(batch_size=16)
        for imgs, labels in a:
            print(imgs.shape, labels.shape)
            assert imgs.shape == [16,32,32,3,]
            assert labels.shape == [16,]
            break

    def test_tensor_dataset(self):
        import jittor as jt
        from jittor.dataset import TensorDataset

        x = jt.array([1,2,3])
        y = jt.array([4,5,6])
        z = jt.array([7,8,9])

        dataset = TensorDataset(x, y, z)
        # dataset.set_attrs(batch_size=2)
        dataset.set_attrs(batch_size=1)

        for i,(a,b,c) in enumerate(dataset):
            # print(a,b,c)
            # print(a.shape)
            assert a.shape == [1]
            assert x[i] == a
            assert y[i] == b
            assert z[i] == c

    def test_children_died(self):
        if os.name == 'nt':
            # TODO: windows cannot pass this test now
            # don't know how to detect child died in windows
            # some clue: https://ikriv.com/blog/?p=1431
            return
        src = """
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        if k>100:
            while 1:
                pass
        return { "a":np.array([1,2,3]) }
if __name__ == "__main__":
    dataset = YourDataset()
    dataset.set_attrs(num_workers=2)

    for d in dataset:
        dataset.workers[0].p.kill()
        pass
"""
        fname = os.path.join(jt.flags.cache_path, "children_dead_test.py")
        with open(fname, 'w') as f:
            f.write(src)
        import subprocess as sp
        import sys
        cmd = sys.executable + " " + fname
        print(cmd)
        r = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        s = r.stderr.decode()
        print(s)
        assert r.returncode != 0
        assert "SIGCHLD" in s
        assert "quick exit" in s


    @unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
    def test_dataset_shuffle_mpi(self):
        src = """
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160, shuffle=True)

    def __getitem__(self, k):
        return k

dataset = YourDataset()
dataset.set_attrs(num_workers=2)

for d in dataset:
    for a in d:
        print("CHECK: ", a.item())
"""
        fname = os.path.join(jt.flags.cache_path, "test_dataset_shuffle_mpi.py")
        with open(fname, 'w') as f:
            f.write(src)
        import subprocess as sp
        import sys
        cmd = sys.executable + " " + fname
        mpirun_path = jt.compile_extern.mpicc_path.replace("mpicc", "mpirun")
        cmd = mpirun_path + " -np 2 " + cmd
        print(cmd)
        r = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        s = r.stdout.decode()
        # print(s)
        st = set([ l for l in s.splitlines() if l.startswith("CHECK:") ])
        assert r.returncode == 0
        # print(st)
        assert len(st) == 160, len(st)

    def test_children_died2(self):
        src = """
import jittor as jt
from jittor.dataset import Dataset
import numpy as np

class YourDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=160)

    def __getitem__(self, k):
        if k>100:
            while 1:
                pass
        return { "a":np.array([1,2,3]) }

if __name__ == "__main__":
    dataset = YourDataset()
    dataset.set_attrs(num_workers=2)

    for d in dataset:
        break
    dataset.terminate()
"""
        fname = os.path.join(jt.flags.cache_path, "children_dead_test.py")
        with open(fname, 'w') as f:
            f.write(src)
        import subprocess as sp
        import sys
        cmd = sys.executable + " " + fname
        print(cmd)
        r = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        s = r.stderr.decode()
        print(s)
        assert r.returncode == 0
        

if __name__ == "__main__":
    unittest.main()
