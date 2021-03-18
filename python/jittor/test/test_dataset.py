# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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
if os.uname()[1] == "jittor-ce":
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


class TestDataset2(unittest.TestCase):
    def test_dataset_use_jittor(self):
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

if __name__ == "__main__":
    unittest.main()
