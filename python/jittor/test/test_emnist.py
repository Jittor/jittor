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
from jittor.dataset.mnist import EMNIST, MNIST
import jittor.transform as transform

@unittest.skipIf(True, f"skip emnist test")
class TestEMNIST(unittest.TestCase):
    def test_emnist(self):
        import pylab as pl
        # emnist_dataset = EMNIST()
        emnist_dataset = EMNIST()
        for imgs, labels in emnist_dataset:
            print(imgs.shape, labels.shape)
            print(labels.max(), labels.min())
            # imgs = imgs.transpose(0,1,3,2).transpose(1,2,0,3)[0].reshape(28, -1)
            imgs = imgs.transpose(1,2,0,3)[0].reshape(28, -1)
            print(labels)
            pl.imshow(imgs), pl.show()
            break


if __name__ == "__main__":
    unittest.main()
