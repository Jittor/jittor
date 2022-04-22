# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import densenet
import numpy as np
import sys, os
import random
import math
import unittest
from jittor.test.test_reorder_tuner import simple_parser
from jittor.test.test_log import find_log_with_re
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
import time

skip_this_test = True

class MnistNet(Module):
    def __init__(self):
        self.model = densenet.densenet169()
        self.layer = nn.Linear(1000,10)
    def execute(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(skip_this_test, "skip_this_test")
class TestDensenet(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # hyper-parameters
        self.batch_size = 100
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.1
        # mnist dataset
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)) \
            .set_attrs(batch_size=self.batch_size, shuffle=True)
        self.train_loader.num_workers = 4

    # setup random seed
    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_densenet(self):
        self.setup_seed(1)
        loss_list=[]
        acc_list=[]
        mnist_net = MnistNet()
        global prev
        prev = time.time()
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)
        # SGD = jt.optim.Adam(mnist_net.parameters(), lr=0.0001)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)
            def callback(batch_idx, loss, output, target):
                # print train info
                global prev
                pred = np.argmax(output, axis=1)
                acc = np.mean(target==pred)
                loss_list.append(loss[0])
                acc_list.append(acc)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'
                    .format(0, batch_idx, 600,1. * batch_idx / 6.0, loss[0], acc, time.time()-prev))
                # prev = time.time()
            jt.fetch(batch_idx, loss, output, target, callback)
            # Train Epoch: 0 [0/600 (0%)]     Loss: 2.402650  Acc: 0.060000
            # Train Epoch: 0 [1/600 (0%)]     Loss: 2.770145  Acc: 0.100000
            # Train Epoch: 0 [2/600 (0%)]     Loss: 3.528072  Acc: 0.100000
            # Train Epoch: 0 [3/600 (0%)]     Loss: 2.992042  Acc: 0.100000
            # Train Epoch: 0 [4/600 (1%)]     Loss: 4.672772  Acc: 0.060000
            # Train Epoch: 0 [5/600 (1%)]     Loss: 5.003410  Acc: 0.080000
            # Train Epoch: 0 [6/600 (1%)]     Loss: 5.417546  Acc: 0.100000
            # Train Epoch: 0 [7/600 (1%)]     Loss: 5.137665  Acc: 0.100000
            # Train Epoch: 0 [8/600 (1%)]     Loss: 5.241075  Acc: 0.070000
            # Train Epoch: 0 [9/600 (2%)]     Loss: 4.515363  Acc: 0.100000
            # Train Epoch: 0 [10/600 (2%)]    Loss: 3.357187  Acc: 0.170000
            # Train Epoch: 0 [20/600 (3%)]    Loss: 2.265879  Acc: 0.100000
            # Train Epoch: 0 [30/600 (5%)]    Loss: 2.107000  Acc: 0.250000
            # Train Epoch: 0 [40/600 (7%)]    Loss: 1.918214  Acc: 0.290000
            # Train Epoch: 0 [50/600 (8%)]    Loss: 1.645694  Acc: 0.400000

        jt.sync_all(True)
        assert np.mean(loss_list[-50:])<0.3
        assert np.mean(acc_list[-50:])>0.9
        
if __name__ == "__main__":
    unittest.main()
