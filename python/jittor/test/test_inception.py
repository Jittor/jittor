# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import inception
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

skip_this_test = False

class MnistNet(Module):
    def __init__(self):
        self.model = inception.inception_v3()
        self.layer = nn.Linear(1000,10)
    def execute(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(skip_this_test, "skip_this_test")
class TestInception(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # hyper-parameters
        self.batch_size = 32
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.1
        # mnist dataset
        self.train_loader = MNIST(train=True, transform=trans.Resize(300)) \
            .set_attrs(batch_size=self.batch_size, shuffle=True)
        self.train_loader.num_workers = 4
        self.train_loader.total_len = self.batch_size * 300

    # setup random seed
    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_inception(self):
        self.setup_seed(1)
        loss_list=[]
        acc_list=[]
        mnist_net = MnistNet()
        global prev
        prev = time.time()
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)

        for batch_idx, (data, target) in enumerate(self.train_loader):

            # train step
            with jt.log_capture_scope(
                log_silent=1,
                log_v=1, log_vprefix="op.cc=100,exe=10",
            ) as logs:
                # breakpoint()
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
                        .format(0, batch_idx, 300,1. * batch_idx / 6.0, loss[0], acc, time.time()-prev))
                    # prev = time.time()
                jt.fetch(batch_idx, loss, output, target, callback)
            
            log_conv = find_log_with_re(logs, 
                "Jit op key (not )?found: ((mkl)|(cudnn))_conv.*")
            log_matmul = find_log_with_re(logs, 
                "Jit op key (not )?found: ((mkl)|(cublas))_matmul.*")
            if batch_idx > 2:
                assert len(log_conv)==283 and len(log_matmul)==6, (len(log_conv), len(log_matmul))

            mem_used = jt.flags.stat_allocator_total_alloc_byte \
                -jt.flags.stat_allocator_total_free_byte
            # assert mem_used < 4e9, mem_used
            # TODO: why bigger?
            assert mem_used < 15.6e9, mem_used
            # example log:
            # Train Epoch: 0 [0/100 (0%)]     Loss: 2.352903  Acc: 0.110000
            # Train Epoch: 0 [1/100 (1%)]     Loss: 2.840830  Acc: 0.080000
            # Train Epoch: 0 [2/100 (2%)]     Loss: 3.473594  Acc: 0.100000
            # Train Epoch: 0 [3/100 (3%)]     Loss: 3.131615  Acc: 0.200000
            # Train Epoch: 0 [4/100 (4%)]     Loss: 2.524094  Acc: 0.230000
            # Train Epoch: 0 [5/100 (5%)]     Loss: 7.780025  Acc: 0.080000
            # Train Epoch: 0 [6/100 (6%)]     Loss: 3.890721  Acc: 0.160000
            # Train Epoch: 0 [7/100 (7%)]     Loss: 6.370137  Acc: 0.140000
            # Train Epoch: 0 [8/100 (8%)]     Loss: 11.390827 Acc: 0.150000
            # Train Epoch: 0 [9/100 (9%)]     Loss: 21.598564 Acc: 0.080000
            # Train Epoch: 0 [10/100 (10%)]   Loss: 23.369165 Acc: 0.130000
            # Train Epoch: 0 [20/100 (20%)]   Loss: 4.804510  Acc: 0.100000
            # Train Epoch: 0 [30/100 (30%)]   Loss: 3.393924  Acc: 0.110000
            # Train Epoch: 0 [40/100 (40%)]   Loss: 2.286762  Acc: 0.130000
            # Train Epoch: 0 [50/100 (50%)]   Loss: 2.055014  Acc: 0.290000

            assert jt.core.number_of_lived_vars() < 50000, jt.core.number_of_lived_vars()

        jt.sync_all(True)
        assert np.mean(loss_list[-20:])<1
        assert np.mean(acc_list[-20:])>0.5
        
if __name__ == "__main__":
    unittest.main()
