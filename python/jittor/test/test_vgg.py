# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import vgg
import numpy as np
import sys, os
import random
import math
import unittest
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re
from jittor.dataset.mnist import MNIST
import jittor.transform as trans

model_test = os.environ.get("model_test", "") == "1"
skip_model_test = not model_test

class MnistNet(Module):
    def __init__(self):
        self.model = vgg.vgg16_bn()
        self.layer = nn.Linear(1000,10)
    def execute(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(skip_model_test, "skip_this_test, model_test != 1")
class TestVGGClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # hyper-parameters
        self.batch_size = 32
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.01
        # mnist dataset
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)) \
            .set_attrs(batch_size=self.batch_size, shuffle=True)

    # setup random seed
    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_vgg(self):
        self.setup_seed(1)
        loss_list=[]
        acc_list=[]
        mnist_net = MnistNet()
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)

            # train step
            with jt.log_capture_scope(
                log_silent=1,
                log_v=1, log_vprefix="op.cc=100,exe=10",
            ) as logs:
                SGD.step(loss)
                def callback(loss, output, target, batch_idx):
                    # print train info
                    pred = np.argmax(output, axis=1)
                    acc = np.sum(target==pred)/self.batch_size
                    loss_list.append(loss[0])
                    acc_list.append(acc)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'
                        .format(0, batch_idx, 100,1. * batch_idx, loss[0], acc))
                jt.fetch(batch_idx, loss, output, target, callback)

            log_conv = find_log_with_re(logs, 
                "Jit op key (not )?found: ((mkl)|(cudnn))_conv.*")
            log_matmul = find_log_with_re(logs, 
                "Jit op key (not )?found: ((mkl)|(cublas))_matmul.*")
            if batch_idx:
                assert len(log_conv)==38 and len(log_matmul)==12, (len(log_conv), len(log_matmul))

            mem_used = jt.flags.stat_allocator_total_alloc_byte \
                -jt.flags.stat_allocator_total_free_byte
            assert mem_used < 11e9, mem_used
            assert jt.core.number_of_lived_vars() < 3500
            if (np.mean(loss_list[-50:])<0.2):
                break

        assert np.mean(loss_list[-50:])<0.2

if __name__ == "__main__":
    unittest.main()
