# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
from jittor.models import resnet
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
        self.model = resnet.Resnet18()
        self.layer = nn.Linear(1000,10)
    def execute(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

@unittest.skipIf(skip_this_test, "skip_this_test")
class TestMemoryProfiler(unittest.TestCase):
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
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1, trace_py_var=3, profile_memory_enable=1)
    def test_resnet(self):
        self.setup_seed(1)
        loss_list=[]
        acc_list=[]
        mnist_net = MnistNet()
        global prev
        prev = time.time()
        SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)

        iters = 10
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if (batch_idx > iters):
                break
            jt.display_memory_info()
            output = mnist_net(data)
            loss = nn.cross_entropy_loss(output, target)
            SGD.step(loss)
            def callback(batch_idx, loss, output, target):
                global prev
                pred = np.argmax(output, axis=1)
                acc = np.mean(target==pred)
                loss_list.append(loss[0])
                acc_list.append(acc)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'
                    .format(0, batch_idx, iters,1. * batch_idx / 6.0, loss[0], acc, time.time()-prev))
            jt.fetch(batch_idx, loss, output, target, callback)
        jt.sync_all(True)
        jt.display_max_memory_info()
        _, out = jt.get_max_memory_treemap()
        out_ = out.split('\n')
        assert(out_[0] == 'root()')
        assert(out_[3].endswith('(_run_module_as_main)'))
        assert(out_[7].endswith('(_run_code)'))
        _, out = jt.get_max_memory_treemap(build_by=1)
        out_ = out.split('\n')
        assert(out_[0] == 'root()')
        assert(out_[4].endswith('(_run_module_as_main)'))
        assert(out_[8].endswith('(_run_code)'))
        
    def test_sample(self):
        net = jt.models.resnet18()
        with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
            imgs = jt.randn((1,3,224,224))
            net(imgs).sync()
            jt.get_max_memory_treemap()
        


if __name__ == "__main__":
    unittest.main()
