# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import nn, Module
from jittor.models import resnet
import numpy as np
from .test_core import expect_error
import re
import random
import math
import unittest
from jittor.test.test_reorder_tuner import simple_parser
from jittor.test.test_log import find_log_with_re
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
import time
import os

class MnistNet(Module):
    def __init__(self):
        self.model = resnet.Resnet18()
        self.layer = nn.Linear(1000,10)
    def execute(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

def gen_data(shape):
    a = np.random.rand(*shape)
    return a.astype("float32")
    
def check_report(rep):
    def check_code(code):
        ress = re.findall( r'atomicAdd\((.*),(.*)\);', code)
        for res in ress:
            assert(code.find("float64 "+res[1]) != -1)
            assert(code.find("float32 "+res[1]) == -1)
            res_ = re.findall(r'&\((.*)\[(.*)\]\)', res[0])
            assert(len(res_) == 1)
            assert(res_[0][0][-1] == "p")
            name = res_[0][0][:-1]
            tp = name[:-1]+'T'+name[-1]
            res_ = re.findall(r'#define '+tp+' (.*)\n', code)
            assert(len(res_)==1)
            assert(res_[0]=='float64')
    for i in range(1,len(rep)):
        filename = rep[i][1]
        with open(filename, "r") as f:
            data = f.read()
        check_code(data)

class TestReduceF64Op(unittest.TestCase):
    def test_add(self):
        jt.flags.use_cuda = True
        x = gen_data((3,32,64,64))

        with jt.profile_scope() as report:
            x_jt = jt.array(x)
            print(x_jt.sum())
        check_report(report)

    def test_mean(self):
        jt.flags.use_cuda = True
        x = gen_data((3,32,64,64))

        with jt.profile_scope() as report:
            x_jt = jt.array(x)
            print(x_jt.mean())
        check_report(report)

    def test_bn(self):
        jt.flags.use_cuda = True
        x = gen_data((3,32,64,64))

        with jt.profile_scope() as report:
            x_jt = jt.array(x)
            bn = nn.BatchNorm(32)
            bn.train()
            y_jy = bn(x_jt)
            print(y_jy.sum())
        check_report(report)

    def setup_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        jt.seed(seed)
            
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_resnet(self):
        with jt.profile_scope() as report:
            self.setup_seed(1)

            # hyper-parameters
            self.batch_size = int(os.environ.get("TEST_BATCH_SIZE", "100"))
            self.weight_decay = 0.0001
            self.momentum = 0.9
            self.learning_rate = 0.1
            if jt.flags.amp_reg:
                self.learning_rate = 0.01
            # mnist dataset
            self.train_loader = MNIST(train=True, transform=trans.Resize(224)) \
                .set_attrs(batch_size=self.batch_size, shuffle=True)
            self.train_loader.num_workers = 4

            loss_list=[]
            acc_list=[]
            mnist_net = MnistNet()
            global prev
            prev = time.time()
            SGD = nn.SGD(mnist_net.parameters(), self.learning_rate, self.momentum, self.weight_decay)
            self.train_loader.endless = True

            for data, target in self.train_loader:
                batch_id = self.train_loader.batch_id
                epoch_id = self.train_loader.epoch_id
                data = data.float_auto()

                # train step
                # with jt.log_capture_scope(
                #     log_silent=1,
                #     log_v=1, log_vprefix="op.cc=100,exe=10",
                # ) as logs:
                output = mnist_net(data)
                loss = nn.cross_entropy_loss(output, target)
                SGD.step(loss)
                def callback(epoch_id, batch_id, loss, output, target):
                    # print train info
                    global prev
                    pred = np.argmax(output, axis=1)
                    acc = np.mean(target==pred)
                    loss_list.append(loss[0])
                    acc_list.append(acc)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f} \tTime:{:.3f}'
                        .format(epoch_id, batch_id, 600,1. * batch_id / 6.0, loss[0], acc, time.time()-prev))
                    # prev = time.time()
                # async version
                jt.fetch(epoch_id, batch_id, loss, output, target, callback)
                # sync version
                # callback(epoch_id, batch_id, loss.numpy(), output.numpy(), target.numpy())
                
                # log_conv = find_log_with_re(logs, 
                #     "Jit op key (not )?found: ((mkl)|(cudnn))_conv.*")
                # log_matmul = find_log_with_re(logs, 
                #     "Jit op key (not )?found: ((mkl)|(cublas))_matmul.*")
                # if batch_id > 2:
                #     assert len(log_conv)==59 and len(log_matmul)==6, (len(log_conv), len(log_matmul))

                mem_used = jt.flags.stat_allocator_total_alloc_byte \
                    -jt.flags.stat_allocator_total_free_byte
                # assert mem_used < 4e9, mem_used
                # TODO: why bigger?
                assert mem_used < 5.6e9, mem_used
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

                if jt.flags.amp_reg:
                    continue
                if jt.in_mpi:
                    assert jt.core.number_of_lived_vars() < 8100, jt.core.number_of_lived_vars()
                else:
                    assert jt.core.number_of_lived_vars() < 7000, jt.core.number_of_lived_vars()
                if self.train_loader.epoch_id >= 2:
                    break

            jt.sync_all(True)
            assert np.mean(loss_list[-50:])<0.5
            assert np.mean(acc_list[-50:])>0.8
        check_report(report)

if __name__ == "__main__":
    unittest.main()
