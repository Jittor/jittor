# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
import unittest
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

class testNet(Module):
    def __init__(self, op):
        self.op = op
        return
        
    def execute(self, x):
        N,H,W,C = x.shape
        y1=x.reindex_reduce(self.op, [N,H], ["i0","i1",])
        y2=x.reindex_reduce(self.op, [H,W], ["i1","i2",])
        y1=y1.broadcast([N,H,W],[2])
        y2=y2.broadcast([N,H,W],[0])
        return y1+y2

class TestAtomicTunerClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.addNet = testNet("add")
        self.maxNet = testNet("maximum")
        self.minNet = testNet("minimum")
        return
    
    def check(self, model, std_log):
        x=jt.random([100,64,128,128])
        with jt.log_capture_scope(
            # log_silent=1,
            log_v=0, log_vprefix="atomic=100,data=100",
        ) as logs:
            y=model(x).numpy()
        with jt.log_capture_scope(
            log_v=0, 
            exclude_pass="atomic",
            # new options to force recompile
            compile_options = {"test_atomic_tuner":1}
        ) as logs2:
            y_std=model(x).numpy()
        
        err=np.max(y_std-y)/(np.mean(y_std)+1e-6)
        assert err<1e-5, (err)
        log_move = find_log_with_re(logs, "atomictuner: move .* to loop .*")
        assert len(log_move)==len(std_log), (len(log_move), len(std_log))
        assert sorted(log_move) == sorted(std_log)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_atomic_tuner(self):
        self.check(self.addNet, ['atomictuner: move atomicAdd to loop 1', 'atomictuner: move atomicAdd to loop 2'])
        self.check(self.maxNet, ['atomictuner: move cuda_atomic_max to loop 1', 'atomictuner: move cuda_atomic_max to loop 2'])
        self.check(self.minNet, ['atomictuner: move cuda_atomic_min to loop 1', 'atomictuner: move cuda_atomic_min to loop 2'])

        self.check(lambda x: x.sum()+x.sqr().mean(), [
            'atomictuner: move atomicAdd to loop -1',
            'atomictuner: move atomicAdd to loop -1',
        ])

        self.check(lambda x: x.reindex_reduce("add", x.shape, ["i2","i3","i0","i1"]), [])
        
if __name__ == "__main__":
    unittest.main()
