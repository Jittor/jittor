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
from jittor.models import vgg, resnet
import numpy as np
import sys, os
import random
import math
import unittest
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

class TestRandomOp(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test(self):
        jt.set_seed(3)
        with jt.log_capture_scope(
            log_silent=1,
            log_v=0, log_vprefix="op.cc=100"
        ) as raw_log:
            t = jt.random([5,5])
            t.data
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "curand_random" + ".*)")
        assert len(logs)==1

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_float64(self):
        jt.set_seed(3)
        with jt.log_capture_scope(
            log_silent=1,
            log_v=0, log_vprefix="op.cc=100"
        ) as raw_log:
            t = jt.random([5,5], dtype='float64')
            t.data
        logs = find_log_with_re(raw_log, "(Jit op key (not )?found: " + "curand_random" + ".*)")
        assert len(logs)==1

    def test_normal(self):
        from jittor import init
        n = 10000
        r = 0.155
        a = init.gauss([n], "float32", 1, 3)
        data = a.data

        assert (np.abs((data<(1-3)).mean() - r) < 0.1)
        assert (np.abs((data<(1)).mean() - 0.5) < 0.1)
        assert (np.abs((data<(1+3)).mean() - (1-r)) < 0.1)

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_normal_cuda(self):
        self.test_normal()


if __name__ == "__main__":
    unittest.main()
