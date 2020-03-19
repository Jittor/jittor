# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
import jittor as jt
import unittest
import time
import numpy as np
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

class TestMatmulTuner(unittest.TestCase):
    def test_matmul_tuner(self):
        n,m,k = 10,10,10
        a = jt.random([n,m])
        b = jt.random([m,k])
        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100,var_relay=100",
            compile_options={"test_matmul_tuner":1}
        ) as rawlogs:
            c = a.broadcast([n,m,k], [2]) * b.broadcast([n,m,k], [0])
            c = c.sum(1)
            jc = c.numpy()
            nc = np.matmul(a.numpy(), b.numpy())
            assert (np.abs(jc-nc)<1e-3).all()
        logs = find_log_with_re(rawlogs, 
            "Run tuner matmul: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1
        assert logs[0][0] == "20", "confidence of reorder should be 20"
        candidates = simple_parser(logs[0][1])
        assert candidates == {"relay0":[1,0]}, candidates
        logs = find_log_with_re(rawlogs, r"get_relay_src([\s\S]*)")
        assert len(logs)==1
        assert "@relay_op" in logs[0]


if __name__ == "__main__":
    unittest.main()
