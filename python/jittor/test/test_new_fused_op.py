# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import sys
import os
import jittor as jt
import unittest
import time
import numpy as np
from .test_log import find_log_with_re

class TestNewFuse(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        return

    def check(self, h, w, cs, rs, pa, rtp, dim):
        a = jt.random([h,w])
        a.sync()

        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100",
            # this value is used for force compile
            compile_options={"test_new_fused_op":1}
        ) as logs:
            amean=jt.mean(a, dims=[dim], keepdims=1)
            a2mean=jt.mean(a*a, dims=[dim], keepdims=1)
            norm_aa=(a-amean.broadcast_var(a))/(jt.sqrt(a2mean-amean*amean).broadcast_var(a))
            norm_aa.sync()
        logs = find_log_with_re(logs, 
            "Run tuner reduce: confidence\\((.*)\\) candidates\\((.*)\\)$")
        assert len(logs) == 3, logs
        
    def test_new_fuse(self):
        self.check(8192,8192, 0, 0, 0, 5, 0)

if __name__ == "__main__":
    unittest.main()
