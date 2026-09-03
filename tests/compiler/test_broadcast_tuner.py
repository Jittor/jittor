# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
from _helpers.logs import find_log_with_re
from _helpers.tuner_parser import simple_parser

class TestBroadcastTuner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        return

    def check(self, h, w, cs, rs, pa, rtp, dim):
        a = jt.random([h,w])
        a.data
        

        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100",
            # this value is used for force compile
            compile_options={"test_broadcast_tuner":1}
        ) as logs:
            amean=jt.mean(a, dims=[dim], keepdims=1)
            a2mean=jt.mean(a*a, dims=[dim], keepdims=1)
            norm_aa=(a-amean.broadcast_var(a))/(jt.sqrt(a2mean-amean*amean).broadcast_var(a))
            norm_aa.data
        logs = find_log_with_re(logs, 
            "Run tuner broadcast: confidence\\((20)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1, logs
        assert logs[0][0] == "20", "confidence of reorder should be 20"
        candidates = simple_parser(logs[0][1])
        assert candidates == {"order0": [0,], "order1": [1,], "order2": [0,], "split1": [2048,], "use_movnt": [1,],}, candidates
        
    def test_broadcast_tuner(self):
        self.check(8192,8192, 0, 0, 0, 5, 0)

    def test_use_movnt_is_emitted_as_a_compiler_intrinsic(self):
        values = np.arange(4096, dtype=np.float32)
        lhs = jt.array(values)
        rhs = jt.ones((32, 4096), dtype="float32")
        with jt.log_capture_scope(
                log_v=0,
                log_vprefix="jit_compiler.cc=1000",
                compile_options={"test_movnt_intrinsic": 1}) as logs:
            result = (rhs-lhs.broadcast_var(rhs)).numpy()

        np.testing.assert_array_equal(result[7], 1-values)
        generated = [
            entry["msg"] for entry in logs
            if "#define op2_OP subtract" in entry["msg"]
            and "op2_zp[op2_i]" in entry["msg"]
        ]
        self.assertEqual(len(generated), 1, generated)
        self.assertNotIn("//@begin", generated[0])
        self.assertNotIn("//@end", generated[0])
        if jt.flags.cc_type == "clang":
            self.assertIn("__builtin_nontemporal_store", generated[0])
        else:
            self.assertIn("op2_zp[op2_i] =", generated[0])

if __name__ == "__main__":
    unittest.main()
