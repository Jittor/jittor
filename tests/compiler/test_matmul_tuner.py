# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
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

    def test_relay_declines_when_output_dtype_differs(self):
        # The relay op takes its output dtype from its operands, so it can only
        # stand in for a reduce that produces that dtype. Auto mixed precision
        # level 4 retypes the reduce output to float16 and leaves the operands
        # float32; relaying there would allocate twice the bytes of the var it
        # replaces, and the size assertion in add_relay_group aborts the whole
        # fused operator. The tuner has to decline instead, leaving the fused
        # kernel to write the requested dtype.
        n, m, k = 16, 8, 16
        a = jt.random([n, m])
        b = jt.random([m, k])
        reference = np.matmul(a.numpy(), b.numpy())
        try:
            jt.flags.auto_mixed_precision_level = 4
            c = a.broadcast([n, m, k], [2]) * b.broadcast([n, m, k], [0])
            c = c.sum(1)
            assert c.dtype == "float16", c.dtype
            got = c.numpy()
        finally:
            jt.flags.auto_mixed_precision_level = 0
        assert np.isfinite(got).all()
        assert (np.abs(got - reference) < 3e-2).all(), np.abs(got - reference).max()

    def test_mixed_precision_linear_trains(self):
        # The same defect reached every model with a linear layer: the forward
        # aborted before producing a value.
        from jittor import nn
        jt.set_global_seed(3)
        x = jt.random([16, 8])
        target = jt.random([16, 4])
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        optimizer = nn.SGD(model.parameters(), lr=1e-2)
        losses = []
        try:
            jt.flags.auto_mixed_precision_level = 4
            assert model(x).dtype == "float16"
            for _ in range(3):
                loss = ((model(x) - target) ** 2).mean()
                losses.append(float(loss.numpy().reshape(-1)[0]))
                optimizer.step(loss)
        finally:
            jt.flags.auto_mixed_precision_level = 0
        assert np.isfinite(losses).all(), losses
        assert losses[-1] < losses[0], losses


if __name__ == "__main__":
    unittest.main()
