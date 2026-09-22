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
from _helpers import capability as _test_capability
from _helpers.logs import find_log_with_re
from _helpers.tuner_parser import simple_parser

class TestMatmulTuner(unittest.TestCase):
    def test_matmul_tuner(self):
        # The relay needs somewhere to relay *to*: `find_op_capability` looks
        # for a registered matmul implementation, and on this backend only the
        # CPU library registers one (`backends/cpu/libraries/mkl/
        # mkl_capabilities.cc`, compiled in only with `use_mkl=1`). Without it
        # the tuner declines correctly -- `if (!make_matmul) continue` -- so
        # this case would be asserting that the build has a library it does not
        # have. Measured 2026-09-22 on the CPU gate: every capability query
        # (`matmul`, `conv2d`, `random`, `transpose`) returns `[]` there.
        # KI-TUNER-001 still holds for builds that *do* have one: the relay
        # cannot carry an operand that lives outside the fused op, which is what
        # the expand became when it turned into a stride-0 view.
        _test_capability.require_library("mkl")
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
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            n, m, k = 16, 8, 16
            a = jt.random([n, m])
            b = jt.random([m, k])
            reference = np.matmul(a.numpy(), b.numpy())
            try:
                _test_policy_stack.enter_context(jt.runtime.scope(auto_mixed_precision_level=4))
                c = a.broadcast([n, m, k], [2]) * b.broadcast([n, m, k], [0])
                c = c.sum(1)
                assert c.dtype == "float16", c.dtype
                got = c.numpy()
            finally:
                _test_policy_stack.enter_context(jt.runtime.scope(auto_mixed_precision_level=0))
            assert np.isfinite(got).all()
            assert (np.abs(got - reference) < 3e-2).all(), np.abs(got - reference).max()

    def test_mixed_precision_linear_trains(self):
        # The same defect reached every model with a linear layer: the forward
        # aborted before producing a value.
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            from jittor import nn
            jt.set_global_seed(3)
            x = jt.random([16, 8])
            target = jt.random([16, 4])
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            optimizer = nn.SGD(model.parameters(), lr=1e-2)
            losses = []
            try:
                _test_policy_stack.enter_context(jt.runtime.scope(auto_mixed_precision_level=4))
                assert model(x).dtype == "float16"
                for _ in range(3):
                    loss = ((model(x) - target) ** 2).mean()
                    losses.append(float(loss.numpy().reshape(-1)[0]))
                    optimizer.step(loss)
            finally:
                _test_policy_stack.enter_context(jt.runtime.scope(auto_mixed_precision_level=0))
            assert np.isfinite(losses).all(), losses
            assert losses[-1] < losses[0], losses


if __name__ == "__main__":
    unittest.main()
