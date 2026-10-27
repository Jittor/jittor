
from _helpers import capability as _test_capability
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

class TestReduceTuner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        return

    def check(self, h, w, cs, rs, pa, rtp, dim):
        a = jt.random([h,w])
        a.data

        with jt.log_capture_scope(
            log_v=0, log_vprefix="tuner_manager=100",
            # this value is used for force compile
            compile_options={"test_reduce_tuner":1}
        ) as logs:
            amean=jt.mean(a, dims=[dim], keepdims=1)
            a2mean=jt.mean(a*a, dims=[dim], keepdims=1)
            norm_aa=(a-amean.broadcast_var(a))/(jt.sqrt(a2mean-amean*amean).broadcast_var(a))
            norm_aa.data
        logs = find_log_with_re(logs, 
            "Run tuner reduce: confidence\\((20)\\) candidates\\((.*)\\)$")
        assert len(logs) == 1 , logs
        assert logs[0][0] == "20", "confidence of reorder should be 20"
        candidates = simple_parser(logs[0][1])
        assert candidates == {"order0": [0,], "order1": [1,], "order2": [0,], "split1": [2048,], }

    def test_reduce_tuner(self):
        self.check(8192,8192, 0, 0, 0, 5, 0)


class TestReduceTunerCuda(unittest.TestCase):
    """Why ReduceTuner still offers a CUDA reduction nothing, and that it could.

    ``ReduceTuner::run`` returns before adding any candidate when the fused op
    is a CUDA op. That is a policy choice about the candidate set -- a CPU L1
    tile size and loop reorderings measured slower on a 4090 -- not a workaround
    for a compile error any more: ``split{i}`` and ``parallel`` compile
    together, as the second test pins.
    """

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
    def test_declines_on_cuda(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.random([1024, 1024])
            a.sync()
            with jt.log_capture_scope(
                log_v=0, log_vprefix="tuner_manager=100",
                compile_options={"test_reduce_tuner_cuda": 1},
            ) as logs:
                jt.reduce(a, "add", (0,)).sync()
        found = find_log_with_re(
            logs, "Run tuner reduce: confidence\\((\\d+)\\) candidates\\((.*)\\)$")
        assert len(found) >= 1, logs
        for confidence, candidates in found:
            self.assertEqual(confidence, "0", found)
            # the empty candidate map prints as "{}"
            self.assertEqual(candidates, "{}", found)

    def test_a_split_candidate_compiles_under_parallel(self):
        """A split-off loop is sized from its bound, so CUDA can compile it.

        SplitLoopPass gives the split-off loop the range
        ``::min(range{i}-id{i}, stride{i})``, defined inside the outer loop and
        varying with it. ParallelPass sizes the thread grid at the call site, so
        it uses that range's upper bound, ``stride{i}``, which is visible there;
        a thread handed an id past the live range fails the unchanged
        ``id < range`` loop condition. That keeps the tail of a range that is
        not a multiple of the stride exact.

        CUDA always runs ParallelPass (``parallel`` is forced on), so the CUDA
        rows are exactly the case the reduce tuner used to be unable to offer.
        """
        cases = (
            # shape, reduction axes, split option, split size
            ([64, 384, 8, 8], (0, 2, 3), "split1", 64),   # exact multiple
            ([64, 100, 8, 8], (0, 2, 3), "split1", 64),   # 36-element tail
            ([100, 384, 8, 8], (0, 2, 3), "split0", 64),  # a reduction axis
        )
        for use_cuda in (0, 1):
            if use_cuda and not _test_capability.check_accelerator(
                    'cuda', backend=jt).enabled:
                continue
            for index, (shape, dims, split, n) in enumerate(cases):
                # A distinct value per row forces a fresh compilation, so a
                # cached kernel from another backend cannot stand in.
                key = 100 + index + 10 * use_cuda
                with self.subTest(use_cuda=use_cuda, shape=shape, split=split), \
                        jt.flag_scope(use_cuda=use_cuda, enable_tuner=0):
                    a = jt.random(shape)
                    a.sync()
                    expected = a.numpy().astype("float64").sum(axis=dims)
                    with jt.flag_scope(compile_options={
                            "parallel": 1, split: n, "test_rt_split": key}):
                        got = jt.reduce(a, "add", dims).numpy()
                    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
