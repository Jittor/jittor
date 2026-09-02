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
    """Why ReduceTuner offers a CUDA reduction nothing.

    ``ReduceTuner::run`` returns before adding any candidate when the fused op is
    a CUDA op. These two tests pin the reason, so that the day the reason stops
    holding, one of them fails and points at the comment to revisit.
    """

    @unittest.skipIf(not jt.has_cuda, "No cuda found")
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

    def test_a_split_candidate_would_not_compile_under_parallel(self):
        """``split{i}`` and ``parallel`` cannot be combined -- today.

        SplitLoopPass gives the split-off loop the range
        ``::min(range{i}-id{i}, stride{i})``, defined inside the outer loop and
        varying with it. ParallelPass evaluates every range at the call site to
        size the thread grid, does not find the definition, and aborts. CUDA
        always runs ParallelPass, so offering ``split1`` -- the main candidate
        this tuner has -- would break every CUDA reduction. Shown here on CPU
        because it is not a CUDA property.

        If this stops raising, the blocker named in reduce_tuner.cc is gone and
        the CUDA guard there should be revisited.
        """
        a = jt.random([64, 384, 8, 8])
        a.sync()
        expected = a.numpy().sum(axis=(0, 2, 3))
        with jt.flag_scope(use_cuda=0):
            # parallel alone is fine
            with jt.flag_scope(compile_options={"parallel": 1, "test_rt_split": 1}):
                got = jt.reduce(a, "add", (0, 2, 3)).numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-4)
            # split alone is fine
            with jt.flag_scope(compile_options={"split1": 64, "test_rt_split": 2}):
                got = jt.reduce(a, "add", (0, 2, 3)).numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-4)
            # together they are not
            with self.assertRaises(RuntimeError):
                with jt.flag_scope(compile_options={
                        "parallel": 1, "split1": 64, "test_rt_split": 3}):
                    jt.reduce(a, "add", (0, 2, 3)).sync()


if __name__ == "__main__":
    unittest.main()
