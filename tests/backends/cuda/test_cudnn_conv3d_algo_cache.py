# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""conv3d must not share an algorithm choice across dtypes (or with conv2d).

The three conv3d ops have no plan cache of their own: they look up the same
global ``fwd_algo_cache`` / ``bwdx_algo_cache`` / ``bwdw_algo_cache`` the 2-D
ops use, and their key used to carry only the input and filter extents, the
convolution parameters and the math type.  An fp32 and an fp16 convolution of
the same shape therefore hashed to the same entry and took one another's
algorithm, a 2-D key could collide with a 3-D one, and changing
``max_workspace_ratio`` at runtime invalidated nothing in 3D.

This is the interim fix -- a fuller one moves conv3d onto the backend-API plan
cache -- so the test asserts on the key itself.
"""
import unittest

import numpy as np

import jittor as jt

from _helpers.logs import find_log_with_re


_KEY_RE = r"(cudnn_conv3d \w+ algo cache key: .*)"


def _forward_keys(x, w):
    with jt.log_capture_scope(log_silent=1, log_v=0,
                              log_vprefix="cudnn_conv3d=100") as raw_log:
        jt.nn.conv3d(x, w).sync()
    return find_log_with_re(raw_log, _KEY_RE)


def _all_pass_keys(x, w):
    """Forward and both gradients, so all three caches are exercised."""
    with jt.log_capture_scope(log_silent=1, log_v=0,
                              log_vprefix="cudnn_conv3d=100") as raw_log:
        y = jt.nn.conv3d(x, w)
        gx, gw = jt.grad(y.float32().sum(), [x, w])
        jt.sync([y, gx, gw])
    return find_log_with_re(raw_log, _KEY_RE)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
@unittest.skipIf(not jt.cudnn, "No cuDNN found")
class TestCudnnConv3dAlgoCache(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.flags.use_tensorcore)
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda, jt.flags.use_tensorcore = self._saved

    def _inputs(self, dtype):
        x = jt.array(np.random.RandomState(0)
                     .randn(2, 3, 5, 6, 7).astype("float32")).cast(dtype)
        w = jt.array(np.random.RandomState(1)
                     .randn(4, 3, 3, 3, 3).astype("float32")).cast(dtype)
        return x, w

    def test_keys_are_namespaced_and_carry_the_dtypes(self):
        keys = _forward_keys(*self._inputs("float32"))
        self.assertTrue(keys, "no conv3d cache key logged")
        for key in keys:
            self.assertIn("conv3d.", key)
            self.assertIn("x=float32", key)
            self.assertIn("workspace_ratio=", key)

    def test_float32_and_float16_do_not_share_an_entry(self):
        fp32 = _all_pass_keys(*self._inputs("float32"))
        fp16 = _all_pass_keys(*self._inputs("float16"))
        self.assertTrue(fp32 and fp16)
        # Same shapes, different dtypes: every key must differ.
        self.assertFalse(set(fp32) & set(fp16),
                         "fp32 and fp16 conv3d share a cache key:\n%s\n%s"
                         % ("\n".join(sorted(set(fp32))),
                            "\n".join(sorted(set(fp16)))))
        for key in fp16:
            self.assertIn("float16", key)

    def test_all_three_passes_are_namespaced_separately(self):
        keys = _all_pass_keys(*self._inputs("float32"))
        tags = {key.split(" ")[1] for key in keys}
        self.assertEqual(tags, {"fwd", "bwdx", "bwdw"})
        for key in keys:
            self.assertIn("conv3d." + key.split(" ")[1] + ";", key)

    def test_values_match_the_cpu_reference_in_both_dtypes(self):
        x_np = np.random.RandomState(2).randn(2, 3, 5, 6, 7).astype("float32")
        w_np = np.random.RandomState(3).randn(4, 3, 3, 3, 3).astype("float32")
        with jt.flag_scope(use_cuda=0):
            want = jt.nn.conv3d(jt.array(x_np), jt.array(w_np)).numpy()
        for dtype, tol in (("float32", 1e-3), ("float16", 6e-2)):
            got = jt.nn.conv3d(jt.array(x_np).cast(dtype),
                               jt.array(w_np).cast(dtype)).float32().numpy()
            np.testing.assert_allclose(got, want, atol=tol,
                                       rtol=tol, err_msg=dtype)


if __name__ == "__main__":
    unittest.main()
