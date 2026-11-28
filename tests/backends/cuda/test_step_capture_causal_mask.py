# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Guarded causal_mask_rewrite safety and replay tests inside StepCapture."""

from _helpers import capability as _test_capability
import gc
import unittest

import numpy as np

import jittor as jt
import jittor.nn.functional.attention as att


@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "no usable CUDA in this build",
)
class TestStepCaptureCausalMask(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self._cleanup_vars)
        self.addCleanup(self.scope.__exit__, None, None, None)

    def _cleanup_vars(self):
        gc.collect()
        jt.gc()

    def test_constant_causal_mask_rewritten_and_replayed(self):
        q = jt.randn((1, 4, 32, 64), dtype="bfloat16")
        k = jt.randn((1, 4, 32, 64), dtype="bfloat16")
        v = jt.randn((1, 4, 32, 64), dtype="bfloat16")
        jt.sync_all(True)

        rewrites = []
        orig = att.causal_mask_rewrite

        def spy(qq, kk, mm, ic):
            res = orig(qq, kk, mm, ic)
            rewrites.append(res)
            return res

        def step(q_in, k_in, v_in):
            m = jt.tril(jt.ones((32, 32), dtype="bool")).reshape((1, 1, 32, 32))
            return att.scaled_dot_product_attention(q_in, k_in, v_in, attn_mask=m)

        with jt.no_grad():
            eager = step(q, k, v).float32().numpy()
            att.causal_mask_rewrite = spy
            try:
                sc = jt.capture_step(step)
                out1 = sc(q, k, v)
                out2 = sc(q, k, v)
                out3 = sc(q, k, v)
                jt.sync_all(True)
            finally:
                att.causal_mask_rewrite = orig
        self.assertIsNone(sc._refused)
        self.assertGreaterEqual(sc.stats["captured"], 1)
        self.assertGreaterEqual(sc.stats["replayed"], 1)
        self.assertTrue(all(rewrites), f"expected causal rewrite during trace, got {rewrites}")
        np.testing.assert_allclose(out3.float32().numpy(), eager, rtol=2e-2, atol=2e-2)

    def test_pretrace_leaf_mutation_invalidates_capture(self):
        q = jt.randn((1, 4, 16, 64), dtype="bfloat16")
        k = jt.randn((1, 4, 16, 64), dtype="bfloat16")
        v = jt.randn((1, 4, 16, 64), dtype="bfloat16")
        keep = jt.ones((1, 16), dtype="int64")
        jt.sync_all(True)

        def step(q_in, k_in, v_in):
            causal = jt.tril(jt.ones((16, 16), dtype="bool")).reshape((1, 1, 16, 16))
            pad = (keep > 0).reshape((1, 1, 1, 16)).broadcast((1, 1, 16, 16))
            return att.scaled_dot_product_attention(q_in, k_in, v_in, attn_mask=(causal & pad))

        with jt.no_grad():
            sc = jt.capture_step(step)
            y1 = sc(q, k, v).float32().numpy()
            y2 = sc(q, k, v).float32().numpy()
            y3 = sc(q, k, v).float32().numpy()
            np.testing.assert_allclose(y3, y1, rtol=2e-2, atol=2e-2)
            # Mutate the pre-trace leaf in place: mask is no longer purely causal!
            new_keep = np.ones((1, 16), dtype=np.int64)
            new_keep[0, 0] = 0
            keep.assign(jt.array(new_keep))
            jt.sync_all(True)
            want_after = step(q, k, v).float32().numpy()
            got_after = sc(q, k, v).float32().numpy()
            np.testing.assert_allclose(got_after, want_after, rtol=2e-2, atol=2e-2)
            self.assertGreater(np.abs(want_after - y1).max(), 1e-2)

    def test_dynamic_mask_input_is_not_baked(self):
        q = jt.randn((1, 4, 16, 64), dtype="bfloat16")
        k = jt.randn((1, 4, 16, 64), dtype="bfloat16")
        v = jt.randn((1, 4, 16, 64), dtype="bfloat16")
        m_causal = jt.tril(jt.ones((16, 16), dtype="bool")).reshape((1, 1, 16, 16))
        m_full = jt.ones((1, 1, 16, 16), dtype="bool")
        jt.sync_all(True)

        def step(q_in, k_in, v_in, m_in):
            return att.scaled_dot_product_attention(q_in, k_in, v_in, attn_mask=m_in)

        with jt.no_grad():
            sc = jt.capture_step(step)
            sc(q, k, v, m_causal)
            sc(q, k, v, m_causal)
            got_full = sc(q, k, v, m_full).float32().numpy()
            want_full = step(q, k, v, m_full).float32().numpy()
            np.testing.assert_allclose(got_full, want_full, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
