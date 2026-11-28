"""Tests for CUDA row-wise scatter_add fast path."""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestScatterAddCuda(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def test_1d_and_2d_broadcast_scatter_add(self):
        rng = np.random.RandomState(7)
        for idx_dtype in ("int32", "int64"):
            base_1d = rng.randn(32).astype("float32")
            idx_1d = rng.randint(0, 32, size=(512,)).astype(idx_dtype)
            src_1d = rng.randn(512).astype("float32")
            want_1d = base_1d.copy()
            np.add.at(want_1d, idx_1d, src_1d)
            with jt.no_grad():
                got_1d = jt.array(base_1d).scatter_add(0, jt.array(idx_1d), jt.array(src_1d))
            np.testing.assert_allclose(got_1d.numpy(), want_1d, rtol=1e-5, atol=1e-5)

            base_2d = rng.randn(64, 128).astype("float32")
            idx_rows = rng.randint(0, 64, size=(2048,)).astype(idx_dtype)
            src_2d = rng.randn(2048, 128).astype("float32")
            want_2d = base_2d.copy()
            np.add.at(want_2d, idx_rows, src_2d)
            with jt.no_grad():
                idx_exp = jt.array(idx_rows).unsqueeze(-1).expand(2048, 128)
                got_2d = jt.array(base_2d).scatter_add(0, idx_exp, jt.array(src_2d))
            np.testing.assert_allclose(got_2d.numpy(), want_2d, rtol=1e-5, atol=1e-4)

    def test_step_capture_cuda_graph_replay(self):
        from jittor._runtime import step_capture

        rng = np.random.RandomState(11)
        idx_rows = rng.randint(0, 128, size=(4096,)).astype("int64")
        src_2d = rng.randn(4096, 64).astype("float32")
        want = np.zeros((128, 64), dtype="float32")
        np.add.at(want, idx_rows, src_2d)

        def step(idx, src):
            out = jt.zeros((128, 64), dtype="float32")
            return out.scatter_add_(0, idx.unsqueeze(-1).expand(4096, 64), src)

        cap = step_capture.StepCapture(step)
        with jt.no_grad():
            j_idx = jt.array(idx_rows)
            j_src = jt.array(src_2d)
            for _ in range(5):
                got = cap(j_idx, j_src)
            self.assertGreaterEqual(cap.stats["graph"], 1)
            np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
