"""Jittor-owned CUDA RNG state covers every native random distribution."""

from _helpers import capability as _test_capability
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

import unittest

import numpy as np

import jittor as jt


def _draw_mixed():
    return [
        jt.random((17,), "float32", "uniform").numpy(),
        jt.random((19,), "float32", "normal").numpy(),
        jt.random((21,), "float64", "uniform").numpy(),
        jt.random((23,), "float64", "normal").numpy(),
    ]


@_test_preserve_policy(jt, "use_cuda")
@unittest.skipIf(
    not _test_capability.check_accelerator("cuda", backend=jt).enabled,
    "No CUDA found",
)
class TestCudaRNGState(unittest.TestCase):
    def setUp(self):
        self._scope = jt.runtime.scope(use_cuda=1)
        self._scope.__enter__()

    def tearDown(self):
        jt.sync_all(True)
        self._scope.__exit__(None, None, None)

    def test_root_api_restores_mixed_odd_length_draws(self):
        jt.set_cuda_seed(0, 1729)
        jt.random((7,), "float32", "normal").sync()
        state = jt.get_cuda_rng_state(0)
        self.assertIsInstance(state, str)
        self.assertTrue(state.startswith("JITTOR_CUDA_PHILOX4X32_10_V1 "))
        expected = _draw_mixed()
        jt.set_cuda_rng_state(0, state)
        actual = _draw_mixed()
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

    def test_capture_and_restore_resolve_pending_random_work(self):
        jt.set_cuda_seed(0, 41)
        pending_before_capture = jt.random((9,), "float64", "normal")
        state = jt.get_cuda_rng_state(0)
        self.assertEqual(tuple(pending_before_capture.shape), (9,))
        expected = jt.random((11,), "float32", "uniform").numpy()
        jt.set_cuda_rng_state(0, state)
        pending_before_restore = jt.random((11,), "float32", "uniform")
        jt.set_cuda_rng_state(0, state)
        np.testing.assert_array_equal(pending_before_restore.numpy(), expected)
        np.testing.assert_array_equal(
            jt.random((11,), "float32", "uniform").numpy(), expected
        )

    def test_large_counter_restore_does_not_replay_history(self):
        sequences = []
        for offset in (0, 1 << 32, 1 << 60):
            state = "JITTOR_CUDA_PHILOX4X32_10_V1 2026 %s\n" % offset
            jt.set_cuda_rng_state(0, state)
            expected = _draw_mixed()
            jt.set_cuda_rng_state(0, state)
            actual = _draw_mixed()
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right)
            sequences.append(expected)
        for previous, current in zip(sequences, sequences[1:]):
            self.assertFalse(np.array_equal(previous[0], current[0]))
            self.assertFalse(np.array_equal(previous[3], current[3]))

    def test_malformed_and_legacy_states_are_atomic(self):
        jt.set_cuda_seed(0, 73)
        jt.random((5,), "float32", "normal").sync()
        state = jt.get_cuda_rng_state(0)
        expected = _draw_mixed()
        for invalid in (
            "",
            "JITTOR_CURAND_XORWOW_U32_V1 12020 73 0\n",
            "JITTOR_CUDA_PHILOX4X32_10_V1 73 nope\n",
            state + "trailing",
        ):
            jt.set_cuda_rng_state(0, state)
            with self.assertRaises((RuntimeError, ValueError)):
                jt.set_cuda_rng_state(0, invalid)
            actual = _draw_mixed()
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right)

    def test_uniform_range_and_normal_statistics(self):
        jt.set_cuda_seed(0, 991)
        for dtype in ("float32", "float64"):
            uniform = jt.random((65537,), dtype, "uniform").numpy()
            self.assertTrue(np.isfinite(uniform).all())
            self.assertTrue(((uniform > 0) & (uniform <= 1)).all())
            normal = jt.random((65537,), dtype, "normal").numpy()
            self.assertTrue(np.isfinite(normal).all())
            self.assertLess(abs(float(normal.mean())), 0.03)
            self.assertLess(abs(float(normal.var()) - 1.0), 0.05)

    @unittest.skipUnless(jt.get_device_count() >= 2, "requires two CUDA devices")
    def test_seed_before_first_draw_survives_later_device_initialization(self):
        previous = jt.current_device()
        try:
            jt.set_cuda_seed(0, 42)
            jt.get_cuda_initial_seed(1)
            self.assertEqual(jt.get_cuda_initial_seed(0), 42)
        finally:
            jt.set_device(previous)


if __name__ == "__main__":
    unittest.main()
