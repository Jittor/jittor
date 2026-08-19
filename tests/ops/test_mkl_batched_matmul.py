"""The CPU batched-matmul relay, and the numbers it has to reproduce.

``MatmulTuner`` only recognises the two-dimensional broadcast/multiply/reduce
form, so before this relay existed every batched matmul on CPU ran as the
generic reindex kernel. That is correct but roughly a fortieth of oneDNN's
throughput, and both attention products of every transformer layer take that
path -- it dominated the CPU step of every transformer in the ecosystem gate.

These tests pin the two things that matter: the relay agrees with the generic
path it replaces (forward *and* both gradients), and it is the path actually
taken for a float32 batched matmul on CPU.
"""

import unittest

import numpy as np

import jittor as jt
from jittor import nn
import jittor.nn.functional.matrix as matrix


def _generic(function):
    """Run ``function`` with the relay disabled, i.e. on the reindex path."""
    saved = matrix._mkl_batched_matmul_is_available
    matrix._mkl_batched_matmul_is_available = lambda a, b: False
    try:
        return function()
    finally:
        matrix._mkl_batched_matmul_is_available = saved


def _matmul_with_grads(a_array, b_array):
    a = jt.array(a_array)
    b = jt.array(b_array)
    out = nn.matmul(a, b)
    grad_a, grad_b = jt.grad(out.sum(), [a, b])
    return out.numpy(), grad_a.numpy(), grad_b.numpy()


@unittest.skipIf(jt.compile_extern.mkl_ops is None, "Jittor was built without oneDNN")
class TestMklBatchedMatmul(unittest.TestCase):
    """oneDNN is a CPU backend, so every test here pins ``use_cuda`` off."""

    def setUp(self):
        self.random = np.random.RandomState(0)

    def _pair(self, a_shape, b_shape):
        return (
            self.random.randn(*a_shape).astype("float32"),
            self.random.randn(*b_shape).astype("float32"),
        )

    def _assert_matches_generic(self, a_shape, b_shape, tolerance=1e-4):
        a_array, b_array = self._pair(a_shape, b_shape)
        with jt.flag_scope(use_cuda=0):
            relayed = _matmul_with_grads(a_array, b_array)
            generic = _generic(lambda: _matmul_with_grads(a_array, b_array))
        for name, actual, expected in zip(
            ("forward", "grad a", "grad b"), relayed, generic
        ):
            scale = max(float(np.abs(expected).max()), 1.0)
            error = float(np.abs(actual - expected).max()) / scale
            self.assertLess(error, tolerance, "{} diverged: {:.3e}".format(name, error))

    def test_three_dimensional_batch(self):
        self._assert_matches_generic((4, 5, 6), (4, 6, 7))

    def test_attention_query_key(self):
        self._assert_matches_generic((2, 4, 64, 16), (2, 4, 16, 64))

    def test_attention_probabilities_times_values(self):
        self._assert_matches_generic((2, 4, 64, 64), (2, 4, 64, 16))

    def test_broadcast_batch_dimension(self):
        self._assert_matches_generic((1, 3, 4, 5), (2, 3, 5, 6))

    def test_bmm_helper_agrees(self):
        a_array, b_array = self._pair((3, 5, 6), (3, 6, 7))
        with jt.flag_scope(use_cuda=0):
            relayed = nn.bmm(jt.array(a_array), jt.array(b_array)).numpy()
        expected = np.matmul(a_array, b_array)
        self.assertLess(float(np.abs(relayed - expected).max()), 1e-4)

    def test_relay_is_the_path_taken_on_cpu(self):
        a_array, b_array = self._pair((2, 4, 8), (2, 8, 4))
        with jt.flag_scope(use_cuda=0):
            a, b = jt.array(a_array), jt.array(b_array)
            self.assertTrue(matrix._mkl_batched_matmul_is_available(a, b))
            with jt.log_capture_scope(log_v=0, log_vprefix="op.cc=100,exe=1000") as logs:
                nn.matmul(a, b).sync()
        names = " ".join(entry.get("msg", "") for entry in logs)
        self.assertIn("mkl_batched_matmul", names)

    def test_non_float32_keeps_the_generic_path(self):
        """The op is float32-only; everything else must not be routed to it."""
        a_array = self.random.randn(2, 3, 4).astype("float64")
        b_array = self.random.randn(2, 4, 3).astype("float64")
        with jt.flag_scope(use_cuda=0):
            # jt.array follows Jittor's float32 default, so ask for the wider
            # dtype explicitly rather than trusting the numpy array's.
            a = jt.array(a_array).float64()
            b = jt.array(b_array).float64()
            self.assertEqual(str(a.dtype), "float64")
            self.assertFalse(matrix._mkl_batched_matmul_is_available(a, b))
            result = nn.matmul(a, b).numpy()
        expected = np.matmul(a.numpy(), b.numpy())
        self.assertLess(float(np.abs(result - expected).max()), 1e-9)

    def test_cuda_is_left_to_cublas(self):
        if not jt.has_cuda:
            self.skipTest("CUDA is unavailable")
        with jt.flag_scope(use_cuda=1):
            a = jt.array(self.random.randn(2, 3, 4).astype("float32"))
            b = jt.array(self.random.randn(2, 4, 3).astype("float32"))
            self.assertFalse(matrix._mkl_batched_matmul_is_available(a, b))


if __name__ == "__main__":
    unittest.main()
