"""Complex matrix products on CUDA run on cuBLAS, with torch's autograd.

They used to fall through the relay table to the generic
``broadcast * mul + reduce`` kernel, because the cuBLAS row required
``is_float()``: a complex64 2048-cube product took 60 ms against torch's
1.4 ms. The rows now accept one shared complex dtype (``cublasGemmEx`` with
``CUDA_C_32F``/``CUDA_C_64F``), so this file checks three things: the relay is
what actually runs, it computes what numpy computes for every layout the
callers hand it, and its gradient is torch's -- dA = G op(B)^H, dB = op(A)^H G
-- which the generic path also produces.
"""
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import override_kernel, select_kernel
from jittor.nn.functional import matrix

_COMPLEX = ("complex64", "complex128")
_TOL = {"complex64": dict(rtol=1e-4, atol=1e-4), "complex128": dict(rtol=1e-10, atol=1e-10)}


def _sample(rng, shape, dtype):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


def _generic():
    """Both cuBLAS rows switched off: the products take the generic path."""
    off = lambda *args: False
    return (override_kernel("matmul", "cuda", matrix._cublas_matmul, supports=off),
            override_kernel("batched_matmul", "cuda", matrix._cublas_batched_matmul,
                            supports=off))


@_test_preserve_policy(jt, 'cuda_allow_tf32')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 "CUDA is required")
class TestCublasComplexMatmul(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(20261015)
        self.scope = jt.flag_scope(use_cuda=1, auto_mixed_precision_level=0)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def _array(self, shape, dtype):
        raw = _sample(self.rng, shape, dtype)
        return raw, jt.array(raw, dtype=dtype)

    def test_the_cublas_rows_take_one_shared_complex_dtype(self):
        for dtype in _COMPLEX:
            with self.subTest(dtype=dtype):
                a = jt.array(_sample(self.rng, (2, 3, 4), dtype), dtype=dtype)
                b = jt.array(_sample(self.rng, (2, 4, 5), dtype), dtype=dtype)
                self.assertIs(select_kernel("matmul", a[0], b[0], False, False),
                              matrix._cublas_matmul)
                self.assertIs(select_kernel("batched_matmul", a, b, False, False),
                              matrix._cublas_batched_matmul)
                self.assertTrue(matrix._cublas_can_take(a[0], b[0]))

    def test_mixed_complex_pairs_stay_on_the_generic_path(self):
        """complex64 x float64 is as wide as complex64 x complex64; the relay
        must not be offered it, and the product must still compute."""
        raw_a, a = self._array((3, 4), "complex64")
        for other in ("float64", "complex128", "float32"):
            with self.subTest(other=other):
                raw_b = self.rng.standard_normal((4, 5)).astype(other)
                if other.startswith("complex"):
                    raw_b = _sample(self.rng, (4, 5), other)
                b = jt.array(raw_b, dtype=other)
                self.assertIsNot(select_kernel("matmul", a, b, False, False),
                                 matrix._cublas_matmul)
                got = jt.matmul(a, b).numpy()
                np.testing.assert_allclose(got, raw_a @ raw_b, rtol=1e-4, atol=1e-4)

    def test_the_relay_is_what_runs(self):
        for dtype in _COMPLEX:
            with self.subTest(dtype=dtype):
                _, a = self._array((64, 32), dtype)
                _, b = self._array((32, 48), dtype)
                _, ab = self._array((3, 64, 32), dtype)
                _, bb = self._array((3, 32, 48), dtype)
                jt.sync([a, b, ab, bb])
                with jt.flag_scope(profiler_hide_relay=0), jt.profile_scope() as report:
                    jt.matmul(a, b).sync()
                    jt.matmul(ab, bb).sync()
                names = " ".join(row[0] for row in report[1:])
                self.assertIn("cublas_matmul", names)
                self.assertIn("cublas_batched_matmul", names)

    def test_forward_matches_numpy_for_every_caller_layout(self):
        for dtype in _COMPLEX:
            tol = _TOL[dtype]
            raw_a, a = self._array((6, 5), dtype)
            raw_b, b = self._array((5, 7), dtype)
            raw_bt, bt = self._array((7, 5), dtype)
            raw_at, at = self._array((5, 6), dtype)
            raw_a3, a3 = self._array((2, 3, 6, 5), dtype)
            raw_ba, ba = self._array((4, 6, 5), dtype)
            raw_bb, bb = self._array((4, 5, 7), dtype)
            raw_bc, bc = self._array((1, 5, 7), dtype)
            raw_bbt, bbt = self._array((4, 7, 5), dtype)
            cases = {
                "2-D": (jt.matmul(a, b), raw_a @ raw_b),
                "trans_b view": (jt.matmul(a, bt.transpose()), raw_a @ raw_bt.T),
                "trans_a view": (jt.matmul(at.transpose(), b), raw_at.T @ raw_b),
                "both views": (jt.matmul(at.transpose(), bt.transpose()), raw_at.T @ raw_bt.T),
                "rank>2 @ 2-D": (jt.matmul(a3, b), raw_a3 @ raw_b),
                "batched": (jt.matmul(ba, bb), raw_ba @ raw_bb),
                "broadcast batch": (jt.matmul(ba, bc), raw_ba @ raw_bc),
                "bmm": (jt.nn.bmm(ba, bb), raw_ba @ raw_bb),
                "bmm_transpose": (jt.nn.bmm_transpose(ba, bbt),
                                  raw_ba @ np.swapaxes(raw_bbt, -1, -2)),
                "matmul_transpose": (jt.nn.matmul_transpose(a, bt), raw_a @ raw_bt.T),
                "tensordot": (jt.nn.tensordot(a3, b, dims=([3], [0])),
                              np.tensordot(raw_a3, raw_b, axes=([3], [0]))),
            }
            for name, (got, want) in cases.items():
                with self.subTest(dtype=dtype, case=name):
                    got = got.numpy()
                    self.assertEqual(got.dtype, np.dtype(dtype))
                    np.testing.assert_allclose(got, want, **tol)

    def _grads(self, a, b, g, ta, tb):
        lhs = a.transpose() if ta else a
        rhs = b.transpose() if tb else b
        loss = (jt.matmul(lhs, rhs) * g.conj()).real.sum()
        return [x.numpy() for x in jt.grad(loss, [a, b])]

    def test_gradients_are_torchs_and_agree_with_the_generic_path(self):
        """loss = Re sum(C * conj(G)) has torch gradient G for C, so
        dA = G op(B)^H and dB = op(A)^H G, in the operands' own layouts."""
        for dtype in _COMPLEX:
            tol = _TOL[dtype]
            for ta in (False, True):
                for tb in (False, True):
                    with self.subTest(dtype=dtype, trans_a=ta, trans_b=tb):
                        raw_a, a = self._array((5, 6) if ta else (6, 5), dtype)
                        raw_b, b = self._array((7, 5) if tb else (5, 7), dtype)
                        raw_g, g = self._array((6, 7), dtype)
                        op_a = raw_a.T if ta else raw_a
                        op_b = raw_b.T if tb else raw_b
                        want_a = raw_g @ op_b.conj().T
                        want_b = op_a.conj().T @ raw_g
                        if ta:
                            want_a = want_a.T
                        if tb:
                            want_b = want_b.T
                        ga, gb = self._grads(a, b, g, ta, tb)
                        self.assertEqual(ga.dtype, np.dtype(dtype))
                        self.assertEqual(gb.dtype, np.dtype(dtype))
                        np.testing.assert_allclose(ga, want_a, **tol)
                        np.testing.assert_allclose(gb, want_b, **tol)
                        first, second = _generic()
                        with first, second:
                            gen_a, gen_b = self._grads(a, b, g, ta, tb)
                        np.testing.assert_allclose(ga, gen_a, **tol)
                        np.testing.assert_allclose(gb, gen_b, **tol)

    def test_batched_gradients_are_torchs(self):
        for dtype in _COMPLEX:
            with self.subTest(dtype=dtype):
                tol = _TOL[dtype]
                raw_a, a = self._array((3, 6, 5), dtype)
                raw_b, b = self._array((3, 5, 7), dtype)
                raw_g, g = self._array((3, 6, 7), dtype)
                loss = (jt.matmul(a, b) * g.conj()).real.sum()
                ga, gb = (x.numpy() for x in jt.grad(loss, [a, b]))
                np.testing.assert_allclose(
                    ga, raw_g @ np.swapaxes(raw_b.conj(), -1, -2), **tol)
                np.testing.assert_allclose(
                    gb, np.swapaxes(raw_a.conj(), -1, -2) @ raw_g, **tol)


if __name__ == "__main__":
    unittest.main()
