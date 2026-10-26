"""Complex matrix products run on a GEMM library, with torch's autograd.

They used to fall through the relay table to the generic
``broadcast * mul + reduce`` kernel: on CUDA because the cuBLAS row required
``is_float()`` (a complex64 2048-cube product took 60 ms against torch's
1.4 ms), on CPU because no row took complex at all (6.1 s against torch's
56 ms). CUDA now has ``cublasGemmEx`` with ``CUDA_C_32F``/``CUDA_C_64F``, CPU
a CBLAS ``cgemm``/``zgemm`` (numpy's bundled OpenBLAS by default). For each
device this file checks three things: the relay is what actually runs, it
computes what numpy computes for every layout the callers hand it, and its
gradient is torch's -- dA = G op(B)^H, dB = op(A)^H G -- which the generic path
also produces.
"""
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
import os
import unittest
from unittest import mock

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import override_kernel, select_kernel
from jittor.nn.functional import matrix

_COMPLEX = ("complex64", "complex128")
_TOL = {"complex64": dict(rtol=1e-4, atol=1e-4), "complex128": dict(rtol=1e-10, atol=1e-10)}


def _sample(rng, shape, dtype):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)


class _ComplexRelays:

    use_cuda = 0
    backend = "cpu"
    matmul_kernel = staticmethod(matrix._blas_matmul)
    batched_kernel = staticmethod(matrix._blas_batched_matmul)
    matmul_name, batched_name = "blas_matmul", "blas_batched_matmul"

    def _generic(self):
        """Both complex rows switched off: the products take the generic path."""
        off = lambda *args: False
        return (override_kernel("matmul", self.backend, self.matmul_kernel, supports=off),
                override_kernel("batched_matmul", self.backend, self.batched_kernel,
                                supports=off))

    def setUp(self):
        self.rng = np.random.default_rng(20261015)
        self.scope = jt.flag_scope(use_cuda=self.use_cuda, auto_mixed_precision_level=0)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def _array(self, shape, dtype):
        raw = _sample(self.rng, shape, dtype)
        return raw, jt.array(raw, dtype=dtype)

    def test_the_rows_take_one_shared_complex_dtype(self):
        for dtype in _COMPLEX:
            with self.subTest(dtype=dtype):
                a = jt.array(_sample(self.rng, (2, 3, 4), dtype), dtype=dtype)
                b = jt.array(_sample(self.rng, (2, 4, 5), dtype), dtype=dtype)
                self.assertIs(select_kernel("matmul", a[0], b[0], False, False),
                              self.matmul_kernel)
                self.assertIs(select_kernel("batched_matmul", a, b, False, False),
                              self.batched_kernel)

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
                                 self.matmul_kernel)
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
                self.assertIn(self.matmul_name, names)
                self.assertIn(self.batched_name, names)

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

    def test_products_large_enough_to_be_split_across_threads(self):
        """The CPU relay splits a product into row or column panels, or a
        batch across threads, once it is past a small-work threshold; odd
        extents leave a short last panel; tiny products bypass the library.
        Every route must give numpy's answer, for every transpose combination."""
        cases = [((37, 301), (301, 129)),   # columns are the longer side
                 ((257, 65), (65, 33)),     # rows are the longer side
                 ((3, 70, 50), (3, 50, 90)),  # fewer products than threads
                 ((40, 33, 31), (40, 31, 35)),  # a product per thread
                 # an inner dimension this short, or matrices this small,
                 # take the direct loop rather than the library
                 ((5, 8), (8, 3001)),
                 ((301, 4), (4, 7)),
                 ((500, 3, 4), (500, 4, 6))]
        for dtype in _COMPLEX:
            tol = _TOL[dtype]
            for shape_a, shape_b in cases:
                for ta in (False, True):
                    for tb in (False, True):
                        with self.subTest(dtype=dtype, a=shape_a, b=shape_b, ta=ta, tb=tb):
                            raw_a = _sample(self.rng, shape_a, dtype)
                            raw_b = _sample(self.rng, shape_b, dtype)
                            # store the transposed layout and hand the view in
                            sa = np.ascontiguousarray(np.swapaxes(raw_a, -1, -2)) if ta else raw_a
                            sb = np.ascontiguousarray(np.swapaxes(raw_b, -1, -2)) if tb else raw_b
                            a, b = jt.array(sa, dtype=dtype), jt.array(sb, dtype=dtype)
                            if ta:
                                a = a.transpose(*range(a.ndim - 2), a.ndim - 1, a.ndim - 2)
                            if tb:
                                b = b.transpose(*range(b.ndim - 2), b.ndim - 1, b.ndim - 2)
                            np.testing.assert_allclose(jt.matmul(a, b).numpy(), raw_a @ raw_b,
                                                       **tol)

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
                        first, second = self._generic()
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


class TestFindBlas(unittest.TestCase):
    """Which CBLAS the CPU rows link, decided from the library's own symbols."""

    def test_default_is_numpys_bundled_openblas(self):
        from jittor.build import compile_extern
        with mock.patch.dict(os.environ):
            os.environ.pop("JT_BUILD_BLAS_LIB_PATH", None)
            found, _ = compile_extern.find_blas()
        self.assertIsNotNone(found)
        path, gemms, ilp64, threads = found
        for kind, gemm in zip("sdcz", gemms):
            self.assertTrue(gemm.endswith(f"cblas_{kind}gemm" + ("64_" if ilp64 else "")), found)
        # numpy's OpenBLAS has the thread controls the relays need to run it
        # on Jittor's OpenMP team instead of a second pool.
        self.assertIsNotNone(threads, found)

    def test_an_explicit_library_without_cgemm_is_declined_with_a_reason(self):
        from jittor.build import compile_extern
        # A shared library that certainly loads and neither exports nor
        # depends on a CBLAS (dependencies are searched too, which is what lets
        # a wrapper such as conda's libcblas.so stand in for its backend).
        import _ctypes
        not_blas = _ctypes.__file__
        with mock.patch.dict(os.environ, {"JT_BUILD_BLAS_LIB_PATH": not_blas}):
            found, reasons = compile_extern.find_blas()
        self.assertIsNone(found)
        self.assertEqual(len(reasons), 1)
        self.assertIn("exports no cblas_sgemm/dgemm/cgemm/zgemm", reasons[0])


class TestComplexRelaysCPU(_ComplexRelays, unittest.TestCase):
    """CBLAS on CPU. Requires the library, which numpy's wheel provides."""

    # Spelled out here, not only inherited: they are the reference routes
    # `tests/structure/test_backend_grad_contract.py` names for BlasMatmulOp
    # and BlasBatchedMatmulOp, and it finds a test by reading this class.
    def test_gradients_are_torchs_and_agree_with_the_generic_path(self):
        super().test_gradients_are_torchs_and_agree_with_the_generic_path()

    def test_batched_gradients_are_torchs(self):
        super().test_batched_gradients_are_torchs()


class TestRealBlasRelaysCPU(unittest.TestCase):
    """float32/float64 on the same CBLAS. float64 had no CPU row at all, and
    float32 only oneDNN's, which does not build without cmake: both fell to
    the generic kernel, a float32 2048-cube product at 5.5 s against torch's
    13 ms."""

    _TOL = {"float32": dict(rtol=1e-4, atol=1e-4), "float64": dict(rtol=1e-10, atol=1e-10)}

    def setUp(self):
        self.rng = np.random.default_rng(20261026)
        self.scope = jt.flag_scope(use_cuda=0, auto_mixed_precision_level=0)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def _no_onednn(self):
        """The CPU rows as a machine without oneDNN has them: CBLAS alone."""
        return (override_kernel("matmul", "cpu", matrix._blas_matmul,
                                dtypes=matrix._BLAS_DTYPES, supports=matrix._supports_blas),
                override_kernel("batched_matmul", "cpu", matrix._blas_batched_matmul,
                                dtypes=matrix._BLAS_DTYPES,
                                supports=matrix._supports_blas_batched))

    def test_float64_takes_the_blas_row_and_float32_does_without_onednn(self):
        a = jt.array(self.rng.standard_normal((4, 5)), dtype="float64")
        b = jt.array(self.rng.standard_normal((5, 6)), dtype="float64")
        self.assertIs(select_kernel("matmul", a, b, False, False), matrix._blas_matmul)
        a, b = a.float32(), b.float32()
        first, second = self._no_onednn()
        with first, second:
            self.assertIs(select_kernel("matmul", a, b, False, False), matrix._blas_matmul)
            self.assertIs(select_kernel("batched_matmul", a.unsqueeze(0), b.unsqueeze(0),
                                        False, False), matrix._blas_batched_matmul)

    def test_forward_and_gradients_match_numpy(self):
        cases = [((6, 5), (5, 7)), ((37, 301), (301, 129)), ((5, 8), (8, 3001)),
                 ((3, 70, 50), (3, 50, 90)), ((500, 3, 4), (500, 4, 6))]
        first, second = self._no_onednn()
        with first, second:
            for dtype in ("float32", "float64"):
                tol = self._TOL[dtype]
                for shape_a, shape_b in cases:
                    for ta in (False, True):
                        for tb in (False, True):
                            with self.subTest(dtype=dtype, a=shape_a, b=shape_b, ta=ta, tb=tb):
                                raw_a = self.rng.standard_normal(shape_a).astype(dtype)
                                raw_b = self.rng.standard_normal(shape_b).astype(dtype)
                                sa = np.ascontiguousarray(np.swapaxes(raw_a, -1, -2)) if ta else raw_a
                                sb = np.ascontiguousarray(np.swapaxes(raw_b, -1, -2)) if tb else raw_b
                                a, b = jt.array(sa, dtype=dtype), jt.array(sb, dtype=dtype)
                                if ta:
                                    a = a.transpose(*range(a.ndim - 2), a.ndim - 1, a.ndim - 2)
                                if tb:
                                    b = b.transpose(*range(b.ndim - 2), b.ndim - 1, b.ndim - 2)
                                c = jt.matmul(a, b)
                                self.assertEqual(c.numpy().dtype, np.dtype(dtype))
                                np.testing.assert_allclose(c.numpy(), raw_a @ raw_b, **tol)
                                raw_g = self.rng.standard_normal(c.shape).astype(dtype)
                                ga, gb = jt.grad((c * jt.array(raw_g, dtype=dtype)).sum(), [a, b])
                                np.testing.assert_allclose(
                                    ga.numpy(), raw_g @ np.swapaxes(raw_b, -1, -2), **tol)
                                np.testing.assert_allclose(
                                    gb.numpy(), np.swapaxes(raw_a, -1, -2) @ raw_g, **tol)

    def test_the_relay_is_what_runs(self):
        a = jt.array(self.rng.standard_normal((64, 32)), dtype="float64")
        b = jt.array(self.rng.standard_normal((32, 48)), dtype="float64")
        jt.sync([a, b])
        with jt.flag_scope(profiler_hide_relay=0), jt.profile_scope() as report:
            jt.matmul(a, b).sync()
        self.assertIn("blas_matmul", " ".join(row[0] for row in report[1:]))


@_test_preserve_policy(jt, 'cuda_allow_tf32')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 "CUDA is required")
class TestComplexRelaysCUDA(_ComplexRelays, unittest.TestCase):
    use_cuda = 1
    backend = "cuda"
    matmul_kernel = staticmethod(matrix._cublas_matmul)
    batched_kernel = staticmethod(matrix._cublas_batched_matmul)
    matmul_name, batched_name = "cublas_matmul", "cublas_batched_matmul"

    def test_cublas_can_take_complex(self):
        _, a = self._array((3, 4), "complex64")
        self.assertTrue(matrix._cublas_can_take(a, a.transpose()))


if __name__ == "__main__":
    unittest.main()
