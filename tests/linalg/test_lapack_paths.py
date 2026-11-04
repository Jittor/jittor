"""The paths solve/inv/svd take into LAPACK and cuSOLVER.

On the host the ``numpy_code`` callbacks run numpy's OpenBLAS LAPACK with its
thread count sized to the problem (``_host_lapack``); on CUDA a 2-D solve or
inverse is one ``getrf``/``getrs`` pair with a single status read
(``_cuda_gesv``), a batch of solves one ``getrfBatched``/``getrsBatched`` pair
(``cublas_solve_batched``) or, for large matrices, getrf/getrs one matrix
after another (``cusolver_solve``), and a 2-D reduced SVD is Jacobi ``gesvdj`` with a ``gesvd``
fallback (``_cuda_svd_reduced``). QR's and Cholesky's backward solve against
a triangular factor with one ``trsm`` (``_solve_triangular``). Each must give numpy's answer for every
dtype and shape its caller hands it.
"""

import unittest

import numpy as np

import jittor as jt
from jittor.linalg import _helpers
from jittor.linalg.solving import _solve_triangular

from _helpers import capability as _test_capability

_HAS_CUDA = _test_capability.check_accelerator("cuda", backend=jt).enabled
_TOL = {"float32": 1e-4, "float64": 1e-10, "complex64": 1e-4, "complex128": 1e-10}


def _sample(rng, shape, dtype):
    x = rng.standard_normal(shape)
    if dtype.startswith("complex"):
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


class _Paths:
    use_cuda = 0

    def setUp(self):
        self.rng = np.random.default_rng(20261026)
        scope = jt.flag_scope(use_cuda=self.use_cuda)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_solve_and_inv_match_numpy(self):
        for dtype, tol in _TOL.items():
            a = _sample(self.rng, (7, 7), dtype) + 7 * np.eye(7, dtype=dtype)
            for rhs in ((7, 3), (7, 1), (7,)):
                with self.subTest(dtype=dtype, rhs=rhs):
                    b = _sample(self.rng, rhs, dtype)
                    got = jt.linalg.solve(jt.array(a, dtype=dtype), jt.array(b, dtype=dtype))
                    self.assertEqual(got.numpy().dtype, np.dtype(dtype))
                    np.testing.assert_allclose(got.numpy(), np.linalg.solve(a, b),
                                               rtol=tol, atol=tol)
            with self.subTest(dtype=dtype, op="inv"):
                np.testing.assert_allclose(jt.linalg.inv(jt.array(a, dtype=dtype)).numpy(),
                                           np.linalg.inv(a), rtol=tol, atol=tol)

    def test_batched_solve_matches_numpy(self):
        self._check_batched_solve()

    def _check_batched_solve(self):
        # Matrix and vector right-hand sides, broadcast batch dimensions on
        # either side, and an empty batch.
        cases = (((4, 7, 7), (4, 7, 3)), ((4, 7, 7), (4, 7, 1)), ((4, 7, 7), (4, 7)),
                 ((7, 7), (2, 3, 7, 2)), ((2, 1, 7, 7), (3, 7, 4)), ((0, 7, 7), (0, 7, 2)))
        for dtype, tol in _TOL.items():
            for a_shape, b_shape in cases:
                with self.subTest(dtype=dtype, a=a_shape, b=b_shape):
                    a = _sample(self.rng, a_shape, dtype) + 7 * np.eye(7, dtype=dtype)
                    b = _sample(self.rng, b_shape, dtype)
                    got = jt.linalg.solve(jt.array(a, dtype=dtype), jt.array(b, dtype=dtype))
                    # torch's vector rule, and numpy given equal ranks so that
                    # no numpy version reads a matrix batch as vectors
                    vector = b.shape == a.shape[:-1]
                    b_mat = b[..., None] if vector else b
                    batch = np.broadcast_shapes(a.shape[:-2], b_mat.shape[:-2])
                    expected = np.linalg.solve(np.broadcast_to(a, batch + a.shape[-2:]),
                                               np.broadcast_to(b_mat, batch + b_mat.shape[-2:]))
                    if vector:
                        expected = expected[..., 0]
                    self.assertEqual(got.numpy().dtype, np.dtype(dtype))
                    self.assertEqual(got.shape, list(expected.shape))
                    np.testing.assert_allclose(got.numpy(), expected, rtol=tol, atol=tol)

    def test_solve_with_mixed_operand_dtypes(self):
        a = self.rng.standard_normal((6, 6)) + 6 * np.eye(6)
        b = self.rng.standard_normal(6)
        got = jt.linalg.solve(jt.array(a, dtype="float32"), jt.array(b, dtype="float64"))
        np.testing.assert_allclose(got.numpy(), np.linalg.solve(a, b), rtol=1e-4, atol=1e-5)

    def test_reduced_svd_matches_numpy_and_differentiates(self):
        for dtype, tol in _TOL.items():
            for shape in ((9, 5), (5, 9), (6, 6)):
                with self.subTest(dtype=dtype, shape=shape):
                    a = _sample(self.rng, shape, dtype)
                    x = jt.array(a, dtype=dtype)
                    u, s, vh = jt.linalg.svd(x, full_matrices=False)
                    s = s.real if dtype.startswith("complex") else s
                    np.testing.assert_allclose(s.numpy(), np.linalg.svd(a, compute_uv=False),
                                               rtol=10 * tol, atol=10 * tol)
                    rebuilt = (u.numpy() * s.numpy()[None, :]) @ vh.numpy()
                    np.testing.assert_allclose(rebuilt, a, rtol=10 * tol, atol=10 * tol)
                    # d(sum s)/dA = U V^H, which does not depend on the phases
                    # either algorithm picks for the singular vectors
                    uu, _, vv = np.linalg.svd(a, full_matrices=False)
                    np.testing.assert_allclose(jt.grad(s.sum(), x).numpy(), uu @ vv,
                                               rtol=10 * tol, atol=10 * tol)

    def test_triangular_solve_is_a_general_solve_to_every_order(self):
        for dtype, tol in _TOL.items():
            for upper in (True, False):
                with self.subTest(dtype=dtype, upper=upper):
                    a = _sample(self.rng, (6, 6), dtype)
                    a = (np.triu(a) if upper else np.tril(a)) + 4 * np.eye(6, dtype=dtype)
                    b = _sample(self.rng, (6, 3), dtype)
                    results = []
                    for solve in (lambda p, q: _solve_triangular(p, q, upper), jt.linalg.solve):
                        pa, qb = jt.array(a, dtype=dtype), jt.array(b, dtype=dtype)
                        x = solve(pa, qb)
                        loss = (x * x.conj()).real.sum()
                        ga, gb = jt.grad(loss, [pa, qb])
                        gga = jt.grad((ga * ga.conj()).real.sum(), pa)
                        results.append([v.numpy() for v in (x, ga, gb, gga)])
                    np.testing.assert_allclose(results[0][0], np.linalg.solve(a, b),
                                               rtol=tol, atol=tol)
                    for got, want in zip(results[0], results[1]):
                        np.testing.assert_allclose(got, want, rtol=10 * tol, atol=10 * tol)
        singular = np.triu(np.ones((3, 3), "float32")) - np.diag([0, 1, 1]).astype("float32")
        with self.assertRaises(np.linalg.LinAlgError):
            _solve_triangular(jt.array(singular), jt.ones((3, 1)), True).sync()


class TestLapackPathsCPU(_Paths, unittest.TestCase):

    def test_small_factorizations_run_on_one_openblas_thread(self):
        threads = _helpers._openblas_threads()
        if not threads:
            self.skipTest("numpy does not run on an OpenBLAS with thread controls")
        setter, getter = threads
        before = getter()
        seen = []

        @_helpers._host_lapack
        def forward(array_module, data):
            seen.append(getter())
            data["outputs"][0][...] = array_module.linalg.inv(data["inputs"][0])

        a = jt.array(np.eye(8, dtype="float32") * 2)
        np.testing.assert_allclose(
            jt.numpy_code([a.shape], [a.dtype], [a], forward)[0].numpy(), np.eye(8) / 2)
        self.assertEqual(seen, [1])
        self.assertEqual(getter(), before)

    def test_thread_count_follows_the_work(self):
        threads_for = _helpers._lapack_threads_for
        self.assertEqual(threads_for([np.empty((128, 64))]), 1)
        self.assertEqual(threads_for([np.empty((4, 256, 256))]), _helpers._LAPACK_FEW_THREADS)
        self.assertEqual(threads_for([np.empty((1024, 1024))]), 0)


@unittest.skipUnless(_HAS_CUDA, "needs CUDA")
class TestLapackPathsCUDA(_Paths, unittest.TestCase):
    use_cuda = 1

    def test_two_dimensional_operands_take_the_native_ops(self):
        """solve/inv/qr run as cuSOLVER ops and QR's triangular solve as cuBLAS
        trsm, on Jittor's stream, with no numpy_code callback in between."""
        from jittor._runtime.dispatch import select_kernel
        if _helpers._cusolver_ops() is None:
            self.skipTest("this CUDA toolchain has no cuSOLVER")
        a = jt.array(np.eye(4, dtype="complex64") * 2)
        b = jt.ones((4, 2), dtype="complex64")
        self.assertIs(select_kernel("linalg_solve", a, b), _helpers._cuda_solve)
        self.assertIs(select_kernel("linalg_inv", a), _helpers._cuda_inv)
        self.assertIs(select_kernel("linalg_qr", a), _helpers._cuda_qr)
        self.assertIs(select_kernel("linalg_solve_triangular", a, b, True),
                      _helpers._cuda_solve_triangular)
        # a batch keeps the callbacks
        self.assertIsNone(select_kernel("linalg_qr", a.unsqueeze(0)))
        x = jt.array(_sample(self.rng, (6, 4), "complex64"))
        jt.sync([a, b, x])
        with jt.flag_scope(profiler_hide_relay=0), jt.profile_scope() as report:
            jt.linalg.solve(a, b).sync()
            jt.linalg.inv(a).sync()
            q, r = jt.linalg.qr(x)
            jt.grad((q.real.sum() + r.real.sum()), x).sync()
        names = " ".join(row[0] for row in report[1:])
        for op in ("cusolver_solve", "cusolver_inv", "cusolver_qr", "cublas_trsm"):
            self.assertIn(op, names)
        self.assertNotIn("numpy_code", names)

    def test_a_batched_solve_takes_the_native_op(self):
        """A batch of solves is one cuBLAS getrfBatched/getrsBatched op, in the
        forward and in both solves of the backward, with no numpy_code
        callback; a broadcast operand is made dense for it."""
        from jittor._runtime.dispatch import select_kernel
        if getattr(_helpers._cublas_ops(), "cublas_solve_batched", None) is None:
            self.skipTest("this CUDA toolchain has no cuBLAS")
        a = jt.array(_sample(self.rng, (5, 5), "complex64") + 5 * np.eye(5, dtype="complex64"))
        b = jt.array(_sample(self.rng, (3, 5, 2), "complex64"))
        jt.sync([a, b])
        if _helpers._cusolver_ops() is not None:
            self.assertIs(select_kernel("linalg_solve", a, b[0]), _helpers._cuda_solve)
        self.assertIs(select_kernel("linalg_solve", a.broadcast((3, 5, 5)), b),
                      _helpers._cuda_solve_batched)
        with jt.flag_scope(profiler_hide_relay=0), jt.profile_scope() as report:
            x = jt.linalg.solve(a, b)
            ga, gb = jt.grad((x * x.conj()).real.sum(), [a, b])
            jt.sync([x, ga, gb])
        names = [row[0] for row in report[1:]]
        self.assertFalse(any("numpy_code" in name for name in names), names)
        # The forward and the adjoint solve; the backward reuses x rather
        # than solving for it again.
        calls = sum(int(row[2]) for row in report[1:] if "cublas_solve_batched" in row[0])
        self.assertEqual(calls, 2, report)
        expected = np.linalg.solve(a.numpy(), b.numpy())
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-4, atol=1e-4)
        # d|x|^2 = 2 Re<x, dx>: gb = solve(a^H, 2x), ga = -sum over the batch of gb x^H
        a_h = a.numpy().conj().T
        gb_expected = np.linalg.solve(a_h, 2 * expected)
        ga_expected = -(gb_expected @ expected.conj().transpose(0, 2, 1)).sum(0)
        np.testing.assert_allclose(gb.numpy(), gb_expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ga.numpy(), ga_expected, rtol=1e-3, atol=1e-3)

    def test_a_batch_of_large_matrices_takes_cusolver_one_after_another(self):
        """Above `_CUBLAS_BATCHED_SOLVE_MAX_N` a batch is `cusolver_solve`
        looping getrf/getrs, still with one status read for the batch."""
        from unittest import mock
        from jittor._runtime.dispatch import select_kernel, _invalidate
        if _helpers._cusolver_ops() is None:
            self.skipTest("this CUDA toolchain has no cuSOLVER")
        n = _helpers._CUBLAS_BATCHED_SOLVE_MAX_N + 1
        a, b = jt.ones((2, n, n), "float32"), jt.ones((2, n, 1), "float32")
        self.assertIs(select_kernel("linalg_solve", a, b), _helpers._cuda_solve)
        with mock.patch.object(_helpers, "_CUBLAS_BATCHED_SOLVE_MAX_N", 0):
            _invalidate()
            self.addCleanup(_invalidate)
            self.assertIs(select_kernel("linalg_solve", a[:, :7, :7], b[:, :7]),
                          _helpers._cuda_solve)
            self._check_batched_solve()
            self._check_singular_batch()

    def _check_singular_batch(self):
        singular = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
        batch = np.stack([np.eye(2, dtype=np.float32), singular])
        with self.assertRaises(np.linalg.LinAlgError) as raised:
            jt.linalg.solve(jt.array(batch), jt.ones((2, 2, 1))).sync()
        self.assertIsInstance(raised.exception, jt.JittorError)
        self.assertIn("matrix 1 of the batch", str(raised.exception))

    def test_singular_input_raises_linalg_error_from_the_native_ops(self):
        singular = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
        self._check_singular_batch()
        for call in (lambda m: jt.linalg.solve(m, jt.ones((2, 1))),
                     lambda m: jt.linalg.inv(m),
                     lambda m: _solve_triangular(m, jt.ones((2, 1)), True)):
            with self.assertRaises(np.linalg.LinAlgError) as raised:
                call(jt.array(singular)).sync()
            # also the core's own error type, as every native op error is
            self.assertIsInstance(raised.exception, jt.JittorError)
            self.assertIn("Singular matrix", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
