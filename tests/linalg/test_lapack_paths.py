"""The paths solve/inv/svd take into LAPACK and cuSOLVER.

On the host the ``numpy_code`` callbacks run numpy's OpenBLAS LAPACK with its
thread count sized to the problem (``_host_lapack``); on CUDA a 2-D solve or
inverse is one ``getrf``/``getrs`` pair with a single status read
(``_cuda_gesv``), and a 2-D reduced SVD is Jacobi ``gesvdj`` with a ``gesvd``
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

    def test_singular_input_raises_linalg_error_from_the_native_ops(self):
        singular = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
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
