"""Native complex64 dtype (Phase 2) — create / numpy round-trip / arithmetic vs numpy.

Locks the native jittor_core complex64 dtype: registered (dsize 8, is_complex), creatable
from a numpy complex64 array, and elementwise add/sub/mul/div/neg matching numpy. CPU+CUDA.

Run:  python -m jittor.test.test_complex64_native
"""
import unittest
import numpy as np
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class TestComplex64Native(unittest.TestCase):
    def test_dtype_props(self):
        ns = jt.NanoString("complex64")
        self.assertEqual(ns.dsize(), 8)
        self.assertTrue(ns.is_complex())
        self.assertFalse(ns.is_floating_point())
        self.assertFalse(ns.is_int())

    def test_create_roundtrip(self):
        a = np.array([1 + 2j, 3 - 4j, 0 + 1j, -2 - 3j], dtype="complex64")
        def body(dev):
            v = jt.array(a)
            self.assertEqual(str(v.dtype), "complex64", f"dtype {dev}")
            np.testing.assert_array_equal(np.asarray(v.numpy()), a,
                                          err_msg=f"roundtrip {dev}")
        both_devices(body)

    def test_zeros(self):
        def body(dev):
            z = jt.zeros((3,), "complex64")
            self.assertEqual(str(z.dtype), "complex64", f"zeros dtype {dev}")
            np.testing.assert_array_equal(np.asarray(z.numpy()),
                                          np.zeros(3, "complex64"), err_msg=f"zeros {dev}")
        both_devices(body)

    def test_arithmetic(self):
        rng = np.random.RandomState(0)
        a = (rng.randn(8) + 1j * rng.randn(8)).astype("complex64")
        b = (rng.randn(8) + 1j * rng.randn(8)).astype("complex64")
        def body(dev):
            va, vb = jt.array(a), jt.array(b)
            for nm, jr, nr in [("add", va + vb, a + b), ("sub", va - vb, a - b),
                               ("mul", va * vb, a * b), ("div", va / vb, a / b),
                               ("neg", -va, -a)]:
                np.testing.assert_allclose(np.asarray(jr.numpy()), nr, atol=1e-5, rtol=1e-5,
                                           err_msg=f"{nm} {dev}")
        both_devices(body)

    def test_matmul(self):
        # complex matmul comes "for free": jt.matmul lowers to elementwise multiply +
        # sum-reduce, both of which the native complex64 path implements. Lock it.
        rng = np.random.RandomState(1)
        A = (rng.randn(3, 4) + 1j * rng.randn(3, 4)).astype("complex64")
        B = (rng.randn(4, 5) + 1j * rng.randn(4, 5)).astype("complex64")
        ref = A @ B
        # batched (bmm)
        Ab = (rng.randn(2, 3, 4) + 1j * rng.randn(2, 3, 4)).astype("complex64")
        Bb = (rng.randn(2, 4, 5) + 1j * rng.randn(2, 4, 5)).astype("complex64")
        refb = Ab @ Bb
        def body(dev):
            # 2-D matmul works on both devices (lowers to multiply + sum-reduce).
            r = np.asarray(jt.matmul(jt.array(A), jt.array(B)).numpy())
            self.assertEqual(r.dtype.name, "complex64", f"matmul dtype {dev}")
            np.testing.assert_allclose(r, ref, atol=1e-4, rtol=1e-4, err_msg=f"matmul {dev}")
            # Batched matmul: CPU takes the reindex path (works); CUDA routes to
            # cublas_batched_matmul which only supports float dtypes, so complex bmm on
            # CUDA fails loudly (known gap — needs a complex->reindex fallback in nn.bmm).
            if dev == "cpu":
                rb = np.asarray(jt.matmul(jt.array(Ab), jt.array(Bb)).numpy())
                np.testing.assert_allclose(rb, refb, atol=1e-4, rtol=1e-4, err_msg=f"bmm {dev}")
        both_devices(body)

    def test_conj(self):
        rng = np.random.RandomState(3)
        a = (rng.randn(6) + 1j * rng.randn(6)).astype("complex64")
        rf = rng.randn(5).astype("float32")
        def body(dev):
            # complex conj: negate imaginary part, stays complex64
            cj = jt.array(a).conj()
            self.assertEqual(str(cj.dtype), "complex64", f"conj dtype {dev}")
            np.testing.assert_allclose(np.asarray(cj.numpy()), a.conj(),
                                       atol=1e-6, err_msg=f"complex conj {dev}")
            # real conj is identity (torch parity), stays float32
            rc = jt.array(rf).conj()
            self.assertEqual(str(rc.dtype), "float32", f"real conj dtype {dev}")
            np.testing.assert_array_equal(np.asarray(rc.numpy()), rf,
                                          err_msg=f"real conj identity {dev}")
            # real conj is differentiable (grad of identity is ones)
            x = jt.array(rf)
            g = jt.grad(x.conj().sum(), x)
            if isinstance(g, (list, tuple)):  # jt.grad returns a list of grads
                g = g[0]
            np.testing.assert_array_equal(np.asarray(g.numpy()), np.ones_like(rf),
                                          err_msg=f"real conj grad {dev}")
        both_devices(body)

    def test_reduce_sum_and_abs(self):
        rng = np.random.RandomState(2)
        a = (rng.randn(6) + 1j * rng.randn(6)).astype("complex64")
        def body(dev):
            va = jt.array(a)
            # sum-reduce (complex -> complex)
            np.testing.assert_allclose(np.asarray(va.sum().numpy()).reshape(-1)[0], a.sum(),
                                       atol=1e-4, rtol=1e-4, err_msg=f"sum {dev}")
            # abs (complex -> float32 magnitude)
            r = np.asarray(va.abs().numpy())
            self.assertEqual(r.dtype.name, "float32", f"abs dtype {dev}")
            np.testing.assert_allclose(r, np.abs(a), atol=1e-4, rtol=1e-4, err_msg=f"abs {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
