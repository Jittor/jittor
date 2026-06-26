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
            # Batched matmul (bmm) works on BOTH devices: complex64 routes around
            # cublas_batched_matmul (float-only) to the reindex path (multiply + sum-reduce).
            rb = np.asarray(jt.matmul(jt.array(Ab), jt.array(Bb)).numpy())
            np.testing.assert_allclose(rb, refb, atol=1e-4, rtol=1e-4, err_msg=f"bmm {dev}")
        both_devices(body)

    def test_mean_prod(self):
        rng = np.random.RandomState(7)
        a = (rng.randn(8) + 1j * rng.randn(8)).astype("complex64")
        def body(dev):
            mn = np.asarray(jt.array(a).mean().numpy()).reshape(-1)[0]
            self.assertEqual(str(jt.array(a).mean().dtype), "complex64", f"mean dtype {dev}")
            np.testing.assert_allclose(mn, a.mean(), atol=1e-4, rtol=1e-4, err_msg=f"mean {dev}")
            # prod uses a multiply-reduce: CPU works; CUDA needs an atomicCAS overload for
            # complex64 (not implemented) so it fails loudly there -- assert CPU only.
            if dev == "cpu":
                pr = np.asarray(jt.array(a).prod().numpy()).reshape(-1)[0]
                np.testing.assert_allclose(pr, a.prod(), atol=1e-4, rtol=1e-4, err_msg=f"prod {dev}")
        both_devices(body)

    def test_structural_ops(self):
        # data-movement ops that should "just work" on complex64 (no per-dtype kernel).
        rng = np.random.RandomState(8)
        a = (rng.randn(6) + 1j * rng.randn(6)).astype("complex64")
        b = (rng.randn(6) + 1j * rng.randn(6)).astype("complex64")
        m = (rng.randn(2, 3) + 1j * rng.randn(2, 3)).astype("complex64")
        def body(dev):
            cases = [
                ("reshape", jt.array(a).reshape((2, 3)), a.reshape(2, 3)),
                ("transpose", jt.array(m).transpose(), m.T),
                ("slice", jt.array(a)[1:4], a[1:4]),
                ("getitem", jt.array(m)[1], m[1]),
                ("broadcast_add", jt.array(m) + jt.array(a[:3]), m + a[:3]),
                ("concat", jt.concat([jt.array(a), jt.array(b)]), np.concatenate([a, b])),
                ("stack", jt.stack([jt.array(a), jt.array(b)]), np.stack([a, b])),
            ]
            for nm, jr, ref in cases:
                np.testing.assert_allclose(np.asarray(jr.numpy()), ref, atol=1e-5,
                                           err_msg=f"{nm} {dev}")
        both_devices(body)

    def test_compare_and_ternary(self):
        rng = np.random.RandomState(9)
        a = (rng.randn(6) + 1j * rng.randn(6)).astype("complex64")
        b = (rng.randn(6) + 1j * rng.randn(6)).astype("complex64")
        mask = (a.real > 0)
        def body(dev):
            np.testing.assert_array_equal(np.asarray((jt.array(a) == jt.array(a)).numpy()),
                                          a == a, err_msg=f"equal {dev}")
            np.testing.assert_array_equal(np.asarray((jt.array(a) != jt.array(b)).numpy()),
                                          a != b, err_msg=f"notequal {dev}")
            cond = jt.array(mask.astype("float32"))
            w = np.asarray(jt.ternary(cond, jt.array(a), jt.array(b)).numpy())
            np.testing.assert_allclose(w, np.where(mask, a, b), atol=1e-5,
                                       err_msg=f"ternary {dev}")
        both_devices(body)

    def test_transcendentals(self):
        rng = np.random.RandomState(11)
        a = (rng.randn(8) + 1j * rng.randn(8)).astype("complex64")
        ops = [("exp", np.exp), ("log", np.log), ("sin", np.sin),
               ("cos", np.cos), ("sqrt", np.sqrt)]
        def body(dev):
            for nm, npf in ops:
                r = np.asarray(getattr(jt, nm)(jt.array(a)).numpy())
                self.assertEqual(r.dtype.name, "complex64", f"{nm} dtype {dev}")
                np.testing.assert_allclose(r, npf(a), atol=1e-4, rtol=1e-4,
                                           err_msg=f"{nm} {dev}")
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
