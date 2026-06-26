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


if __name__ == "__main__":
    unittest.main(verbosity=2)
