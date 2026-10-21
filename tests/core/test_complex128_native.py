# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native complex128 dtype (closes KI-COMPLEX-001; see
# docs/notes/complex-dtype.md). Before this, `NanoString::_dsize_nbits` (2
# bits) capped every dtype's size at 8 bytes, so a genuine 16-byte complex128
# Var could not be represented at all -- `NPY_CDOUBLE` mapped to `ns_void`
# and any numpy/plain-Python complex128 input raised loudly ("Numpy type not
# support"). This suite exercises the acceptance list the closing issue asks
# for: lossless numpy round-trip, a real 16-byte dtype, mixed complex64/
# complex128 promotion, CPU+CUDA arithmetic and first-order gradients,
# scalar `.item()`, complex64<->complex128 cast (values AND gradients), and
# the linalg bridge (inv/svd/qr/eigh/pinv) at complex128 precision.
#
# FFT's own complex128 kernel support is a separate cuFFT/kernel-level piece
# not in this issue's owner list and is NOT covered here -- see
# tests/opinfo/definitions/fft.py's `_NO_COMPLEX128` marker, left in place.
import unittest

import numpy as np
import jittor as jt
from jittor import linalg

from _helpers import capability as _test_capability
from _helpers.cupy_bridge import cuda_numpy_code_available
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy


def _c128(a):
    return jt.array(np.asarray(a, dtype=np.complex128))


def _c64(a):
    return jt.array(np.asarray(a, dtype=np.complex64))


@_test_preserve_policy(jt, 'use_cuda')
class _Mixin:
    use_cuda = 0

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self.use_cuda))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

    # ------------------------------------------------------------- registry

    def test_dtype_registers_as_16_bytes(self):
        v = _c128([1 + 2j, 3 - 4j])
        self.assertEqual(str(v.dtype), "complex128")
        self.assertEqual(v.dtype.dsize(), 16)
        self.assertTrue(v.dtype.is_complex())

    def test_lossless_numpy_round_trip(self):
        rng = np.random.RandomState(0)
        a = (rng.randn(3, 5) + 1j * rng.randn(3, 5)).astype(np.complex128)
        v = jt.array(a)
        self.assertEqual(str(v.dtype), "complex128")
        back = v.numpy()
        self.assertEqual(back.dtype, np.complex128)
        # bit-for-bit: this is a round trip through a 16-byte dtype, not a
        # narrow-then-compare -- any precision loss anywhere in the bridge
        # would show up here.
        np.testing.assert_array_equal(back, a)

    # -------------------------------------------------------------- promote

    def test_mixed_precision_promotes_to_complex128(self):
        c64 = _c64([1 + 1j])
        f64 = jt.array(np.array([2.0], dtype=np.float64), dtype="float64")
        self.assertEqual(str((c64 * f64).dtype), "complex128")
        c128 = _c128([1 + 1j])
        f32 = jt.array(np.array([2.0], dtype=np.float32))
        self.assertEqual(str((c128 * f32).dtype), "complex128")

    def test_complex64_stays_complex64_with_narrow_operand(self):
        c64 = _c64([1 + 1j])
        f32 = jt.array(np.array([2.0], dtype=np.float32))
        self.assertEqual(str((c64 * f32).dtype), "complex64")

    # --------------------------------------------------------- arithmetic

    def test_arithmetic_and_transcendentals(self):
        a = _c128([1 + 2j, 3 + 4j])
        b = _c128([2 + 1j, 1 - 1j])
        np.testing.assert_allclose((a * b + a).numpy(), a.numpy() * b.numpy() + a.numpy())
        absa = jt.abs(a)
        self.assertEqual(str(absa.dtype), "float64")
        np.testing.assert_allclose(absa.numpy(), np.abs(a.numpy()))
        np.testing.assert_allclose(jt.conj(a).numpy(), np.conj(a.numpy()))
        np.testing.assert_allclose(jt.exp(a).numpy(), np.exp(a.numpy()), rtol=1e-10)

    def test_first_order_gradient(self):
        a = _c128([1 + 2j, 3 + 4j])
        loss = (jt.abs(a) ** 2).sum()
        g = jt.grad(loss, a)
        self.assertEqual(str(g.dtype), "complex128")
        np.testing.assert_allclose(g.numpy(), 2 * a.numpy(), rtol=1e-10)

    # ------------------------------------------------------------- .item()

    def test_item_both_widths(self):
        v64 = _c64([1 + 2j])[0]
        v128 = _c128([1 + 2j])[0]
        self.assertEqual(v64.item(), 1 + 2j)
        self.assertEqual(v128.item(), 1 + 2j)
        self.assertIsInstance(v64.item(), complex)
        self.assertIsInstance(v128.item(), complex)

    # ----------------------------------------------------------- cast

    def test_cast_values_both_directions(self):
        a = _c64([1 + 2j, 3 + 4j])
        up = a.cast("complex128")
        self.assertEqual(str(up.dtype), "complex128")
        np.testing.assert_allclose(up.numpy(), a.numpy())

        b = _c128([1 + 2j, 3 + 4j])
        down = b.cast("complex64")
        self.assertEqual(str(down.dtype), "complex64")
        np.testing.assert_allclose(down.numpy(), b.numpy())

    def test_cast_gradients_both_directions(self):
        a = _c64([1 + 2j, 3 + 4j])
        loss_up = (jt.abs(a.cast("complex128")) ** 2).sum()
        g_up = jt.grad(loss_up, a)
        self.assertEqual(str(g_up.dtype), "complex64")
        np.testing.assert_allclose(g_up.numpy(), 2 * a.numpy(), rtol=1e-5)

        b = _c128([1 + 2j, 3 + 4j])
        loss_down = (jt.abs(b.cast("complex64")) ** 2).sum()
        g_down = jt.grad(loss_down, b)
        self.assertEqual(str(g_down.dtype), "complex128")
        np.testing.assert_allclose(g_down.numpy(), 2 * b.numpy(), rtol=1e-5)

    # ------------------------------------------------------------- linalg

    def test_linalg_bridge_high_precision(self):
        rng = np.random.RandomState(1)

        def rc(*shape):
            return _c128(rng.randn(*shape) + 1j * rng.randn(*shape))

        A = rc(4, 4)
        Ai = linalg.inv(A)
        self.assertEqual(str(Ai.dtype), "complex128")
        np.testing.assert_allclose((A.numpy() @ Ai.numpy()), np.eye(4), atol=1e-8)

        A2 = rc(4, 3)
        u, s, vh = linalg.svd(A2)
        self.assertEqual(str(u.dtype), "complex128")
        rec = (u.numpy() * s.real.numpy()[..., None, :]) @ vh.numpy()
        np.testing.assert_allclose(rec, A2.numpy(), atol=1e-8)

        q, r = linalg.qr(A2)
        self.assertEqual(str(q.dtype), "complex128")
        np.testing.assert_allclose(q.numpy() @ r.numpy(), A2.numpy(), atol=1e-8)

        H = A + A.transpose(-1, -2).conj()
        w, v = linalg.eigh(H)
        self.assertEqual(str(w.dtype), "complex128")
        # w is real-valued (carried as complex128 with zero imag); check the
        # eigendecomposition identity at complex128 precision.
        wv = w.numpy()
        vv = v.numpy()
        rec_h = vv @ np.diag(wv) @ np.conj(vv.T)
        np.testing.assert_allclose(rec_h, H.numpy(), atol=1e-8)

        Ap = linalg.pinv(A2)
        self.assertEqual(str(Ap.dtype), "complex128")
        np.testing.assert_allclose(A2.numpy() @ Ap.numpy() @ A2.numpy(), A2.numpy(), atol=1e-7)


# complex128 linalg runs through jt.numpy_code (same as complex64), and
# py_converter hands that callback `cupy` instead of `numpy` when use_cuda is
# on -- without CuPy the operator raises from inside execution. See
# test_complex64_linalg.py for the full rationale.
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "no cuda found")
@unittest.skipIf(not cuda_numpy_code_available(),
                 "CUDA numpy-code operators need CuPy; it is not installed")
class TestComplex128NativeCUDA(_Mixin, unittest.TestCase):
    use_cuda = 1


class TestComplex128NativeCPU(_Mixin, unittest.TestCase):
    use_cuda = 0


if __name__ == "__main__":
    unittest.main()
