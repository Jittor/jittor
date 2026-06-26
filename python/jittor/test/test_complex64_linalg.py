# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Phase 6 "P4": verify the NATIVE complex64 path of jt.linalg.{inv,svd,svdvals,
# qr,eig,eigh,pinv}. Each public entry point bridges a native complex64 Var to
# the legacy nn.ComplexNumber implementation and returns native complex64. We
# compare the FORWARD result to numpy via reconstruction (eig/svd/qr/inv/eigh
# have sign/phase/order ambiguity, so identities are checked, not raw values),
# on BOTH the CUDA and the CPU backend, and assert the legacy ComplexNumber
# input path is unchanged (no regression).
import unittest
import numpy as np
import jittor as jt
from jittor import linalg


def _to_complex64_var(a):
    """numpy complex array -> native complex64 jt.Var."""
    return jt.array(a.astype("complex64"))


def _np(z):
    """native complex64 jt.Var -> numpy complex128 array (for comparison)."""
    assert isinstance(z, jt.Var) and "complex" in str(z.dtype), \
        f"expected native complex64 Var, got {type(z)} dtype={getattr(z,'dtype',None)}"
    return z.numpy().astype("complex128")


def _dot(a, b):
    return np.einsum("...ij,...jk->...ik", a, b)


def _diag_embed(s):
    # s: (...,K) -> (...,K,K) diagonal matrices
    k = s.shape[-1]
    out = np.zeros(s.shape + (k,), dtype=s.dtype)
    idx = np.arange(k)
    out[..., idx, idx] = s
    return out


class _Mixin:
    use_cuda = 0

    def setUp(self):
        jt.flags.use_cuda = self.use_cuda

    # -------------------------------------------------------------------- inv
    def test_inv(self):
        rng = np.random.RandomState(0)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex64_var(a)
        zi = linalg.inv(z)
        self.assertTrue("complex" in str(zi.dtype), "inv must return native complex64")
        eye = np.broadcast_to(np.eye(4), (2, 4, 4))
        rec = _dot(a, _np(zi))
        np.testing.assert_allclose(rec, eye, atol=1e-3, rtol=1e-3)
        # also a @ inv(a) via inv(a) @ a
        rec2 = _dot(_np(zi), a)
        np.testing.assert_allclose(rec2, eye, atol=1e-3, rtol=1e-3)

    # -------------------------------------------------------------------- svd
    def test_svd(self):
        rng = np.random.RandomState(1)
        a = (rng.randn(3, 5, 5) + 1j * rng.randn(3, 5, 5))
        z = _to_complex64_var(a)
        u, s, vh = linalg.svd(z)
        for name, t in (("u", u), ("s", s), ("vh", vh)):
            self.assertTrue("complex" in str(t.dtype), f"svd {name} must be native complex64")
        un, sn, vhn = _np(u), _np(s), _np(vh)
        # singular values are real and non-negative
        np.testing.assert_allclose(sn.imag, 0, atol=1e-3)
        # reconstruction: a == u @ diag(s) @ vh  (numpy svd returns vh already)
        rec = _dot(_dot(un, _diag_embed(sn)), vhn)
        np.testing.assert_allclose(rec, a, atol=1e-3, rtol=1e-3)

    def test_svd_nonsquare(self):
        rng = np.random.RandomState(11)
        a = (rng.randn(2, 6, 3) + 1j * rng.randn(2, 6, 3))  # M>N
        z = _to_complex64_var(a)
        u, s, vh = linalg.svd(z)
        un, sn, vhn = _np(u), _np(s), _np(vh)
        rec = _dot(_dot(un, _diag_embed(sn)), vhn)
        np.testing.assert_allclose(rec, a, atol=1e-3, rtol=1e-3)

    def test_svdvals(self):
        rng = np.random.RandomState(2)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        z = _to_complex64_var(a)
        s = linalg.svdvals(z)
        sn = _np(s)
        ref = np.linalg.svd(a, compute_uv=False)
        np.testing.assert_allclose(np.sort(sn.real), np.sort(ref), atol=1e-3, rtol=1e-3)

    # --------------------------------------------------------------------- qr
    def test_qr(self):
        rng = np.random.RandomState(3)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex64_var(a)
        q, r = linalg.qr(z)
        self.assertTrue("complex" in str(q.dtype) and "complex" in str(r.dtype),
                        "qr q,r must be native complex64")
        qn, rn = _np(q), _np(r)
        # q @ r == a
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)
        # q unitary: q^H q == I
        qhq = _dot(np.conj(np.swapaxes(qn, -1, -2)), qn)
        np.testing.assert_allclose(qhq, np.broadcast_to(np.eye(4), (2, 4, 4)),
                                   atol=1e-3, rtol=1e-3)

    # -------------------------------------------------------------------- eig
    def test_eig(self):
        if self.use_cuda:
            # PRE-EXISTING platform limitation (not a regression of this bridge):
            # general non-Hermitian eig runs through jt.numpy_code, which binds
            # `np` to cupy on CUDA, and cupy.linalg has NO `eig` (only `eigh`).
            # The legacy ComplexNumber eig path is broken on CUDA for the same
            # reason; the native bridge faithfully reuses it. Verified on CPU.
            self.skipTest("cupy.linalg has no eig() (general eig is CPU-only); "
                          "pre-existing — see test_eig docstring")
        rng = np.random.RandomState(4)
        a = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        z = _to_complex64_var(a)
        w, v = linalg.eig(z)
        self.assertTrue("complex" in str(w.dtype) and "complex" in str(v.dtype),
                        "eig w,v must be native complex64")
        wn, vn = _np(w), _np(v)
        # a @ v == v @ diag(w)
        lhs = _dot(a, vn)
        rhs = _dot(vn, _diag_embed(wn))
        np.testing.assert_allclose(lhs, rhs, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------- eigh
    def test_eigh(self):
        rng = np.random.RandomState(5)
        b = (rng.randn(2, 4, 4) + 1j * rng.randn(2, 4, 4))
        a = b + np.conj(np.swapaxes(b, -1, -2))  # Hermitian
        z = _to_complex64_var(a)
        w, v = linalg.eigh(z)
        self.assertTrue("complex" in str(w.dtype) and "complex" in str(v.dtype),
                        "eigh w,v must be native complex64")
        wn, vn = _np(w), _np(v)
        # eigenvalues of a Hermitian matrix are real
        np.testing.assert_allclose(wn.imag, 0, atol=1e-3)
        # a @ v == v @ diag(w)
        lhs = _dot(a, vn)
        rhs = _dot(vn, _diag_embed(wn))
        np.testing.assert_allclose(lhs, rhs, atol=1e-3, rtol=1e-3)
        # eigh reads the LOWER triangle (UPLO='L'); compare to numpy eigh values
        ref = np.linalg.eigh(a, UPLO='L')[0]
        np.testing.assert_allclose(np.sort(wn.real, axis=-1), ref, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------- pinv
    def test_pinv_square(self):
        rng = np.random.RandomState(6)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        z = _to_complex64_var(a)
        p = linalg.pinv(z)
        self.assertTrue("complex" in str(p.dtype), "pinv must be native complex64")
        pn = _np(p)
        # Moore-Penrose: A P A == A
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-3, rtol=1e-3)

    def test_pinv_nonsquare(self):
        rng = np.random.RandomState(7)
        a = (rng.randn(2, 5) + 1j * rng.randn(2, 5))  # 2x5 -> pinv 5x2
        z = _to_complex64_var(a)
        p = linalg.pinv(z)
        pn = _np(p)
        self.assertEqual(tuple(p.shape), (5, 2))
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-3, rtol=1e-3)
        ref = np.linalg.pinv(a)
        np.testing.assert_allclose(pn, ref, atol=1e-3, rtol=1e-3)

    # ---------------------------------------------------- legacy CN regression
    def test_complexnumber_input_unchanged(self):
        # The legacy nn.ComplexNumber input path must still return a
        # ComplexNumber with correct values (no regression from the bridge).
        rng = np.random.RandomState(8)
        a = (rng.randn(4, 4) + 1j * rng.randn(4, 4))
        cn = jt.nn.ComplexNumber(jt.array(a.real.astype("float32")),
                                 jt.array(a.imag.astype("float32")))

        # inv
        cni = linalg.inv(cn)
        self.assertIsInstance(cni, jt.nn.ComplexNumber)
        inv_np = cni.real.numpy() + 1j * cni.imag.numpy()
        np.testing.assert_allclose(_dot(a, inv_np), np.eye(4), atol=1e-3, rtol=1e-3)

        # qr
        cq, cr = linalg.qr(cn)
        self.assertIsInstance(cq, jt.nn.ComplexNumber)
        self.assertIsInstance(cr, jt.nn.ComplexNumber)
        qn = cq.real.numpy() + 1j * cq.imag.numpy()
        rn = cr.real.numpy() + 1j * cr.imag.numpy()
        np.testing.assert_allclose(_dot(qn, rn), a, atol=1e-3, rtol=1e-3)

        # svd (returns the SVD namedtuple of ComplexNumbers)
        su, ss, sv = linalg.svd(cn)
        self.assertIsInstance(su, jt.nn.ComplexNumber)
        un = su.real.numpy() + 1j * su.imag.numpy()
        sn = ss.real.numpy() + 1j * ss.imag.numpy()
        vn = sv.real.numpy() + 1j * sv.imag.numpy()
        rec = _dot(_dot(un, _diag_embed(sn)), vn)
        np.testing.assert_allclose(rec, a, atol=1e-3, rtol=1e-3)

        # eig (CPU only: cupy.linalg has no eig — pre-existing, see test_eig)
        if not self.use_cuda:
            ew, ev = linalg.eig(cn)
            self.assertIsInstance(ew, jt.nn.ComplexNumber)
            self.assertIsInstance(ev, jt.nn.ComplexNumber)
            wn = ew.real.numpy() + 1j * ew.imag.numpy()
            evn = ev.real.numpy() + 1j * ev.imag.numpy()
            np.testing.assert_allclose(_dot(a, evn), _dot(evn, _diag_embed(wn)),
                                       atol=1e-3, rtol=1e-3)

        # eigh (Hermitian) — previously eigh(ComplexNumber) raised; now supported
        h = a + np.conj(a.T)
        cnh = jt.nn.ComplexNumber(jt.array(h.real.astype("float32")),
                                  jt.array(h.imag.astype("float32")))
        hw, hv = linalg.eigh(cnh)
        self.assertIsInstance(hw, jt.nn.ComplexNumber)
        self.assertIsInstance(hv, jt.nn.ComplexNumber)
        hwn = hw.real.numpy() + 1j * hw.imag.numpy()
        hvn = hv.real.numpy() + 1j * hv.imag.numpy()
        np.testing.assert_allclose(_dot(h, hvn), _dot(hvn, _diag_embed(hwn)),
                                   atol=1e-3, rtol=1e-3)

        # pinv — previously pinv(ComplexNumber) raised; now supported
        cp = linalg.pinv(cn)
        self.assertIsInstance(cp, jt.nn.ComplexNumber)
        pn = cp.real.numpy() + 1j * cp.imag.numpy()
        np.testing.assert_allclose(_dot(_dot(a, pn), a), a, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------- real path intact
    def test_real_path_unchanged(self):
        # real inputs must still go through the real code path unchanged.
        rng = np.random.RandomState(9)
        a = rng.randn(4, 4).astype("float32")
        z = jt.array(a)
        ri = linalg.inv(z)
        self.assertFalse("complex" in str(ri.dtype))
        np.testing.assert_allclose(_dot(a, ri.numpy()), np.eye(4), atol=1e-3, rtol=1e-3)
        # symmetric for eigh
        s = a + a.T
        w, v = linalg.eigh(jt.array(s))
        self.assertFalse("complex" in str(w.dtype))
        ref = np.linalg.eigh(s, UPLO='L')[0]
        np.testing.assert_allclose(np.sort(w.numpy()), ref, atol=1e-3, rtol=1e-3)


@unittest.skipIf(not jt.has_cuda, "no cuda found")
class TestComplex64LinalgCUDA(_Mixin, unittest.TestCase):
    use_cuda = 1


class TestComplex64LinalgCPU(_Mixin, unittest.TestCase):
    use_cuda = 0


if __name__ == "__main__":
    unittest.main()
