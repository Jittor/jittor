"""Torch-grade FFT / einsum / complex-number semantics tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (sibling of ``test_torch_compat_ops.py``).
Every check compares jittor-as-torch against an INDEPENDENT ``numpy`` reference
(``numpy.fft``, ``numpy.einsum``, numpy complex arithmetic) and runs on BOTH CPU and
CUDA when the build has it, locking the torch-facing *semantics* rather than jittor
self-consistency.

Notes on jittor's representation (verified against the source):
  * ``torch.fft.fft/ifft/fft2/ifft2/fftn/ifftn/rfft`` return a ``jt.nn.ComplexNumber``
    (a real/imag pair, NOT a native-complex Var and NOT a 2-tuple); read parts via
    ``.real`` / ``.imag``. ``torch.fft.irfft`` is the exception -> it returns a real Var.
  * The DFT is computed via matmul with cos/sin matrices (O(N^2), autograd-able,
    dual-backend), so results are float32-accurate but not bit-exact to numpy's FFTW;
    tolerances are set accordingly.
  * jittor has no 0-d scalar Var: a scalar-output einsum (e.g. ``ii->``) yields shape
    ``[1]`` rather than numpy's ``()``. Those are compared via ``.item()``.

Run:  python -m jittor.test.test_torch_compat_fft_einsum
      python -m pytest python/jittor/test/test_torch_compat_fft_einsum.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref), err_msg=msg)

    def acplx(self, cn, ref, atol=1e-4, msg=""):
        """Compare a ComplexNumber against a numpy complex array (real + imag parts)."""
        self.ac(cn.real.numpy(), np.real(ref), atol=atol, msg=f"{msg} (real)")
        self.ac(cn.imag.numpy(), np.imag(ref), atol=atol, msg=f"{msg} (imag)")


# ---------------------------------------------------------------------------- FFT

class TestFFT1D(Base):
    def setUp(self):
        self.x = np.random.RandomState(0).randn(8).astype("float32")
        self.xb = np.random.RandomState(1).randn(3, 8).astype("float32")  # batched

    def test_fft(self):
        x = self.x
        def body(dev):
            self.acplx(torch.fft.fft(torch.tensor(x)), np.fft.fft(x), msg=f"fft {dev}")
        both_devices(body)

    def test_ifft(self):
        x = self.x
        def body(dev):
            self.acplx(torch.fft.ifft(torch.tensor(x)), np.fft.ifft(x), msg=f"ifft {dev}")
        both_devices(body)

    def test_fft_ifft_roundtrip(self):
        x = self.x
        def body(dev):
            spec = torch.fft.fft(torch.tensor(x))
            back = torch.fft.ifft(spec)
            self.ac(back.real.numpy(), x, atol=1e-4, msg=f"fft->ifft roundtrip {dev}")
            self.ac(back.imag.numpy(), np.zeros_like(x), atol=1e-4,
                    msg=f"fft->ifft imag~0 {dev}")
        both_devices(body)

    def test_fft_batched_along_dim(self):
        x = self.xb
        def body(dev):
            self.acplx(torch.fft.fft(torch.tensor(x), dim=-1), np.fft.fft(x, axis=-1),
                       msg=f"fft batched dim=-1 {dev}")
            self.acplx(torch.fft.fft(torch.tensor(x), dim=0), np.fft.fft(x, axis=0),
                       msg=f"fft batched dim=0 {dev}")
        both_devices(body)

    def test_fft_norm_ortho(self):
        x = self.x
        def body(dev):
            self.acplx(torch.fft.fft(torch.tensor(x), norm="ortho"),
                       np.fft.fft(x, norm="ortho"), msg=f"fft norm=ortho {dev}")
            self.acplx(torch.fft.ifft(torch.tensor(x), norm="ortho"),
                       np.fft.ifft(x, norm="ortho"), msg=f"ifft norm=ortho {dev}")
        both_devices(body)

    def test_fft_n_truncate_pad(self):
        x = self.x  # length 8
        def body(dev):
            # n < len  -> truncate;  n > len -> zero-pad (numpy does the same)
            self.acplx(torch.fft.fft(torch.tensor(x), n=4), np.fft.fft(x, n=4),
                       msg=f"fft n=4 (truncate) {dev}")
            self.acplx(torch.fft.fft(torch.tensor(x), n=12), np.fft.fft(x, n=12),
                       msg=f"fft n=12 (pad) {dev}")
        both_devices(body)


class TestRFFT(Base):
    def setUp(self):
        self.x = np.random.RandomState(2).randn(8).astype("float32")

    def test_rfft(self):
        x = self.x
        def body(dev):
            r = torch.fft.rfft(torch.tensor(x))
            ref = np.fft.rfft(x)          # length n//2 + 1
            self.assertEqual(tuple(r.real.shape), ref.shape, f"rfft shape {dev}")
            self.acplx(r, ref, msg=f"rfft {dev}")
        both_devices(body)

    def test_irfft_roundtrip(self):
        x = self.x
        def body(dev):
            r = torch.fft.rfft(torch.tensor(x))
            back = torch.fft.irfft(r, n=8)
            self.assertFalse(torch.is_complex(back), f"irfft must return a real Var {dev}")
            self.ac(back.numpy(), x, atol=1e-4, msg=f"rfft->irfft roundtrip {dev}")
        both_devices(body)

    def test_irfft_matches_numpy(self):
        x = self.x
        def body(dev):
            # feed the SAME (numpy-derived) half-spectrum to both and compare
            ref_spec = np.fft.rfft(x)
            cn = torch.complex(torch.tensor(ref_spec.real.astype("float32")),
                               torch.tensor(ref_spec.imag.astype("float32")))
            got = torch.fft.irfft(cn, n=8).numpy()
            self.ac(got, np.fft.irfft(ref_spec, n=8), atol=1e-4,
                    msg=f"irfft vs numpy {dev}")
        both_devices(body)


class TestFFT2D(Base):
    def setUp(self):
        self.x = np.random.RandomState(3).randn(4, 6).astype("float32")

    def test_fft2(self):
        x = self.x
        def body(dev):
            self.acplx(torch.fft.fft2(torch.tensor(x)), np.fft.fft2(x), atol=1e-3,
                       msg=f"fft2 {dev}")
        both_devices(body)

    def test_ifft2_roundtrip(self):
        x = self.x
        def body(dev):
            back = torch.fft.ifft2(torch.fft.fft2(torch.tensor(x)))
            self.ac(back.real.numpy(), x, atol=1e-3, msg=f"fft2->ifft2 roundtrip {dev}")
        both_devices(body)

    def test_fftn(self):
        x = self.x
        def body(dev):
            self.acplx(torch.fft.fftn(torch.tensor(x), dim=(-2, -1)),
                       np.fft.fftn(x, axes=(-2, -1)), atol=1e-3, msg=f"fftn {dev}")
        both_devices(body)

    def test_ifftn_roundtrip(self):
        x = self.x
        def body(dev):
            spec = torch.fft.fftn(torch.tensor(x), dim=(-2, -1))
            back = torch.fft.ifftn(spec, dim=(-2, -1))
            self.ac(back.real.numpy(), x, atol=1e-3, msg=f"fftn->ifftn roundtrip {dev}")
        both_devices(body)


class TestFFTShift(Base):
    def test_fftshift_even(self):
        v = np.arange(8).astype("float32")
        def body(dev):
            self.ac(torch.fft.fftshift(torch.tensor(v)).numpy(), np.fft.fftshift(v),
                    msg=f"fftshift even {dev}")
            self.ac(torch.fft.ifftshift(torch.tensor(v)).numpy(), np.fft.ifftshift(v),
                    msg=f"ifftshift even {dev}")
        both_devices(body)

    def test_fftshift_odd_roundtrip(self):
        # odd length: fftshift then ifftshift must recover the original exactly.
        v = np.arange(7).astype("float32")
        def body(dev):
            t = torch.tensor(v)
            self.ac(torch.fft.ifftshift(torch.fft.fftshift(t)).numpy(), v,
                    msg=f"fftshift/ifftshift odd roundtrip {dev}")
        both_devices(body)

    def test_fftshift_2d(self):
        x = np.random.RandomState(4).randn(4, 6).astype("float32")
        def body(dev):
            self.ac(torch.fft.fftshift(torch.tensor(x)).numpy(), np.fft.fftshift(x),
                    msg=f"fftshift 2d all-dims {dev}")
            self.ac(torch.fft.fftshift(torch.tensor(x), dim=1).numpy(),
                    np.fft.fftshift(x, axes=1), msg=f"fftshift 2d dim=1 {dev}")
        both_devices(body)

    def test_fftshift_complex(self):
        # fftshift must accept a ComplexNumber and roll both parts.
        v = np.arange(8).astype("float32")
        def body(dev):
            spec = torch.fft.fft(torch.tensor(v))
            shifted = torch.fft.fftshift(spec)
            self.assertTrue(torch.is_complex(shifted), f"fftshift(complex) stays complex {dev}")
            ref = np.fft.fftshift(np.fft.fft(v))
            self.acplx(shifted, ref, msg=f"fftshift complex {dev}")
        both_devices(body)


# ------------------------------------------------------------------------ complex

class TestComplex(Base):
    def setUp(self):
        rs = np.random.RandomState(5)
        self.re = rs.randn(3, 4).astype("float32")
        self.im = rs.randn(3, 4).astype("float32")
        self.z = self.re + 1j * self.im

    def _cn(self):
        return torch.complex(torch.tensor(self.re), torch.tensor(self.im))

    def test_complex_real_imag(self):
        def body(dev):
            c = self._cn()
            self.ac(c.real.numpy(), self.re, msg=f"complex.real {dev}")
            self.ac(c.imag.numpy(), self.im, msg=f"complex.imag {dev}")
            self.assertTrue(torch.is_complex(c), f"is_complex {dev}")
            self.assertFalse(torch.is_complex(torch.tensor(self.re)), f"is_complex(real) {dev}")
        both_devices(body)

    def test_top_level_real_imag(self):
        def body(dev):
            c = self._cn()
            self.ac(torch.real(c).numpy(), self.re, msg=f"torch.real {dev}")
            self.ac(torch.imag(c).numpy(), self.im, msg=f"torch.imag {dev}")
            # torch.real/imag on a real tensor: real passes through, imag is zeros.
            self.ac(torch.real(torch.tensor(self.re)).numpy(), self.re,
                    msg=f"torch.real(real) {dev}")
            self.ac(torch.imag(torch.tensor(self.re)).numpy(), np.zeros_like(self.re),
                    msg=f"torch.imag(real) {dev}")
        both_devices(body)

    def test_conj(self):
        def body(dev):
            c = self._cn()
            self.acplx(c.conj(), np.conj(self.z), msg=f"conj method {dev}")
            self.acplx(torch.conj(c), np.conj(self.z), msg=f"torch.conj {dev}")
        both_devices(body)

    def test_abs(self):
        def body(dev):
            c = self._cn()
            # torch.abs of a complex tensor is its magnitude.
            self.ac(c.abs().numpy(), np.abs(self.z), atol=1e-5, msg=f"complex.abs {dev}")
            self.ac(torch.abs(c).numpy(), np.abs(self.z), atol=1e-5, msg=f"torch.abs(complex) {dev}")
            self.ac(c.norm().numpy(), np.abs(self.z), atol=1e-5, msg=f"complex.norm {dev}")
        both_devices(body)

    def test_angle(self):
        # angle = atan2(imag, real); jittor's float32 atan2 is ~1e-3 looser than numpy.
        def body(dev):
            c = self._cn()
            self.ac(c.angle().numpy(), np.angle(self.z), atol=3e-3, msg=f"complex.angle {dev}")
            self.ac(torch.angle(c).numpy(), np.angle(self.z), atol=3e-3,
                    msg=f"torch.angle {dev}")
        both_devices(body)

    def test_arithmetic(self):
        rs = np.random.RandomState(6)
        re2 = rs.randn(3, 4).astype("float32"); im2 = rs.randn(3, 4).astype("float32")
        z2 = re2 + 1j * im2
        def body(dev):
            a = self._cn()
            b = torch.complex(torch.tensor(re2), torch.tensor(im2))
            self.acplx(a + b, self.z + z2, msg=f"complex add {dev}")
            self.acplx(a - b, self.z - z2, msg=f"complex sub {dev}")
            self.acplx(a * b, self.z * z2, msg=f"complex mul {dev}")
            self.acplx(a / b, self.z / z2, atol=1e-4, msg=f"complex div {dev}")
        both_devices(body)

    def test_exp(self):
        def body(dev):
            c = self._cn()
            self.acplx(c.exp(), np.exp(self.z), atol=1e-4, msg=f"complex exp {dev}")
        both_devices(body)

    def test_matmul(self):
        rs = np.random.RandomState(7)
        ar = rs.randn(3, 4).astype("float32"); ai = rs.randn(3, 4).astype("float32")
        br = rs.randn(4, 5).astype("float32"); bi = rs.randn(4, 5).astype("float32")
        za = ar + 1j * ai; zb = br + 1j * bi
        def body(dev):
            a = torch.complex(torch.tensor(ar), torch.tensor(ai))
            b = torch.complex(torch.tensor(br), torch.tensor(bi))
            self.acplx(a @ b, za @ zb, atol=1e-4, msg=f"complex matmul {dev}")
        both_devices(body)


class TestViewAsComplexReal(Base):
    def setUp(self):
        self.re = np.random.RandomState(8).randn(3, 4).astype("float32")
        self.im = np.random.RandomState(9).randn(3, 4).astype("float32")

    def test_view_as_real_shape(self):
        def body(dev):
            c = torch.complex(torch.tensor(self.re), torch.tensor(self.im))
            v = torch.view_as_real(c)            # -> (..., 2): [real, imag]
            self.assertEqual(tuple(v.shape), (3, 4, 2), f"view_as_real shape {dev}")
            self.ac(v.numpy()[..., 0], self.re, msg=f"view_as_real [...,0] {dev}")
            self.ac(v.numpy()[..., 1], self.im, msg=f"view_as_real [...,1] {dev}")
        both_devices(body)

    def test_view_as_complex_roundtrip(self):
        stacked = np.stack([self.re, self.im], axis=-1)   # (3,4,2)
        def body(dev):
            c = torch.view_as_complex(torch.tensor(stacked))
            self.ac(c.real.numpy(), self.re, msg=f"view_as_complex real {dev}")
            self.ac(c.imag.numpy(), self.im, msg=f"view_as_complex imag {dev}")
            # round-trip view_as_real(view_as_complex(x)) == x
            self.ac(torch.view_as_real(c).numpy(), stacked,
                    msg=f"view_as_real(view_as_complex) {dev}")
        both_devices(body)


class TestPolar(Base):
    def test_polar(self):
        rs = np.random.RandomState(10)
        mag = np.abs(rs.randn(3, 4)).astype("float32") + 0.1
        ang = rs.uniform(-np.pi, np.pi, (3, 4)).astype("float32")
        ref = mag * np.exp(1j * ang)
        def body(dev):
            p = torch.polar(torch.tensor(mag), torch.tensor(ang))
            self.acplx(p, ref, atol=1e-4, msg=f"polar {dev}")
        both_devices(body)

    def test_polar_abs_angle_roundtrip(self):
        # |polar(r, theta)| == r, angle(polar(r, theta)) == theta (theta in (-pi,pi))
        rs = np.random.RandomState(11)
        mag = np.abs(rs.randn(2, 5)).astype("float32") + 0.5
        ang = rs.uniform(-np.pi + 0.1, np.pi - 0.1, (2, 5)).astype("float32")
        def body(dev):
            p = torch.polar(torch.tensor(mag), torch.tensor(ang))
            self.ac(p.abs().numpy(), mag, atol=1e-4, msg=f"|polar|==r {dev}")
            self.ac(p.angle().numpy(), ang, atol=3e-3, msg=f"angle(polar)==theta {dev}")
        both_devices(body)


# ------------------------------------------------------------------------- einsum

class TestEinsum(Base):
    def setUp(self):
        self.rs = np.random.RandomState(12)

    def _chk(self, eq, *arrs, atol=1e-4, msg=""):
        """Compare torch.einsum to numpy.einsum. Scalar outputs (jittor shape [1] vs
        numpy ()) are compared via .item()."""
        def body(dev):
            ts = [torch.tensor(a) for a in arrs]
            got = torch.einsum(eq, *ts)
            ref = np.einsum(eq, *arrs)
            if np.ndim(ref) == 0:
                # jittor has no 0-d scalar: result is shape [1]; compare scalar value.
                self.assertEqual(tuple(np.asarray(got.numpy()).shape), (1,),
                                 f"scalar einsum shape {eq} {dev}")
                np.testing.assert_allclose(float(got.item()), float(ref), atol=atol,
                                           err_msg=f"{eq} {dev} {msg}")
            else:
                self.ac(got.numpy(), ref, atol=atol, msg=f"{eq} {dev} {msg}")
        both_devices(body)

    def test_matmul(self):
        A = self.rs.randn(3, 4).astype("float32"); B = self.rs.randn(4, 5).astype("float32")
        self._chk("ij,jk->ik", A, B)

    def test_batch_matmul(self):
        A = self.rs.randn(2, 3, 4).astype("float32"); B = self.rs.randn(2, 4, 5).astype("float32")
        self._chk("bij,bjk->bik", A, B)

    def test_trace(self):
        S = self.rs.randn(5, 5).astype("float32")
        self._chk("ii->", S)            # scalar trace

    def test_diagonal(self):
        S = self.rs.randn(5, 5).astype("float32")
        self._chk("ii->i", S)

    def test_batched_diagonal(self):
        S = self.rs.randn(3, 4, 4).astype("float32")
        self._chk("bii->b", S)

    def test_transpose(self):
        A = self.rs.randn(3, 4).astype("float32")
        self._chk("ij->ji", A)

    def test_outer_product(self):
        u = self.rs.randn(3).astype("float32"); v = self.rs.randn(4).astype("float32")
        self._chk("i,j->ij", u, v)

    def test_inner_product(self):
        u = self.rs.randn(6).astype("float32"); v = self.rs.randn(6).astype("float32")
        self._chk("i,i->", u, v)        # scalar dot

    def test_sum_all(self):
        A = self.rs.randn(3, 4).astype("float32")
        self._chk("ij->", A)            # scalar full reduction

    def test_sum_over_axis(self):
        A = self.rs.randn(3, 4).astype("float32")
        self._chk("ij->j", A)
        self._chk("ij->i", A)

    def test_elementwise_then_sum(self):
        A = self.rs.randn(3, 4).astype("float32"); B = self.rs.randn(3, 4).astype("float32")
        self._chk("ij,ij->", A, B)      # Frobenius inner product (scalar)

    def test_bilinear(self):
        # x^T A y contraction, a real pattern (attention-ish).
        x = self.rs.randn(2, 3).astype("float32")
        A = self.rs.randn(3, 4).astype("float32")
        y = self.rs.randn(2, 4).astype("float32")
        self._chk("bi,ij,bj->b", x, A, y)

    def test_packed_operands_form(self):
        # torch.einsum also accepts operands packed in a single tuple/list.
        A = self.rs.randn(3, 4).astype("float32"); B = self.rs.randn(4, 5).astype("float32")
        def body(dev):
            got = torch.einsum("ij,jk->ik", (torch.tensor(A), torch.tensor(B)))
            self.ac(got.numpy(), np.einsum("ij,jk->ik", A, B), atol=1e-4,
                    msg=f"packed einsum {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
