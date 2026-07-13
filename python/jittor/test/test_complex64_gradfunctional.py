"""Native complex64 through gradfunctional (Phase 6 P5).

Locks that jittor.gradfunctional.{vjp, jvp} accept and correctly differentiate the NEW
native complex64 dtype (a first-class differentiable Var), not just the legacy
jt.nn.ComplexNumber real/imag-pair simulation.

What is checked (CPU + CUDA):
  - vjp on a complex->complex function with a native complex64 input and a complex
    grad-output v: result matches a numpy finite-difference oracle of the SAME real
    seeded loss the implementation uses, L = Re(sum(out * conj(v))) over the (real,imag)
    representation. (This is torch's conjugate/Wirtinger input-grad convention.)
  - vjp result is numerically identical to the legacy ComplexNumber path (polymorphism).
  - jvp on native complex64 raises NotImplementedError cleanly (native complex64 has no
    second-order autograd yet, and jvp is the double-backward trick) -- "宁可响亮崩也不静默错".
  - jvp on the legacy ComplexNumber path STILL works (regression guard).
  - jvp on native REAL Vars still works (the guard only trips on native complex).

No real torch in this env: numpy is the oracle.

Scope notes (limitations of native complex64 itself, NOT of gradfunctional):
  - jvp can't run on native complex64 because there is no second-order autograd for the
    dtype yet (the double-backward needs a complex64->float32 cast-backward). gradfunctional
    detects this and raises NotImplementedError instead of an opaque C++ compile error.
  - vjp with a *real* input feeding a complex output (so the input grad must be cast from
    complex64 back to float32) hits the same missing cast-backward in the op layer; that is
    outside gradfunctional's control, so it is intentionally not exercised here. All-complex
    and all-real input tuples work; those are what we lock.

Run:  python -m jittor.test.test_complex64_gradfunctional
"""
import unittest
import numpy as np
import jittor as jt
from jittor.gradfunctional import vjp, jvp
from jittor.nn import ComplexNumber

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _to_complex(g):
    # numpy complex array from a native complex64 Var.
    return np.asarray(g.numpy())


def _cn_to_complex(cn):
    # numpy complex array from a legacy ComplexNumber.
    v = np.asarray(cn.value.numpy())
    return v[..., 0] + 1j * v[..., 1]


def _np_complex(rng, shape):
    return (rng.randn(*shape) + 1j * rng.randn(*shape)).astype("complex64")


def _fd_vjp(fnp, a, v, eps=1e-3):
    """Finite-difference of the real seeded loss L(z) = sum(Re f(z)*Re v + Im f(z)*Im v),
    w.r.t. real and imag parts of a, returned as a complex grad (the vjp value)."""
    a = a.astype("complex64")
    zr0 = a.real.astype(np.float64)
    zi0 = a.imag.astype(np.float64)

    def loss(zr, zi):
        fo = fnp((zr + 1j * zi).astype("complex64"))
        return float(np.sum(fo.real * v.real + fo.imag * v.imag))

    gr = np.zeros(a.shape, dtype=np.float64)
    gi = np.zeros(a.shape, dtype=np.float64)
    it = np.ndindex(*a.shape)
    for idx in it:
        zr = zr0.copy(); zr[idx] += eps
        zrm = zr0.copy(); zrm[idx] -= eps
        gr[idx] = (loss(zr, zi0) - loss(zrm, zi0)) / (2 * eps)
        zi = zi0.copy(); zi[idx] += eps
        zim = zi0.copy(); zim[idx] -= eps
        gi[idx] = (loss(zr0, zi) - loss(zr0, zim)) / (2 * eps)
    return gr + 1j * gi


class TestComplex64GradFunctional(unittest.TestCase):
    # -------------------------------------------------------------- vjp: exp().sum(1)
    def test_vjp_exp_complex_input(self):
        rng = np.random.RandomState(0)
        s = (5, 6)
        a = _np_complex(rng, s)
        v = _np_complex(rng, (5,))          # matches the (5,) output of exp().sum(1)

        def f(x):
            return x.exp().sum(1)

        def fnp(z):
            return np.exp(z).sum(1)

        fd = _fd_vjp(fnp, a, v)

        def body(dev):
            out, g = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            self.assertEqual(str(out.dtype), "complex64", f"out dtype {dev}")
            self.assertEqual(str(g.dtype), "complex64", f"vjp dtype {dev}")
            self.assertEqual(tuple(g.shape), s, f"vjp shape {dev}")
            gnp = _to_complex(g)
            self.assertTrue(np.isfinite(gnp).all(), f"vjp finite {dev}")
            np.testing.assert_allclose(gnp, fd, atol=2e-3, rtol=2e-3,
                                       err_msg=f"vjp vs finite-diff {dev}")
        both_devices(body)

    # -------------------------------------------------------- vjp: closed form (z*z)
    def test_vjp_square_closed_form(self):
        # f(z) = z*z (elementwise, holomorphic). For the real seeded loss
        # L = Re(sum(f * conj(v))) the input grad is conj(df/dz)^T applied to v:
        # since df/dz = 2z and the map is diagonal, grad = conj(2z) * v ... but the
        # implementation differentiates the REAL loss, whose grad equals 2*conj(z)*v
        # (Wirtinger conj convention). We assert against that closed form AND the
        # finite-diff oracle agrees.
        rng = np.random.RandomState(4)
        s = (4, 3)
        a = _np_complex(rng, s)
        v = _np_complex(rng, s)
        closed = np.conj(2 * a) * v        # == 2*conj(a)*v

        def f(x):
            return x * x

        def fnp(z):
            return z * z

        fd = _fd_vjp(fnp, a, v)
        np.testing.assert_allclose(fd, closed, atol=2e-3, rtol=2e-3,
                                   err_msg="finite-diff vs closed form (sanity)")

        def body(dev):
            out, g = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            gnp = _to_complex(g)
            np.testing.assert_allclose(gnp, closed, atol=2e-3, rtol=2e-3,
                                       err_msg=f"vjp(z*z) vs closed form {dev}")
        both_devices(body)

    # ------------------------------------- vjp: native complex64 == legacy ComplexNumber
    def test_vjp_native_matches_complexnumber(self):
        rng = np.random.RandomState(1)
        s = (5, 6)
        a = _np_complex(rng, s)
        v = _np_complex(rng, (5,))

        def f(x):
            return x.exp().sum(1)

        def body(dev):
            # native complex64
            _, g_native = vjp(f, jt.array(a), jt.array(v), create_graph=True)
            gn = _to_complex(g_native)
            # legacy ComplexNumber (real/imag stacked value)
            cn_a = ComplexNumber(jt.array(np.stack([a.real, a.imag], -1)),
                                 is_concat_value=True)
            cn_v = ComplexNumber(jt.array(np.stack([v.real, v.imag], -1)),
                                 is_concat_value=True)
            _, g_cn = vjp(f, cn_a, cn_v, create_graph=True)
            gc = _cn_to_complex(g_cn)
            np.testing.assert_allclose(gn, gc, atol=1e-4, rtol=1e-4,
                                       err_msg=f"native vjp != ComplexNumber vjp {dev}")
        both_devices(body)

    # ------------------------------------------ vjp: tuple of two native complex inputs
    def test_vjp_two_complex_inputs(self):
        # adder(x, y) = w1*x + w2*y, both native complex64 -> complex output, two grads.
        rng = np.random.RandomState(2)
        s = (4, 5)
        x = _np_complex(rng, s)
        y = _np_complex(rng, s)
        v = _np_complex(rng, s)
        w1 = np.array(0.7 + 0.4j, dtype="complex64")
        w2 = np.array(-0.3 + 0.9j, dtype="complex64")

        def f(a, b):
            return jt.array(w1) * a + jt.array(w2) * b

        def body(dev):
            out, (gx, gy) = vjp(f, (jt.array(x), jt.array(y)), jt.array(v),
                                create_graph=True)
            self.assertEqual(str(out.dtype), "complex64", f"out dtype {dev}")
            self.assertEqual(str(gx.dtype), "complex64", f"gx dtype {dev}")
            self.assertEqual(str(gy.dtype), "complex64", f"gy dtype {dev}")
            gxnp, gynp = _to_complex(gx), _to_complex(gy)
            # d/da of Re(sum((w1 a + w2 b) conj(v))) = conj(w1) * v  (Wirtinger conj conv.)
            np.testing.assert_allclose(gxnp, np.conj(w1) * v, atol=2e-3, rtol=2e-3,
                                       err_msg=f"two-complex grad x {dev}")
            np.testing.assert_allclose(gynp, np.conj(w2) * v, atol=2e-3, rtol=2e-3,
                                       err_msg=f"two-complex grad y {dev}")
        both_devices(body)

    # --------------------------------------------------- jvp: native complex -> raises
    def test_jvp_native_complex_raises(self):
        rng = np.random.RandomState(3)
        s = (5, 6)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)        # jvp v matches input shape

        def f(x):
            return x.exp().sum(1)

        def body(dev):
            with self.assertRaises(NotImplementedError):
                jvp(f, jt.array(a), jt.array(vin), create_graph=True)
            # also when complex appears only in the OUTPUT (real input -> complex out):
            # real Var * complex64 Var promotes to complex64.
            cone = jt.array(np.array(1.0 + 0.0j, dtype="complex64"))
            def g(x):
                return (x * cone).sum(1)
            xr = rng.randn(*s).astype("float32")
            vr = rng.randn(*s).astype("float32")
            with self.assertRaises(NotImplementedError):
                jvp(g, jt.array(xr), jt.array(vr), create_graph=True)
        both_devices(body)

    # ----------------------------------------------- jvp: native REAL still works
    def test_jvp_native_real_ok(self):
        rng = np.random.RandomState(5)
        s = (5, 6)
        x = rng.randn(*s).astype("float32")
        vin = rng.randn(*s).astype("float32")

        def f(a):
            return (a * a).sum(1)       # out (5,)

        # jvp = J @ vin ; for f=sum_j a_ij^2, (J v)_i = sum_j 2 a_ij v_ij
        ref = (2 * x * vin).sum(1)

        def body(dev):
            out, j = jvp(f, jt.array(x), jt.array(vin), create_graph=True)
            jnp = np.asarray(j.numpy())
            self.assertTrue(np.isfinite(jnp).all(), f"real jvp finite {dev}")
            np.testing.assert_allclose(jnp, ref, atol=2e-3, rtol=2e-3,
                                       err_msg=f"real jvp vs closed form {dev}")
        both_devices(body)

    # --------------------------------------------- jvp: legacy ComplexNumber still works
    def test_jvp_complexnumber_still_works(self):
        rng = np.random.RandomState(6)
        s = (5, 6)
        a = _np_complex(rng, s)
        vin = _np_complex(rng, s)

        def f(x):
            return x.exp().sum(1)

        def body(dev):
            cn_a = ComplexNumber(jt.array(np.stack([a.real, a.imag], -1)),
                                 is_concat_value=True)
            cn_v = ComplexNumber(jt.array(np.stack([vin.real, vin.imag], -1)),
                                 is_concat_value=True)
            out, j = jvp(f, cn_a, cn_v, create_graph=True)
            self.assertIsInstance(j, ComplexNumber, f"jvp returns ComplexNumber {dev}")
            jnp = _cn_to_complex(j)
            self.assertTrue(np.isfinite(jnp).all(), f"CN jvp finite {dev}")
            self.assertEqual(jnp.shape, (5,), f"CN jvp shape {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
