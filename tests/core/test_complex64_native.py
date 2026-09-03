"""Native complex64 dtype (Phase 2) — create / numpy round-trip / arithmetic vs numpy.

Locks the native jittor_core complex64 dtype: registered (dsize 8, is_complex), creatable
from a numpy complex64 array, and elementwise add/sub/mul/div/neg matching numpy. CPU+CUDA.

Run:  python -m pytest tests/core/test_complex64_native.py
"""
import unittest
import numpy as np
import jittor as jt
from jittor.nn.functional.complex import _real2_to_complex64

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

    def test_python_complex_scalar_setitem(self):
        def body(dev):
            z = jt.zeros((2, 2), "complex64")
            z[0, 1] = 1 + 2j
            z[1, 0] = np.complex64(-3 + 4j)
            expected = np.array([[0, 1 + 2j], [-3 + 4j, 0]], dtype="complex64")
            np.testing.assert_array_equal(np.asarray(z.numpy()), expected,
                                          err_msg=f"complex scalar setitem {dev}")
        both_devices(body)

    def test_real_complex_cast_and_backward(self):
        def body(dev):
            z = jt.array(np.array([1 + 2j, -3 + 4j], dtype="complex64"))
            np.testing.assert_array_equal(np.asarray(z.cast("float32").numpy()),
                                          np.array([1, -3], dtype="float32"),
                                          err_msg=f"complex-to-real cast {dev}")
            zb = jt.array(np.array([0 + 1j, 0 + 0j, 2 + 0j], dtype="complex64"))
            np.testing.assert_array_equal(np.asarray(zb.cast("bool").numpy()),
                                          np.array([True, False, True]),
                                          err_msg=f"complex-to-bool cast {dev}")

            seed = jt.array(np.array([2.0, -0.5], dtype="float32"))
            z_for_grad = jt.array(np.array([1 + 2j, -3 + 4j], dtype="complex64"))
            z_grad = jt.grad((z_for_grad.cast("float32") * seed).sum(), z_for_grad)
            if isinstance(z_grad, (list, tuple)):
                z_grad = z_grad[0]
            np.testing.assert_array_equal(np.asarray(z_grad.numpy()),
                                          np.array([2 + 0j, -0.5 + 0j], dtype="complex64"),
                                          err_msg=f"complex-to-real cast backward {dev}")

            x = jt.array(np.array([0.5, -1.25], dtype="float32"))
            x.start_grad()
            loss = x.cast("complex64").real.sum()
            grad = jt.grad(loss, x)
            if isinstance(grad, (list, tuple)):
                grad = grad[0]
            self.assertEqual(str(grad.dtype), "float32", f"cast grad dtype {dev}")
            np.testing.assert_array_equal(np.asarray(grad.numpy()),
                                          np.ones(2, dtype="float32"),
                                          err_msg=f"real-to-complex cast backward {dev}")
        both_devices(body)

    def test_python_complex_scalar_binary(self):
        x_np = np.array([0.5, -1.25], dtype="float32")
        def body(dev):
            x = jt.array(x_np)
            for name, got, expected in (
                ("rmul", 1j * x, 1j * x_np),
                ("add", x + np.complex64(2 - 3j), x_np + np.complex64(2 - 3j)),
                ("rdiv", (1 + 2j) / x, (1 + 2j) / x_np),
            ):
                self.assertEqual(str(got.dtype), "complex64", f"{name} dtype {dev}")
                np.testing.assert_allclose(np.asarray(got.numpy()), expected,
                                           atol=1e-6, rtol=1e-6,
                                           err_msg=f"complex scalar {name} {dev}")
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

    def test_grad(self):
        # Native complex64 autograd, verified vs real torch 2.12 (the backward formulas
        # below are torch's convention for a real loss L = sum(|y|), y = op(a, b)).
        rng = np.random.RandomState(11)
        a = (rng.randn(4) + 1j * rng.randn(4)).astype("complex64")
        b = (rng.randn(4) + 1j * rng.randn(4)).astype("complex64")

        def expected(name):
            y = {"add": a + b, "mul": a * b, "div": a / b,
                 "neg": -a, "conj": np.conj(a)}[name]
            gy = y / np.abs(y)                       # d|y| backward, dout = 1
            if name == "add":  return gy, gy
            if name == "mul":  return gy * np.conj(b), gy * np.conj(a)
            if name == "div":  return gy / np.conj(b), -gy * np.conj(a) / np.conj(b) ** 2
            if name == "neg":  return -gy, None
            if name == "conj": return np.conj(gy), None

        def body(dev):
            for name in ["add", "mul", "div", "neg", "conj"]:
                ja, jb = jt.array(a), jt.array(b)
                y = {"add": ja + jb, "mul": ja * jb, "div": ja / jb,
                     "neg": -ja, "conj": ja.conj()}[name]
                L = y.abs().sum()
                ga = jt.grad(L, ja)
                ga = ga[0] if isinstance(ga, (list, tuple)) else ga
                self.assertEqual(str(ga.dtype), "complex64", f"{name} grad dtype {dev}")
                ea, eb = expected(name)
                np.testing.assert_allclose(np.asarray(ga.numpy()), ea, atol=1e-3, rtol=1e-3,
                                           err_msg=f"{name} a-grad {dev}")
                if eb is not None:
                    gb = jt.grad(L, jb)
                    gb = gb[0] if isinstance(gb, (list, tuple)) else gb
                    np.testing.assert_allclose(np.asarray(gb.numpy()), eb, atol=1e-3, rtol=1e-3,
                                               err_msg=f"{name} b-grad {dev}")
        both_devices(body)

    def test_grad_transcendental(self):
        # holomorphic unary grads: grad = (y/|y|) * conj(f'(a)), verified vs real torch 2.12.
        a = np.array([0.5 + 0.3j, -0.4 + 0.8j, 1.0 - 0.2j], dtype="complex64")
        def expected(name):
            y = {"exp": np.exp(a), "log": np.log(a), "sin": np.sin(a),
                 "cos": np.cos(a), "sqrt": np.sqrt(a)}[name]
            fp = {"exp": np.exp(a), "log": 1 / a, "sin": np.cos(a),
                  "cos": -np.sin(a), "sqrt": 1 / (2 * np.sqrt(a))}[name]
            return (y / np.abs(y)) * np.conj(fp)
        def body(dev):
            for name, jf in [("exp", jt.exp), ("log", jt.log), ("sin", jt.sin),
                             ("cos", jt.cos), ("sqrt", jt.sqrt)]:
                ja = jt.array(a)
                L = jf(ja).abs().sum()
                ga = jt.grad(L, ja)
                ga = ga[0] if isinstance(ga, (list, tuple)) else ga
                np.testing.assert_allclose(np.asarray(ga.numpy()), expected(name),
                                           atol=1e-3, rtol=1e-3, err_msg=f"{name} grad {dev}")
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

    def test_view_bridge(self):
        # Phase 6 keystone: native complex64 <-> float32[...,2] bridge, differentiable, both
        # cards (see docs/architecture/complex-dtype.md). view_as_real lowers complex64 to its [real,imag]
        # pair; _real2_to_complex64 rebuilds it; the pair is autograd-transparent.
        rng = np.random.RandomState(5)
        a = (rng.randn(3, 4) + 1j * rng.randn(3, 4)).astype("complex64")
        def body(dev):
            z = jt.array(a)
            # view_as_real: complex64 -> float32[...,2] == [real, imag]
            vr = jt.nn.view_as_real(z)
            self.assertEqual(str(vr.dtype), "float32", f"view_as_real dtype {dev}")
            self.assertEqual(tuple(vr.shape), (3, 4, 2), f"view_as_real shape {dev}")
            np.testing.assert_allclose(np.asarray(vr.numpy()),
                                       np.stack([a.real, a.imag], axis=-1), atol=1e-6,
                                       err_msg=f"view_as_real {dev}")
            # reverse round-trip is exact (bit-identical reinterpret)
            zc = _real2_to_complex64(vr)
            self.assertEqual(str(zc.dtype), "complex64", f"reverse dtype {dev}")
            np.testing.assert_array_equal(np.asarray(zc.numpy()), a,
                                          err_msg=f"bridge roundtrip {dev}")
            # bridge is autograd-transparent: grad through it == direct grad on |z|.sum()
            zg = jt.array(a)
            L = _real2_to_complex64(jt.nn.view_as_real(zg)).abs().sum()
            g = jt.grad(L, zg); g = g[0] if isinstance(g, (list, tuple)) else g
            zg2 = jt.array(a); L2 = zg2.abs().sum()
            g2 = jt.grad(L2, zg2); g2 = g2[0] if isinstance(g2, (list, tuple)) else g2
            np.testing.assert_allclose(np.asarray(g.numpy()), np.asarray(g2.numpy()),
                                       atol=1e-5, err_msg=f"bridge autograd {dev}")
            # view_as_real stays polymorphic over the legacy ComplexNumber (real/imag pair)
            cn = jt.nn.ComplexNumber(jt.array(a.real.copy()), jt.array(a.imag.copy()))
            np.testing.assert_allclose(np.asarray(jt.nn.view_as_real(cn).numpy()),
                                       np.stack([a.real, a.imag], axis=-1), atol=1e-6,
                                       err_msg=f"view_as_real(ComplexNumber) {dev}")
        both_devices(body)

    def test_accessors(self):
        # Phase 6 P2: native complex64 .real/.imag/.angle (torch parity), and view_as_complex
        # / polar now return native complex64. Real-dtype Vars: real->self, imag->zeros.
        rng = np.random.RandomState(6)
        re = rng.randn(3, 4).astype("float32")
        im = rng.randn(3, 4).astype("float32")
        a = (re + 1j * im).astype("complex64")
        mag = (np.abs(rng.randn(3, 4)) + 0.1).astype("float32")
        ang = rng.uniform(-np.pi, np.pi, (3, 4)).astype("float32")
        def body(dev):
            z = jt.array(a)
            np.testing.assert_allclose(np.asarray(z.real.numpy()), re, atol=1e-6, err_msg=f"real {dev}")
            np.testing.assert_allclose(np.asarray(z.imag.numpy()), im, atol=1e-6, err_msg=f"imag {dev}")
            np.testing.assert_allclose(np.asarray(z.angle().numpy()), np.angle(a), atol=3e-3,
                                       err_msg=f"angle {dev}")
            # view_as_complex now returns native complex64
            c = jt.nn.view_as_complex(jt.array(np.stack([re, im], axis=-1)))
            self.assertEqual(str(c.dtype), "complex64", f"view_as_complex dtype {dev}")
            np.testing.assert_array_equal(np.asarray(c.numpy()), a, err_msg=f"view_as_complex {dev}")
            # polar -> native complex64
            p = jt.nn.polar(jt.array(mag), jt.array(ang))
            self.assertEqual(str(p.dtype), "complex64", f"polar dtype {dev}")
            np.testing.assert_allclose(np.asarray(p.numpy()), mag * np.exp(1j * ang), atol=1e-4,
                                       err_msg=f"polar {dev}")
            # real-dtype parity: real(x)=x, imag(x)=0
            x = jt.array(re)
            np.testing.assert_array_equal(np.asarray(x.real.numpy()), re, err_msg=f"real-of-real {dev}")
            np.testing.assert_array_equal(np.asarray(x.imag.numpy()), np.zeros_like(re),
                                          err_msg=f"imag-of-real {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
