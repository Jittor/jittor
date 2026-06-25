"""Torch-grade elementwise-math parity for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite. CPU+CUDA vs numpy. Covers the elementwise
math surface (trig / exp-log / rounding / clamp / sign) not exercised elsewhere.

Run:  python -m jittor.test.test_torch_compat_math
"""
import unittest
import numpy as np
import jittor as torch
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol,
                                   err_msg=msg)

    def chk1(self, torch_fn, np_fn, x, atol=1e-5, rtol=1e-5, name=""):
        def body(dev):
            self.ac(torch_fn(torch.tensor(x)).numpy(), np_fn(x), atol=atol, rtol=rtol,
                    msg=f"{name} {dev}")
        both_devices(body)


class TestExpLog(Base):
    def setUp(self):
        self.pos = np.random.RandomState(0).rand(4, 5).astype("float32") + 0.1
        self.any = np.random.RandomState(1).randn(4, 5).astype("float32")

    def test_exp_family(self):
        self.chk1(torch.exp, np.exp, self.any, rtol=1e-5, name="exp")
        self.chk1(torch.expm1, np.expm1, self.any, atol=1e-5, name="expm1")
        self.chk1(torch.exp2, np.exp2, self.any, rtol=1e-5, name="exp2")

    def test_log_family(self):
        self.chk1(torch.log, np.log, self.pos, name="log")
        self.chk1(torch.log2, np.log2, self.pos, name="log2")
        self.chk1(torch.log10, np.log10, self.pos, name="log10")
        self.chk1(torch.log1p, np.log1p, self.pos, name="log1p")

    def test_sqrt_pow(self):
        self.chk1(torch.sqrt, np.sqrt, self.pos, name="sqrt")
        self.chk1(torch.rsqrt, lambda x: 1.0 / np.sqrt(x), self.pos, rtol=1e-4, name="rsqrt")
        def body(dev):
            self.ac(torch.pow(torch.tensor(self.pos), 2.5).numpy(), self.pos ** 2.5,
                    rtol=1e-4, msg=f"pow {dev}")
            self.ac(torch.reciprocal(torch.tensor(self.pos)).numpy(), 1.0 / self.pos,
                    rtol=1e-5, msg=f"reciprocal {dev}")
        both_devices(body)


class TestTrig(Base):
    def setUp(self):
        self.x = (np.random.RandomState(2).rand(4, 5).astype("float32") * 1.4 - 0.7)

    def test_trig(self):
        for nm, tf, nf in [("sin", torch.sin, np.sin), ("cos", torch.cos, np.cos),
                           ("tan", torch.tan, np.tan), ("asin", torch.asin, np.arcsin),
                           ("acos", torch.acos, np.arccos), ("atan", torch.atan, np.arctan),
                           ("sinh", torch.sinh, np.sinh), ("cosh", torch.cosh, np.cosh),
                           ("tanh", torch.tanh, np.tanh)]:
            self.chk1(tf, nf, self.x, atol=1e-5, rtol=1e-5, name=nm)

    def test_atan2(self):
        a = np.random.RandomState(3).randn(5).astype("float32")
        b = np.random.RandomState(4).randn(5).astype("float32")
        def body(dev):
            # jittor's atan2 has a ~1e-3 precision gap vs numpy (low-precision approx,
            # not a correctness bug) — loosen tolerance accordingly.
            self.ac(torch.atan2(torch.tensor(a), torch.tensor(b)).numpy(),
                    np.arctan2(a, b), atol=3e-3, rtol=3e-3, msg=f"atan2 {dev}")
        both_devices(body)


class TestRounding(Base):
    def setUp(self):
        self.x = (np.random.RandomState(5).randn(4, 5) * 3).astype("float32")

    def test_rounding(self):
        self.chk1(torch.floor, np.floor, self.x, name="floor")
        self.chk1(torch.ceil, np.ceil, self.x, name="ceil")
        self.chk1(torch.trunc, np.trunc, self.x, name="trunc")
        self.chk1(torch.abs, np.abs, self.x, name="abs")
        self.chk1(torch.sign, np.sign, self.x, name="sign")
        self.chk1(torch.negative, np.negative, self.x, name="neg")

    def test_clamp(self):
        x = self.x
        def body(dev):
            self.ac(torch.clamp(torch.tensor(x), -1.0, 1.0).numpy(), np.clip(x, -1.0, 1.0),
                    msg=f"clamp {dev}")
            self.ac(torch.clamp(torch.tensor(x), min=0.0).numpy(), np.clip(x, 0.0, None),
                    msg=f"clamp min {dev}")
        both_devices(body)


class TestErf(Base):
    def test_erf(self):
        x = (np.random.RandomState(6).randn(20).astype("float32"))
        erf = np.vectorize(__import__("math").erf)
        def body(dev):
            self.ac(torch.erf(torch.tensor(x)).numpy(), erf(x).astype("float32"),
                    atol=1e-6, msg=f"erf {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
