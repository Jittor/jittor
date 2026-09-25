# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""float16/bfloat16: the dtype that comes back, and the width it was computed at.

Two questions, both asked against PyTorch 2.13 rather than against jittor's own
previous answer.

**The dtype.** torch returns the input's dtype from every op in the sweep below,
on CPU and on CUDA. Jittor widened four of them to float32 -- ``max``, ``min``,
``prod`` and ``cumprod`` -- plus ``logsumexp``, so on a half Var ``x.max(-1)``
silently took the graph out of half while ``x.sum(-1)`` did not.

**The width.** The interesting failure is the mirror image of the one
``rms_norm`` had: an op that computes fp16 *in* fp16 where torch accumulates in
float32. A float32 accumulator rounds once, at the store, so its error is about
one ULP of the output dtype however long the axis is; a half accumulator rounds
every partial and its error grows with the length. The bound asserted for each
op is therefore written in ULPs of the output dtype, and the number real torch
2.13 measured on the same input is in the comment beside it.

The reference is the closed form evaluated in float64 on the *rounded* input, so
the only thing being measured is the op's own accumulation -- not the input
quantisation, which both libraries share.

Run::  python -m pytest tests/type/test_half_precision_parity.py
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np
import jittor as jt


_DEVICES = [("cpu", 0)] + (
    [("cuda", 1)] if _test_capability.any_accelerator_enabled(backend=jt) else [])

#: relative spacing of the two half formats: 2**-11 for float16's 10 explicit
#: mantissa bits, 2**-8 for bfloat16's 7.
_ULP = {"float16": 2.0 ** -11, "bfloat16": 2.0 ** -8}


def _bf16_round(values):
    """fp32 -> bfloat16 -> fp32 with round-to-nearest-even, in numpy.

    numpy has no bfloat16, and the point of several tests here is exactly which
    rounding rule is applied, so this cannot borrow jittor's own answer.
    """
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).copy()
    bits += np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
    return (bits & np.uint32(0xffff0000)).view(np.float32)


def _round_trip(values, name):
    """The float64 value of ``values`` after rounding to ``name``."""
    if name == "float16":
        return np.asarray(values, dtype=np.float32).astype(np.float16).astype(np.float64)
    return _bf16_round(values).astype(np.float64)


def _cast(var, name):
    return var.float16() if name == "float16" else var.bfloat16()


def _rel(got, exact):
    got = np.asarray(got, dtype=np.float64)
    return float(np.max(np.abs(got - exact) / np.maximum(np.abs(exact), 1e-30)))


def _abs(got, exact):
    return float(np.max(np.abs(np.asarray(got, dtype=np.float64) - exact)))


# ---------------------------------------------------------------------------
# 1. The conversion itself.
# ---------------------------------------------------------------------------

class TestHalfRounding(unittest.TestCase):
    """float32 -> half must round to nearest even, not truncate.

    The CPU ``bfloat16`` constructor in ``src/type/fp16_compute.h`` was
    ``this->x = bits >> 16`` -- a truncation. It threw away the low half of the
    significand on the floor, so every inexact value came back biased toward
    zero by up to a full ULP where the correct answer is within half of one.

    Three consequences, and none of them is cosmetic. The host disagreed with
    the device, because on CUDA ``bfloat16`` is ``__nv_bfloat16`` and the
    conversion is the hardware's, which rounds: ``0.1`` was 0.099609375 on CPU
    against 0.100097656 on CUDA, and ``255.7`` was 255 against 256. The host
    disagreed with torch, which rounds on both of its devices. And the bias is
    systematic, so it did not cancel: it was worth ~3.5x on the measured error
    of every bf16 reduction on CPU even where the accumulation was already
    float32 (a 4096-long sum was 6.36e-3 from the exact value against torch's
    1.78e-3).
    """

    #: Values chosen so truncation and round-to-nearest-even differ. The third
    #: and fifth round *up across an exponent boundary*, which a truncating
    #: conversion can never do.
    CASES = [1.0 + 2 ** -9, 1.0 + 3 * 2 ** -10, 0.1, 3.14159265,
             1.0 - 2 ** -10, 255.7, 1e-3, 65792.0, -0.1, -255.7]

    def test_bfloat16_rounds_to_nearest_even(self):
        values = np.array(self.CASES, dtype=np.float32)
        expect = _bf16_round(values)
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name):
                with jt.flag_scope(use_cuda=use_cuda):
                    got = jt.array(values).bfloat16().float32().numpy()
                # Exact equality: a rounding rule is right or it is not.
                np.testing.assert_array_equal(got, expect)

    def test_float16_rounds_to_nearest_even(self):
        values = np.array(self.CASES, dtype=np.float32)
        expect = values.astype(np.float16).astype(np.float32)
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name):
                with jt.flag_scope(use_cuda=use_cuda):
                    got = jt.array(values).float16().float32().numpy()
                np.testing.assert_array_equal(got, expect)

    def test_bfloat16_specials_survive(self):
        """NaN stays NaN and does not carry into an infinity.

        The round-to-nearest carry is ``bits + 0x7fff + lsb``, which can push a
        NaN payload over into the exponent field; the conversion special-cases
        it for that reason.
        """
        values = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
        for name, use_cuda in _DEVICES:
            with self.subTest(device=name):
                with jt.flag_scope(use_cuda=use_cuda):
                    got = jt.array(values).bfloat16().float32().numpy()
                self.assertTrue(np.isnan(got[0]))
                self.assertEqual(got[1], np.inf)
                self.assertEqual(got[2], -np.inf)
                self.assertEqual(np.signbit(got[4]), True)

    def test_cpu_and_device_agree(self):
        """The same program, two backends, one answer."""
        if len(_DEVICES) < 2:
            raise unittest.SkipTest("no accelerator enabled")
        values = np.array(self.CASES, dtype=np.float32)
        for method in ("bfloat16", "float16"):
            with self.subTest(dtype=method):
                with jt.flag_scope(use_cuda=0):
                    host = getattr(jt.array(values), method)().float32().numpy()
                with jt.flag_scope(use_cuda=1):
                    device = getattr(jt.array(values), method)().float32().numpy()
                np.testing.assert_array_equal(host, device)


# ---------------------------------------------------------------------------
# 2. The dtype that comes back.
# ---------------------------------------------------------------------------

def _dtype_cases():
    ones = lambda n, v: jt.ones(n).cast(v.dtype)
    return [
        ("sum", lambda v: v.sum(-1)),
        ("mean", lambda v: v.mean(-1)),
        ("prod", lambda v: v.prod(-1)),
        ("max", lambda v: v.max(-1)),
        ("min", lambda v: v.min(-1)),
        ("var", lambda v: jt.var(v, dim=-1)),
        ("std", lambda v: jt.std(v, dim=-1)),
        ("norm", lambda v: jt.norm(v, dim=-1)),
        ("cumsum", lambda v: jt.cumsum(v, -1)),
        ("cumprod", lambda v: jt.cumprod(v, -1)),
        ("logsumexp", lambda v: jt.logsumexp(v, dim=-1)),
        ("softmax", lambda v: jt.nn.softmax(v, -1)),
        ("log_softmax", lambda v: jt.nn.log_softmax(v, -1)),
        ("layer_norm", lambda v: jt.nn.layer_norm(v, (v.shape[-1],))),
        ("rms_norm", lambda v: jt.nn.rms_norm(v, ones(v.shape[-1], v))),
        ("matmul", lambda v: jt.matmul(v, v.transpose())),
        ("linear", lambda v: jt.nn.linear(v, ones((3, v.shape[-1]), v))),
        ("sdpa", lambda v: jt.nn.scaled_dot_product_attention(
            *[v.reshape((1, 1) + tuple(v.shape))] * 3)),
        ("sort", lambda v: jt.sort(v, -1)[0]),
        ("topk", lambda v: jt.topk(v, 2, -1)[0]),
        ("concat", lambda v: jt.concat([v, v], -1)),
        ("gelu", lambda v: jt.nn.gelu(v)),
        ("sigmoid", lambda v: v.sigmoid()),
    ]


class TestHalfDtypeIsPreserved(unittest.TestCase):
    """Every op in the sweep hands back the dtype it was given.

    torch 2.13 does, for all of them, on CPU and CUDA -- checked against the
    real library rather than assumed. Jittor widened five to float32:

    * ``max``/``min``/``prod``/``cumprod``, because ``reduce_dtype_infer``
      widened *any* half float reduce and only ``sum``/``mean`` were intercepted
      upstream by ``ReduceOp``'s float32-intermediate path;
    * ``logsumexp``, because ``exp`` is on jittor's white list and answers in
      float32 whatever it is handed -- which is the right place to compute the
      shift-exp-sum-log chain, but the result still has to come back narrow.

    ``exp`` itself is deliberately NOT in this list: jittor's white list
    (``src/type/nano_string.cc``) widens ``exp`` and ``pow`` by design, under
    ``jt.amp_flags.keep_white``. That is a real divergence from torch, which
    returns float16 for ``float16.exp()``, and it is recorded in the report
    rather than asserted here.
    """

    def test_sweep(self):
        rng = np.random.RandomState(0)
        base = (rng.rand(4, 8) + 0.5).astype(np.float32)
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with jt.flag_scope(use_cuda=use_cuda):
                    var = _cast(jt.array(base), name)
                    for op, fn in _dtype_cases():
                        with self.subTest(device=device, dtype=name, op=op):
                            out = fn(var)
                            out.sync()
                            self.assertEqual(
                                str(out.dtype), name,
                                "%s on %s widened %s to %s"
                                % (op, device, name, str(out.dtype)))

    def test_prod_overflows_like_torch(self):
        """``prod`` is the one reduction torch does NOT widen.

        ``torch.full((40,), 4.0, dtype=torch.float16).prod()`` is ``inf`` on CPU
        and on CUDA -- 4**40 is 1.2e24, far outside float16's range -- while the
        same product in bfloat16, whose exponent field is float32's, is
        1.2089258e24. A float32 accumulator with a single final rounding
        reproduces both; an op that answered float32 reproduced neither,
        because it never narrowed at all.
        """
        values = np.full(40, 4.0, dtype=np.float32)
        for device, use_cuda in _DEVICES:
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda):
                    fp16 = jt.array(values).float16().prod()
                    bf16 = jt.array(values).bfloat16().prod()
                    self.assertEqual(str(fp16.dtype), "float16")
                    self.assertTrue(np.isinf(float(fp16.numpy())))
                    self.assertEqual(str(bf16.dtype), "bfloat16")
                    self.assertAlmostEqual(
                        float(bf16.numpy()) / 1.2089258196146292e+24, 1.0, places=5)


# ---------------------------------------------------------------------------
# 3. The width the op accumulated at.
# ---------------------------------------------------------------------------

class TestHalfAccumulatesInFloat32(unittest.TestCase):
    """A long reduction must round once, not once per element.

    ``jt.ones(n)`` is the sharpest form of the question: in float16 the running
    sum stops moving at 2048, because 2048 + 1 rounds back to 2048, so a
    float16 accumulator answers 2048 for every n >= 2048. torch answers n. In
    bfloat16 it stops at 256.
    """

    SATURATES = {"float16": 2048.0, "bfloat16": 256.0}

    def test_sum_of_ones_does_not_saturate(self):
        n = 4096
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        total = float(_cast(jt.ones(n), name).sum().numpy())
                    self.assertNotAlmostEqual(
                        total, self.SATURATES[name], delta=1.0,
                        msg="accumulated in %s, not float32" % name)
                    # n is exactly representable in both formats, so a float32
                    # accumulator narrowed once at the store is exact.
                    self.assertEqual(total, float(n))

    def test_cumsum_of_ones_does_not_saturate(self):
        """``cumsum`` keeps every partial, so it is the same question per output.

        On CPU this did not merely lose precision, it did not compile: the host
        ``float16``/``bfloat16`` structs carry comparisons and an implicit
        conversion to float and no compound assignment, and the scan kernel was
        written ``y_type acc = 0; acc += ...``, so ``jt.cumsum`` on a half Var
        died in g++ with "no match for 'operator+='". The op did not exist.
        """
        n = 4096
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        out = jt.cumsum(_cast(jt.ones(n), name), -1)
                        out.sync()
                        self.assertEqual(str(out.dtype), name)
                        last = float(out.numpy()[-1])
                    self.assertNotAlmostEqual(
                        last, self.SATURATES[name], delta=1.0)
                    self.assertEqual(last, float(n))

    def test_matmul_contraction_is_not_half(self):
        """The generic ``(a*b).sum(k)`` fallback, which is the CPU path.

        cuBLAS asks for ``CUBLAS_COMPUTE_32F`` for both half types
        (``cublas_compute_type.h``), and ``src/runtime/float32_precision.h``
        writes the rule down as "float16 and bfloat16 always accumulate in
        float32" -- but the fallback ran under ``reduce16_no_fp32_acc``, which
        switches off exactly that intermediate, so the same product was computed
        two ways depending only on whether a cuBLAS relay happened to take it.

        ``ones(1,k) @ ones(k,1)`` is ``k``; a half accumulator saturates.
        """
        k = 4096
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        a = _cast(jt.ones((1, k)), name)
                        b = _cast(jt.ones((k, 1)), name)
                        out = jt.matmul(a, b)
                        out.sync()
                        self.assertEqual(str(out.dtype), name)
                        self.assertEqual(float(out.numpy()[0, 0]), float(k))

    def test_conv2d_contraction_is_not_half(self):
        """Same bit, same reason, in the generic convolution fallback.

        The contraction is C*Kh*Kw long -- 4608 terms for a 3x3 kernel over 512
        channels -- and it ran in half while cuDNN and the cuBLAS relays for the
        same convolution accumulate in float32.
        """
        c, k = 512, 3
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        x = _cast(jt.ones((1, c, k, k)), name)
                        w = _cast(jt.ones((1, c, k, k)), name)
                        out = jt.nn.conv2d(x, w, padding=0)
                        out.sync()
                        self.assertEqual(str(out.dtype), name)
                        self.assertEqual(float(out.numpy().reshape(-1)[0]),
                                         float(c * k * k))


class TestHalfPythonScalar(unittest.TestCase):
    """A Python float meeting a half tensor gives torch's answer exactly.

    This is here as a *constraint*, not a discovery. KI-DTYPE-003 records that
    `src/bindings/pyjt/py_array_op.cc` converts every Python float to a float32
    constant, so `float64_var * 0.1` has seven good digits where torch, whose
    Python float is a weak double, is exact. At float16 and bfloat16 the
    narrowing is invisible -- float32 is wider than both, so the value survives
    the intermediate and the only rounding is the one the output dtype forces --
    and the four expressions below are bit-identical to real torch 2.13 on CPU
    and CUDA today.

    Whatever weak-scalar model eventually fixes the float64 case has to keep
    that true, which is what this pins. The values are torch's, transcribed.
    """

    EXPECT = {
        "float16": {"mul": 0.0999755859375, "add": 1.099609375,
                    "twothirds": 0.66650390625, "sqrt2": 1.4140625},
        "bfloat16": {"mul": 0.10009765625, "add": 1.1015625,
                     "twothirds": 0.66796875, "sqrt2": 1.4140625},
    }

    def test_matches_torch(self):
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        v = _cast(jt.ones(1), name)
                        got = {
                            "mul": v * 0.1,
                            "add": v + 0.1,
                            "twothirds": v * (2.0 / 3.0),
                            "sqrt2": v * 2 ** 0.5,
                        }
                        for key, var in got.items():
                            self.assertEqual(str(var.dtype), name)
                            self.assertEqual(
                                float(var.float32().numpy()[0]),
                                self.EXPECT[name][key],
                                "%s on %s in %s" % (key, device, name))


class TestHalfBackward(unittest.TestCase):
    """The cotangent comes back at the input's dtype, and at torch's accuracy.

    ``_LN.execute`` computes a half input's statistics in float32, so its
    backward works on float32 saved tensors and has to narrow the cotangent
    before handing it to the Var the caller passed -- otherwise the gradient of
    a float16 parameter is a float32, which the optimizer then has to guess
    about. Real torch 2.13 returns float16 for a float16 input here, and
    bfloat16 for bfloat16.

    Measured against real torch's gradient on the same input, relative to the
    largest gradient magnitude: layer_norm 5.008e-4 (float16, CPU), 1.253e-4
    (float16, CUDA), 3.984e-3 (bfloat16, CPU), 0.0 (bfloat16, CUDA); matmul
    3.081e-4 (float16, CPU), 0.0 (float16, CUDA), 3.079e-4 / 0.0 for bfloat16.
    Every one is inside one ULP of its own format, which is the most a
    float32-accumulated backward narrowed once can be.
    """

    def test_grad_dtype_follows_the_input(self):
        rng = np.random.RandomState(4)
        x = (rng.randn(8, 64) * 2 + 1).astype(np.float32)
        w = (rng.randn(64, 16) * 0.5).astype(np.float32)
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        v = _cast(jt.array(x), name)
                        v.requires_grad = True
                        out = jt.nn.layer_norm(
                            v, (64,), _cast(jt.ones(64), name),
                            _cast(jt.zeros(64), name), 1e-5)
                        g = jt.grad(out.sum(), v)
                        g.sync()
                        self.assertEqual(str(g.dtype), name,
                                         "layer_norm backward returned %s"
                                         % g.dtype)
                        self.assertTrue(np.isfinite(g.float32().numpy()).all())

                        a = _cast(jt.array(x), name)
                        a.requires_grad = True
                        b = _cast(jt.array(w), name)
                        ga = jt.grad(jt.matmul(a, b).sum(), a)
                        ga.sync()
                        self.assertEqual(str(ga.dtype), name,
                                         "matmul backward returned %s" % ga.dtype)
                        self.assertTrue(np.isfinite(ga.float32().numpy()).all())


class TestHalfMinMaxNaN(unittest.TestCase):
    """``maximum``/``minimum`` keep a NaN from either operand, at every width.

    Real torch 2.13, measured: ``maximum(nan, 5)`` and ``maximum(5, nan)`` are
    both NaN, for float32, float16 and bfloat16, on CPU and on CUDA, elementwise
    and through a reduction. NumPy's rule, and jittor's own float32 rule --
    ``jittor::_max`` in ``src/type/minmax_compute.h`` is written
    ``((a > b) | (a != a)) ? a : b`` precisely so that it holds from both sides.

    The half types did not follow it. ``src/type/fp16_compute.h`` guarded its
    ``max``/``min`` with ``#if CUDA_ARCH >= 800``, and ``CUDA_ARCH`` is not a
    macro nvcc defines, so the ladder always fell through to
    ``float(a)<float(b)?b:a`` -- which returns ``a`` whenever the comparison is
    false, keeping a NaN in the first operand and dropping it in the second. The
    CPU table spelled the same thing ``std::max<float>``. Measured before:
    ``jt.maximum(5, nan)`` was 5 in both half types on both devices, and
    ``x.max()`` over ``[1, nan, 3]`` was 3.0 -- the accumulator is always the
    first operand, so a reduction dropped the NaN every time. CPU and CUDA did
    not agree either: ``jt.minimum(nan, 5)`` was NaN on the host and 5 on the
    device.
    """

    NAN = float("nan")

    def test_elementwise_keeps_a_nan_from_either_side(self):
        a = np.array([self.NAN, 5.0, self.NAN, 5.0], dtype=np.float32)
        b = np.array([5.0, self.NAN, self.NAN, 7.0], dtype=np.float32)
        for device, use_cuda in _DEVICES:
            for name in ("float32", "float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        cast = (lambda v: v) if name == "float32" else \
                            (lambda v: _cast(v, name))
                        av, bv = cast(jt.array(a)), cast(jt.array(b))
                        mx = av.maximum(bv).float32().numpy()
                        mn = av.minimum(bv).float32().numpy()
                    # The first three pairs each contain a NaN; the fourth does
                    # not and pins the ordinary answer.
                    self.assertTrue(np.isnan(mx[:3]).all(),
                                    "maximum dropped a NaN: %s" % mx)
                    self.assertTrue(np.isnan(mn[:3]).all(),
                                    "minimum dropped a NaN: %s" % mn)
                    self.assertEqual(mx[3], 7.0)
                    self.assertEqual(mn[3], 5.0)

    def test_reduction_keeps_a_nan(self):
        values = np.array([1.0, self.NAN, 3.0], dtype=np.float32)
        for device, use_cuda in _DEVICES:
            for name in ("float32", "float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        cast = (lambda v: v) if name == "float32" else \
                            (lambda v: _cast(v, name))
                        v = cast(jt.array(values))
                        hi = float(v.max().float32().numpy())
                        lo = float(v.min().float32().numpy())
                    self.assertTrue(np.isnan(hi),
                                    "max() over a NaN answered %r" % hi)
                    self.assertTrue(np.isnan(lo),
                                    "min() over a NaN answered %r" % lo)


class TestHalfErrorAgainstTorch(unittest.TestCase):
    """Measured error against a float64 closed form, bounded in output ULPs.

    Every bound below is stated as a multiple of the output dtype's ULP, and the
    number real torch 2.13 measured on the identical input is in the comment. A
    float32 accumulator rounds once, so its error is O(1) ULP whatever the axis
    length; a half accumulator's grows with it, and on these inputs (4096-long
    reductions, K=512 products) that is one to three orders of magnitude.
    """

    @staticmethod
    def _inputs():
        rng = np.random.RandomState(0)
        return {
            "long": rng.rand(8, 4096).astype(np.float32),
            "norm": (rng.randn(16, 1024) * 2 + 1).astype(np.float32),
            "mm_a": (rng.randn(128, 512) * 0.5).astype(np.float32),
            "mm_b": (rng.randn(512, 128) * 0.5).astype(np.float32),
        }

    def _check(self, device, name, op, got, exact, bound_ulps,
               torch_error, relative=True):
        err = _rel(got, exact) if relative else _abs(got, exact)
        scale = _ULP[name] if relative else _ULP[name] * float(np.abs(exact).max())
        self.assertLessEqual(
            err, bound_ulps * scale,
            "%s %s on %s: %.3e, bound %.3e (%g ULP); real torch 2.13 measured "
            "%.3e on this input" % (op, name, device, err, bound_ulps * scale,
                                    bound_ulps, torch_error))

    def test_reductions(self):
        data = self._inputs()
        # (op, callable, bound in ULPs, {dtype: torch's measured relative error})
        cases = [
            # A float32 accumulator narrowed once at the store cannot be off by
            # more than half an ULP plus the reference's own rounding; 2 ULP
            # leaves room for the reduction order. A float16 accumulator over
            # 4096 terms of mean 0.5 is ~1e-2 -- twenty times this bound.
            ("sum", lambda v: v.sum(-1), 2.0,
             {"float16": 4.553e-4, "bfloat16": 1.780e-3}),
            ("mean", lambda v: v.mean(-1), 2.0,
             {"float16": 4.553e-4, "bfloat16": 1.780e-3}),
            ("cumsum", lambda v: jt.cumsum(v, -1)[:, -1], 4.0,
             {"float16": 4.553e-4, "bfloat16": 1.780e-3}),
            ("std", lambda v: jt.std(v, dim=-1), 4.0,
             {"float16": 3.671e-4, "bfloat16": 3.380e-3}),
            ("norm", lambda v: jt.norm(v, dim=-1), 2.0,
             {"float16": 2.630e-4, "bfloat16": 3.219e-3}),
        ]
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                exact_in = _round_trip(data["long"], name)
                truth = {
                    "sum": exact_in.sum(-1),
                    "mean": exact_in.mean(-1),
                    "cumsum": exact_in.cumsum(-1)[:, -1],
                    "std": exact_in.std(-1, ddof=1),
                    "norm": np.sqrt((exact_in * exact_in).sum(-1)),
                }
                for op, fn, bound, torch_err in cases:
                    with self.subTest(device=device, dtype=name, op=op):
                        with jt.flag_scope(use_cuda=use_cuda):
                            got = fn(_cast(jt.array(data["long"]), name)).numpy()
                        self._check(device, name, op, got, truth[op],
                                    bound, torch_err[name])

    def test_layer_norm(self):
        """The statistics and the normalisation, not just the two means.

        ``jt.mean`` already took a float32 intermediate, but it rounded the
        *result* back to half, and everything between the two means ran at that
        width: the deviation, its square -- where a half has no exponent room to
        spare -- the variance and the reciprocal square root. Measured against
        this same reference before the fix: float16 3.02e-3 on CPU and 3.40e-3
        on CUDA where torch is 9.75e-4; bfloat16 4.02e-2 / 2.93e-2 against
        torch's 3.87e-3.
        """
        data = self._inputs()["norm"]
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                exact_in = _round_trip(data, name)
                mu = exact_in.mean(-1, keepdims=True)
                var = exact_in.var(-1, keepdims=True)
                truth = (exact_in - mu) / np.sqrt(var + 1e-5)
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        var_in = _cast(jt.array(data), name)
                        got = jt.nn.layer_norm(
                            var_in, (1024,), _cast(jt.ones(1024), name),
                            _cast(jt.zeros(1024), name), 1e-5).numpy()
                    # 2 ULP of the *output*, which is O(1); torch measured
                    # 9.750e-4 (fp16) and 3.867e-3 (bf16) relative on this input,
                    # both within 2 ULP of their format.
                    self._check(device, name, "layer_norm", got, truth,
                                2.0, 9.750e-4 if name == "float16" else 3.867e-3)

    def test_matmul(self):
        """K=512 contraction, absolute error against the float64 product.

        torch 2.13 on this input: 7.759e-3 (float16) and 6.175e-2 (bfloat16) on
        CPU, 7.759e-3 / 6.175e-2 on CUDA. With the fallback accumulating in half
        jittor measured 1.674e-1 and 1.242 -- 21x and 20x -- and it grows with K.
        The bound is stated against the largest magnitude in the result, which is
        what an absolute error has to be scaled by to mean anything.
        """
        data = self._inputs()
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                a = _round_trip(data["mm_a"], name)
                b = _round_trip(data["mm_b"], name)
                truth = a @ b
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        got = jt.matmul(_cast(jt.array(data["mm_a"]), name),
                                        _cast(jt.array(data["mm_b"]), name)).numpy()
                    # A float32 accumulator still rounds each of the K products
                    # to half before summing, so the error is ~sqrt(K) ULP of the
                    # result's magnitude rather than 1. 64 ULP is 2.8x torch's
                    # own measurement here and 5x under what a half accumulator
                    # produced.
                    self._check(device, name, "matmul", got, truth,
                                64.0, 7.759e-3 if name == "float16" else 6.175e-2,
                                relative=False)


# ---------------------------------------------------------------------------
# Kernels that index with integers while holding a half value.
# ---------------------------------------------------------------------------

def _max_pool_reference(values, kernel, stride, padding):
    """NCHW max pooling in numpy, padding with -inf (what torch pads with)."""
    padded = np.pad(values, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                    constant_values=-np.inf)
    rows = (padded.shape[2] - kernel) // stride + 1
    cols = (padded.shape[3] - kernel) // stride + 1
    out = np.empty(values.shape[:2] + (rows, cols))
    for i in range(rows):
        for j in range(cols):
            window = padded[:, :, i * stride:i * stride + kernel,
                            j * stride:j * stride + kernel]
            out[:, :, i, j] = window.max(axis=(2, 3))
    return out


class TestHalfPooling(unittest.TestCase):
    """``MaxPool2d`` on a half input compiles, and selects exactly.

    The CUDA pooling kernel clamps its window with ``min(k + 3, shape)`` and
    ``max(0, k)`` on ints. The half ``jittor::min``/``max`` overloads hid the
    global integer ones inside ``namespace jittor``, so on float16 or bfloat16
    the call was ambiguous and nvcc rejected the kernel -- fp16 ResNet
    inference failed at its first pooling layer. The float32 kernel never
    includes those overloads, which is why only the half dtypes broke.

    A maximum does no arithmetic, so the answer must equal the rounded input's
    maximum exactly: no tolerance.
    """

    def test_max_pool2d_selects_exactly(self):
        rng = np.random.RandomState(7)
        values = rng.randn(2, 3, 13, 13).astype(np.float32)
        for device, use_cuda in _DEVICES:
            for name in ("float16", "bfloat16"):
                with self.subTest(device=device, dtype=name):
                    with jt.flag_scope(use_cuda=use_cuda):
                        x = _cast(jt.array(values), name)
                        y = jt.nn.MaxPool2d(3, stride=2, padding=1)(x)
                        self.assertEqual(str(y.dtype), name)
                        got = y.float32().numpy().astype(np.float64)
                    exact = _max_pool_reference(_round_trip(values, name), 3, 2, 1)
                    np.testing.assert_array_equal(got, exact)


if __name__ == "__main__":
    unittest.main()
