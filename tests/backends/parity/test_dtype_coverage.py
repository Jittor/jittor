# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Per-dtype VALUE coverage: integer-width sweep + low-precision (fp16/bf16) forward.

``test_ops`` runs each op on its declared dtypes (mostly float32/64); ``test_low_precision``
locks the fp16/bf16 *gradient dtype*. Neither pins, across the **full dtype lattice**, that
the kernel computes the right NUMBERS -- which is where the accelerator's per-width integer
paths and its half-precision paths actually break (this project already carries a real
sub-32-bit-int CUDA reduce bug; int8/16 scatter and fp16 elementwise are the same risk class).

Two batteries, both value-level:

  * **Integer width sweep** -- elementwise integer ops across uint8/int8/int16/int32/int64,
    asserted EXACTLY against numpy AND (CPU vs CUDA) bit-identical. A width whose kernel
    overflows, sign-extends wrong, or is missing falls out here, per width.
  * **Low-precision forward** -- fp16 and bf16 elementwise / matmul / reduce / softmax,
    asserted CPU-vs-accelerator within half-precision tolerance and against an fp32-computed
    reference. Catches a half kernel that diverges from the fp32 math by more than rounding.

Run::  python -m pytest tests/backends/parity/test_dtype_coverage.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, HAS_CUDA, HAS_ACL, to_numpy,
)

_ACCEL = HAS_CUDA or HAS_ACL

_INT_DTYPES = ["uint8", "int8", "int16", "int32", "int64"]


def _run(build, use_cuda):
    with jt.flag_scope(use_cuda=use_cuda):
        return to_numpy(build())


class TestIntegerWidthSweep(JittorTestCase):
    """Elementwise integer ops, every width, exact vs numpy + CPU-vs-accelerator."""

    # (name, jittor op, numpy op). Kept to ops with a well-defined integer result; values
    # are sampled in a SAFE range (no overflow) so the test measures kernel correctness,
    # not wraparound semantics.
    _BINARY = [
        ("add", lambda a, b: a + b, np.add),
        ("sub", lambda a, b: a - b, np.subtract),
        ("mul", lambda a, b: a * b, np.multiply),
        ("maximum", jt.maximum, np.maximum),
        ("minimum", jt.minimum, np.minimum),
        ("bitwise_and", jt.bitwise_and, np.bitwise_and),
        ("bitwise_or", jt.bitwise_or, np.bitwise_or),
        ("bitwise_xor", jt.bitwise_xor, np.bitwise_xor),
    ]
    _UNARY = [
        ("negative", lambda a: -a, np.negative),
        ("abs", jt.abs, np.abs),
    ]

    def _operands(self, dt):
        # small non-negative range valid for every width incl. uint8; signed widths also
        # get a negative operand for sub/negative/abs sign handling.
        signed = not dt.startswith("uint")
        lo = -20 if signed else 0
        a = np.random.RandomState(1).randint(lo, 20, size=(3, 4)).astype(dt)
        b = np.random.RandomState(2).randint(1, 15, size=(3, 4)).astype(dt)
        return a, b

    def test_binary_integer_widths(self):
        for dt in _INT_DTYPES:
            a, b = self._operands(dt)
            for name, jop, nop in self._BINARY:
                with self.subTest(dtype=dt, op=name):
                    cpu = _run(lambda: jop(jt.array(a), jt.array(b)), 0)
                    self.assertEqual(cpu, nop(a, b),
                                     msg=f"{name}/{dt} cpu vs numpy")
                    if _ACCEL:
                        acc = _run(lambda: jop(jt.array(a), jt.array(b)), 1)
                        self.assertEqual(cpu, acc,
                                         msg=f"{name}/{dt} cpu vs accelerator")

    def test_unary_integer_widths(self):
        for dt in _INT_DTYPES:
            a, _ = self._operands(dt)
            for name, jop, nop in self._UNARY:
                with self.subTest(dtype=dt, op=name):
                    cpu = _run(lambda: jop(jt.array(a)), 0)
                    # numpy negative/abs on uint8 wrap; compare the SAME numpy op so the
                    # reference matches the width's two's-complement semantics.
                    self.assertEqual(cpu, nop(a), msg=f"{name}/{dt} cpu vs numpy")
                    if _ACCEL:
                        acc = _run(lambda: jop(jt.array(a)), 1)
                        self.assertEqual(cpu, acc, msg=f"{name}/{dt} cpu vs accelerator")


@unittest.skipUnless(_ACCEL, "low-precision dtypes need an accelerator (CUDA/NPU)")
class TestLowPrecisionForward(JittorTestCase):
    """fp16 / bf16 forward values: accelerator-vs-CPU and vs an fp32 reference."""

    # half-precision tolerances: fp16 ~ 2^-10, bf16 ~ 2^-7 relative.
    _TOL = {"float16": (3e-3, 3e-3), "bfloat16": (3e-2, 3e-2)}

    def _cast(self, np32, dtype, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            return jt.array(np32).cast(dtype)

    def _check(self, dtype, build_jt, ref32, label):
        atol, rtol = self._TOL[dtype]
        # accelerator value vs fp32 reference
        acc = _run(lambda: build_jt(dtype, 1), 1).astype("float32")
        self.assertEqual(acc, ref32, atol=atol, rtol=rtol,
                         msg=f"{label} [{dtype}] accelerator vs fp32 ref")
        # CPU vs accelerator (same low-precision dtype) -- tighter, just rounding apart
        cpu = _run(lambda: build_jt(dtype, 0), 0).astype("float32")
        self.assertEqual(cpu, acc, atol=atol, rtol=rtol,
                         msg=f"{label} [{dtype}] cpu vs accelerator")

    def test_elementwise(self):
        rng = np.random.RandomState(0)
        a32 = rng.randn(4, 8).astype("float32")
        ref = np.exp(a32 * 0.5) + np.tanh(a32)
        for dt in ("float16", "bfloat16"):
            with self.subTest(dtype=dt):
                self._check(dt,
                            lambda dtype, uc: jt.exp(self._cast(a32, dtype, uc) * 0.5)
                            + jt.tanh(self._cast(a32, dtype, uc)),
                            ref, "elementwise exp+tanh")

    def test_matmul(self):
        rng = np.random.RandomState(1)
        a32 = rng.randn(8, 8).astype("float32")
        b32 = rng.randn(8, 8).astype("float32")
        ref = a32 @ b32
        for dt in ("float16", "bfloat16"):
            with self.subTest(dtype=dt):
                self._check(dt,
                            lambda dtype, uc: self._cast(a32, dtype, uc) @ self._cast(b32, dtype, uc),
                            ref, "matmul")

    def test_reduce_sum(self):
        rng = np.random.RandomState(2)
        a32 = rng.randn(6, 10).astype("float32")
        ref = a32.sum(1)
        for dt in ("float16", "bfloat16"):
            with self.subTest(dtype=dt):
                self._check(dt,
                            lambda dtype, uc: self._cast(a32, dtype, uc).sum(1),
                            ref, "reduce sum")

    def test_softmax(self):
        rng = np.random.RandomState(3)
        a32 = rng.randn(4, 12).astype("float32")
        e = np.exp(a32 - a32.max(1, keepdims=True))
        ref = e / e.sum(1, keepdims=True)
        for dt in ("float16", "bfloat16"):
            with self.subTest(dtype=dt):
                self._check(dt,
                            lambda dtype, uc: jt.nn.softmax(self._cast(a32, dtype, uc), dim=1),
                            ref, "softmax")


if __name__ == "__main__":
    unittest.main(verbosity=2)
