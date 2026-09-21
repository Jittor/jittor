# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

"""The identity a max/min reduction folds *from* must be an infinity for floats.

``init_maximum`` in the CPU table of ``common_op_type.cc`` used to be
``std::numeric_limits<T>::lowest()`` while the CUDA table used `-CUDART_INF``.
For a float those are -3.4e38 and -inf, and they are not interchangeable:
``max(-3.4e38, -inf)`` keeps the identity rather than the element, so a tensor
whose maximum really is -inf reported -3.4e38 on CPU and -inf on CUDA -- one
call, two answers, no error. An attention mask is the obvious way to hit it: a
fully masked row is all -inf, and ``logits.max(-1)`` is exactly this reduction.
That was KI-OPS-008; the CPU table now dispatches on ``has_infinity``.

Integers are unaffected and must stay that way: ``lowest()`` *is* their
identity, and there is no integer infinity to lose. The integer test below is
the control for that half of the dispatch.

The *half-precision* table (``fp16_op_type.cc``) carries the same defect in a
worse form, and that is KI-OPS-012: its identities are plain finite literals
(``-32768.0f`` on the CPU path, ``-65000.0f``/``-1e38`` on the CUDA one), and
``-32768`` is *above* float16's lowest finite value of -65504. So there the
reduction is wrong on finite input, not only on infinities.

The CUDA class asserts the correct behaviour for float32/float64 and passes; it
is kept in the same file on purpose, because for that half the defect was a
divergence *between* the backends and neither half means anything alone.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np
import pytest

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


class _IdentityContract:

    device_flag = 0

    def _var(self, values, dtype):
        with jt.flag_scope(auto_convert_64_to_32=0):
            v = jt.array(np.asarray(values, dtype=dtype))
        self.assertEqual(str(v.dtype), dtype)
        return v

    def test_integer_reductions_use_the_right_identity(self):
        # The control: for integers `lowest()` is the correct identity, so this
        # half must keep passing through any fix.
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("int8", "int16", "int32", "int64"):
                info = np.iinfo(dtype)
                raw = np.array([info.min, info.min], dtype=dtype)
                self.assertEqual(int(self._var(raw, dtype).max().numpy()), int(info.min))
                raw = np.array([info.max, info.max], dtype=dtype)
                self.assertEqual(int(self._var(raw, dtype).min().numpy()), int(info.max))

    def test_finite_reductions_are_unaffected(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([-1e38, 3.0, -7.0], dtype="float32")
            self.assertEqual(float(self._var(raw, "float32").max().numpy()),
                             float(np.max(raw)))
            self.assertEqual(float(self._var(raw, "float32").min().numpy()),
                             float(np.min(raw)))

    def _check_infinite_identity(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for dtype in ("float32", "float64"):
                neg = np.array([-np.inf, -np.inf], dtype=dtype)
                self.assertEqual(float(self._var(neg, dtype).max().numpy()),
                                 float(np.max(neg)),
                                 "%s: max() of an all -inf tensor" % dtype)
                pos = np.array([np.inf, np.inf], dtype=dtype)
                self.assertEqual(float(self._var(pos, dtype).min().numpy()),
                                 float(np.min(pos)),
                                 "%s: min() of an all +inf tensor" % dtype)

    def _check_half_identity(self):
        """KI-OPS-012: the half tables fold from a finite literal.

        Two failures per dtype, and the second is the worse one: the value at
        the edge of the format is an ordinary number, not a diagnostic. -32768
        is not a float16 value for `max` to fall back to -- it is above -65504,
        so the reduction answers a number that was not in its input.

        The finite cases compare against the value the tensor actually holds
        rather than against the Python literal that went in. A three-step cast
        (float64 -> float32 -> bfloat16) does not have to land back on the
        literal it started from, and an assertion that depends on that rounding
        would be about the cast, not about the identity.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            bounds = {
                "float16": (float(np.finfo("float16").min),
                            float(np.finfo("float16").max)),
                "bfloat16": (-3.3895313892515355e38, 3.3895313892515355e38),
            }
            for dtype, (lo, hi) in bounds.items():
                source = "float16" if dtype == "float16" else "float32"

                neg_inf = self._var([-np.inf, -np.inf], "float16").cast(dtype)
                self.assertEqual(float(neg_inf.max().numpy()), -np.inf,
                                 "%s: max() of an all -inf tensor" % dtype)
                pos_inf = self._var([np.inf, np.inf], "float16").cast(dtype)
                self.assertEqual(float(pos_inf.min().numpy()), np.inf,
                                 "%s: min() of an all +inf tensor" % dtype)

                neg = self._var([lo, lo], source).cast(dtype)
                self.assertEqual(float(neg.max().numpy()),
                                 float(neg.numpy()[0]),
                                 "%s: max() of an all-lowest-finite tensor" % dtype)
                pos = self._var([hi, hi], source).cast(dtype)
                self.assertEqual(float(pos.min().numpy()),
                                 float(pos.numpy()[0]),
                                 "%s: min() of an all-highest-finite tensor" % dtype)


class TestMinMaxReductionIdentityCpu(_IdentityContract, unittest.TestCase):
    device_flag = 0

    def test_infinite_reductions_use_the_right_identity(self):
        # Was a strict expected failure for KI-OPS-008. The CPU table now
        # dispatches on has_infinity, so this is a plain assertion again.
        self._check_infinite_identity()

    def test_half_precision_reductions_use_the_right_identity(self):
        # KI-OPS-012, CPU half: the table answered from `-32768.0f`, which is
        # above float16's lowest finite value. It now answers from an infinity.
        self._check_half_identity()


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestMinMaxReductionIdentityCuda(_IdentityContract, unittest.TestCase):
    device_flag = 1

    def test_infinite_reductions_use_the_right_identity(self):
        self._check_infinite_identity()

    @pytest.mark.xfail(strict=True,
                       reason="KI-OPS-012: the CUDA half identities are finite literals")
    def test_half_precision_reductions_use_the_right_identity(self):
        # The CUDA half table still spells its identities as -65000.0f / -1e38
        # for float16 and bfloat16. Both sit above the dtype's lowest finite
        # value, so this is the same wrong answer as the CPU one had -- and it
        # is stated as an expected failure rather than fixed, because a fix for
        # it cannot be verified on this machine.
        self._check_half_identity()


if __name__ == "__main__":
    unittest.main()
