# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CPU max/min reduction starts from the wrong identity -- KI-OPS-008.

``init_maximum`` is ``std::numeric_limits<T>::lowest()`` in the CPU table and
``-CUDART_INF`` in the CUDA one. For a float that is -3.4e38 versus -inf, and
the two are not interchangeable: -inf is a perfectly ordinary value for a
tensor to contain, and ``max(-3.4e38, -inf)`` keeps the identity rather than the
element. So a tensor whose maximum really is -inf reports -3.4e38 on CPU and
-inf on CUDA -- one call, two answers, no error. An attention mask is the
obvious way to hit it: a fully masked row is all -inf, and ``logits.max(-1)``
is exactly this reduction.

Integers are unaffected: ``lowest()`` *is* their identity, and there is no
integer infinity to lose. That is what the identity has to dispatch on.

The CUDA class asserts the correct behaviour and passes; the CPU class carries
the strict expected failure. Keeping both in one file is the point -- the
divergence is between the two backends, so neither half means anything alone.
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


class TestMinMaxReductionIdentityCpu(_IdentityContract, unittest.TestCase):
    device_flag = 0

    @pytest.mark.xfail(strict=True,
                       reason="KI-OPS-008: the CPU identity is lowest(), not -inf")
    def test_infinite_reductions_use_the_right_identity(self):
        self._check_infinite_identity()


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestMinMaxReductionIdentityCuda(_IdentityContract, unittest.TestCase):
    device_flag = 1

    def test_infinite_reductions_use_the_right_identity(self):
        self._check_infinite_identity()


if __name__ == "__main__":
    unittest.main()
