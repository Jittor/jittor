# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reducing a rank-0 tensor returns the value, as PyTorch and NumPy do.

``loss.sum()`` where the loss is already a scalar is ordinary code -- generic
training loops reduce without checking rank -- and it died on both devices
(KI-OPS-004). The reduce kernel opens with ``index_t ystride@{DIM-1} = 1;``,
and ``DIM`` is zero for a rank-0 input, so the generated source read
``index_t ystride-1 = 1;`` and never compiled. The error surfaced as a
compiler diagnostic about a template file, naming neither the shape nor the
operator.

What makes the fix correct rather than merely compiling: with the guard, every
``@for`` in the kernel produces an empty nest, the body runs once with
``yid == xid == 0``, and the result is the single input element. That is what
the reduction of one element is, and it is what the reference implementations
return.

Shape is asserted alongside value. A reduction that returned ``3.5`` with shape
``(1,)`` would satisfy a value-only check and still break every caller that
feeds the result somewhere expecting a scalar -- which is the code this exists
for.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Every reduction that takes an identity, plus the value it must return.
VALUE = 3.5
REDUCTIONS = ("sum", "mean", "max", "min", "prod")


class _Rank0Reductions:

    device_flag = 0

    def _scalar(self):
        x = jt.array(np.float32(VALUE))
        # KI-DTYPE-002: jt.array narrows 64-bit values; say which width this is.
        assert str(x.dtype) == "float32", x.dtype
        self.assertEqual(x.ndim, 0, "the input is not rank-0; the test is void")
        return x

    def test_every_reduction_returns_the_value(self):
        for name in REDUCTIONS:
            fn = getattr(jt, name, None)
            if fn is None:
                continue
            with self.subTest(reduction=name):
                with jt.flag_scope(use_cuda=self.device_flag):
                    result = fn(self._scalar())
                    value = float(np.asarray(result.numpy()).ravel()[0])
                    shape = tuple(result.shape)
                self.assertAlmostEqual(value, VALUE, places=5,
                                       msg="%s of a rank-0 tensor" % name)
                self.assertEqual(shape, (),
                                 "%s of a rank-0 tensor should stay rank-0, "
                                 "got shape %s" % (name, shape))

    def test_the_method_spelling_too(self):
        """`loss.sum()` is how this is written in a training loop."""
        with jt.flag_scope(use_cuda=self.device_flag):
            loss = jt.array(np.float32(2.0))
            self.assertAlmostEqual(float(loss.sum().item()), 2.0, places=5)
            self.assertAlmostEqual(float(loss.mean().item()), 2.0, places=5)

    def test_ordinary_ranks_are_unchanged(self):
        """The guard must not have moved anything that already worked.

        A `@if(DIM>0, ...)` guard is exactly the kind of edit that can drop a
        stride declaration for every rank, not just the one it was written for,
        and a whole-tensor sum would still look right while a per-axis one
        silently used the wrong strides. So both are checked, and the per-axis
        case is compared elementwise rather than by its total.
        """
        host = np.arange(24, dtype="float32").reshape(2, 3, 4)
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(host)
            np.testing.assert_allclose(
                float(x.sum().numpy()), float(host.sum()), rtol=1e-6)
            for axis in range(3):
                np.testing.assert_allclose(
                    x.sum(axis).numpy(), host.sum(axis), rtol=1e-6,
                    err_msg="sum over axis %d" % axis)
            np.testing.assert_allclose(
                jt.max(x, 1).numpy(), host.max(1), rtol=1e-6)

    def test_a_one_element_rank_one_tensor_is_not_the_same_case(self):
        """Shape `(1,)` already worked and must keep its own shape.

        Collapsing rank-0 and rank-1-of-length-1 into one path would make this
        return a scalar, which is a different answer from the one PyTorch
        gives.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.array(np.array([VALUE], dtype="float32"))
            self.assertEqual(tuple(x.shape), (1,))
            result = jt.sum(x)
            self.assertAlmostEqual(float(result.item()), VALUE, places=5)


class TestRank0ReductionsCpu(_Rank0Reductions, unittest.TestCase):
    device_flag = 0


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestRank0ReductionsCuda(_Rank0Reductions, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
