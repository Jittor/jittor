# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``maximum``/``minimum`` and ``max()``/``min()`` answer NaN the way NumPy does.

NumPy defines both operators exactly::

    maximum(a, b) = (a > b || isnan(a)) ? a : b
    minimum(a, b) = (a < b || isnan(a)) ? a : b

Two consequences are asserted here and neither is a matter of taste. A NaN in
either operand comes out, and the sign of a zero is decided by ``>``/``<``
alone, which makes the operators *order dependent* on signed zeros:
``maximum(-0.0, 0.0)`` is ``+0.0`` and ``maximum(0.0, -0.0)`` is ``-0.0``.
NumPy is the oracle for both, and it is queried on the same operands here
rather than quoted from memory.

What used to happen (KI-BACKEND-004, KI-OPS-006):

* CPU lowered ``maximum`` to ``std::max(a, b)``, which is ``a < b ? b : a``.
  Every comparison against NaN is false, so the *first* operand came back --
  the NaN survived by accident when it happened to be written first, and was
  discarded when it was written second. The same expression returned ``-0.0``
  for ``maximum(-0.0, 0.0)`` where NumPy returns ``+0.0``.
* CUDA lowered it to ``::max``, which is ``fmaxf``: IEEE ``maxNum``, which
  deliberately returns the operand that is *not* NaN. So the same expression
  on the same data disagreed between the two devices.
* The reduction was worse than either. It folds ``acc = max(acc, x)``, so an
  incoming NaN is always the *second* operand and was dropped on both devices
  at every size. ``x.max()`` is a common way to ask whether a tensor has gone
  bad, and it could not see a NaN at all.

``sum``/``mean``/``prod`` are asserted alongside as the control: IEEE addition
and multiplication propagate NaN in hardware, with no comparison involved, so
they were never affected. They are what made this a divergence *inside* one
operator family rather than a global policy.

The sizes are not decoration. 5 is below any vector width, 4096 is one
parallel block, and 1<<20 is large enough that the CPU reduction runs threaded
and the CUDA one folds through ``cuda_atomic_max``/``min`` -- three different
code paths that each have to keep the NaN. Both NaN signs are fed for the same
reason: the CUDA atomic encodes a float as an ordered integer, in which a
positive NaN sorts above ``+inf`` and a negative NaN below ``-inf``, so one
sign is carried by ``atomicMax`` and the other only by ``atomicMin``.

Integers are asserted unchanged. ``a != a`` is constant-false for them, so one
template serves both, and the point of the assertion is that nothing about
integer max/min moved.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: One NaN in the middle, so neither a first-element nor a last-element special
#: case can hide the answer.
WITH_NAN = np.array([1.0, np.nan, 2.0, -3.0], dtype="float32")

INF = float("inf")
NAN = float("nan")

#: Every float class that has an opinion about max/min, in one vector.
SPECIAL = (NAN, -INF, -0.0, 0.0, INF)

#: Below a vector register, one parallel block, and past the threshold where
#: the reduction is threaded on CPU and atomic on CUDA.
SIZES = (5, 4096, 1 << 20)


def _var(host):
    """A Var whose dtype is exactly the dtype of ``host``.

    KI-DTYPE-002: ``jt.array`` narrows 64-bit NumPy input to 32 bits by
    default. Left alone, the float64 cases below would quietly become second
    float32 cases and the int64 one would test truncated int32 extremes, so the
    width is asserted here rather than assumed.
    """
    with jt.flag_scope(auto_convert_64_to_32=0):
        v = jt.array(host)
    assert str(v.dtype) == str(host.dtype), (v.dtype, host.dtype)
    return v


def _assert_matches_numpy(got, want, what):
    """Equal to NumPy including the sign of zero, with NaN matched positionally.

    ``assert_array_equal`` already treats NaN as equal to NaN and ``0.0`` as
    equal to ``-0.0``; the second is exactly one of the two defects here, so the
    signs are compared separately and cannot pass by omission. NaN *payloads*
    are not compared: both implementations return one of their operands, so the
    payload carries no information the sign and the position do not.
    """
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.dtype == want.dtype, "%s: dtype %s vs %s" % (what, got.dtype, want.dtype)
    got_nan = np.isnan(got)
    want_nan = np.isnan(want)
    np.testing.assert_array_equal(
        got_nan, want_nan,
        err_msg="%s: NaN in different places\n  got  %s\n  want %s" % (what, got, want))
    np.testing.assert_array_equal(
        got[~got_nan], want[~want_nan],
        err_msg="%s: values differ\n  got  %s\n  want %s" % (what, got, want))
    np.testing.assert_array_equal(
        np.signbit(got[~got_nan]), np.signbit(want[~want_nan]),
        err_msg="%s: sign of zero differs\n  got  %s\n  want %s" % (what, got, want))


class _NanPropagationContract:

    device_flag = 0

    # ---- controls ---------------------------------------------------------

    def test_sum_mean_and_prod_propagate_nan(self):
        # The control: these reductions were always correct, which is what made
        # max/min a divergence inside one operator family.
        with jt.flag_scope(use_cuda=self.device_flag):
            x = _var(WITH_NAN)
            self.assertEqual(str(x.dtype), "float32")
            self.assertTrue(np.isnan(float(x.sum().numpy())))
            self.assertTrue(np.isnan(float(x.mean().numpy())))
            self.assertTrue(np.isnan(float(jt.prod(x).numpy())))

    def test_max_and_min_without_nan_are_correct(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([1.0, 5.0, 2.0, -3.0], dtype="float32")
            x = _var(raw)
            self.assertEqual(float(x.max().numpy()), float(raw.max()))
            self.assertEqual(float(x.min().numpy()), float(raw.min()))

    # ---- reductions -------------------------------------------------------

    def test_max_and_min_reductions_propagate_nan(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            x = _var(WITH_NAN)
            self.assertEqual(str(x.dtype), "float32")
            self.assertTrue(np.isnan(float(x.max().numpy())),
                            "max() dropped the NaN")
            self.assertTrue(np.isnan(float(x.min().numpy())),
                            "min() dropped the NaN")

    def test_reductions_propagate_nan_at_every_size_and_sign(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for n in SIZES:
                for sign, label in ((1.0, "+nan"), (-1.0, "-nan")):
                    host = np.ones(n, dtype="float32")
                    host[n // 2] = np.float32(sign * NAN)
                    x = _var(host)
                    for name in ("max", "min"):
                        with self.subTest(n=n, nan=label, reduction=name):
                            got = float(getattr(x, name)().numpy())
                            want = float(getattr(np, name)(host))
                            self.assertTrue(
                                np.isnan(want), "the oracle stopped being one")
                            self.assertTrue(
                                np.isnan(got),
                                "%s over %d elements with a %s returned %r"
                                % (name, n, label, got))

    def test_reduction_along_a_dim_propagates_nan(self):
        # The dim reduction is the parallel/atomic path: many outputs, each
        # folded by more than one thread. It is not the kernel the whole-tensor
        # reduce above compiles to.
        with jt.flag_scope(use_cuda=self.device_flag):
            host = np.ones((64, 4096), dtype="float32")
            host[7, 100] = NAN
            host[63, 4095] = -NAN
            x = _var(host)
            for name in ("max", "min"):
                with self.subTest(reduction=name):
                    got = getattr(jt, name)(x, dim=1).numpy()
                    want = getattr(np, name)(host, axis=1)
                    _assert_matches_numpy(got, want, "%s(dim=1)" % name)

    def test_float64_reduction_propagates_nan(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            host = np.ones(4096, dtype="float64")
            host[3] = NAN
            x = _var(host)
            self.assertEqual(str(x.dtype), "float64")
            self.assertTrue(np.isnan(float(x.max().numpy())))
            self.assertTrue(np.isnan(float(x.min().numpy())))

    # ---- elementwise ------------------------------------------------------

    def test_elementwise_maximum_and_minimum_propagate_nan(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            a = np.array([np.nan, 1.0, np.nan], dtype="float32")
            b = np.array([1.0, np.nan, np.nan], dtype="float32")
            got_max = jt.maximum(_var(a), _var(b)).numpy()
            got_min = jt.minimum(_var(a), _var(b)).numpy()
            np.testing.assert_array_equal(np.isnan(got_max), np.isnan(np.maximum(a, b)))
            np.testing.assert_array_equal(np.isnan(got_min), np.isnan(np.minimum(a, b)))

    def test_elementwise_matches_numpy_on_every_float_class(self):
        # Both operand orders: the CPU defect returned whichever operand came
        # first, so a single order would have passed by accident.
        for dtype in ("float32", "float64"):
            special = np.array(SPECIAL, dtype=dtype)
            zeros = np.zeros(len(SPECIAL), dtype=dtype)
            for name in ("maximum", "minimum"):
                for left, right, order in ((special, zeros, "special,zeros"),
                                           (zeros, special, "zeros,special")):
                    with self.subTest(dtype=dtype, op=name, order=order):
                        with jt.flag_scope(use_cuda=self.device_flag):
                            got = getattr(jt, name)(
                                _var(left), _var(right)).numpy()
                        want = getattr(np, name)(left, right)
                        _assert_matches_numpy(
                            got, want, "%s(%s) %s" % (name, order, dtype))

    def test_signed_zero_follows_numpy_and_is_order_dependent(self):
        # NumPy decides this with `>` alone, so it is order dependent and the
        # two orders give different answers. Asserted against NumPy computed
        # here, not against a remembered convention.
        for dtype in ("float32", "float64"):
            for name in ("maximum", "minimum"):
                for left, right in ((-0.0, 0.0), (0.0, -0.0)):
                    with self.subTest(dtype=dtype, op=name, left=left, right=right):
                        a = np.array([left], dtype=dtype)
                        b = np.array([right], dtype=dtype)
                        with jt.flag_scope(use_cuda=self.device_flag):
                            got = getattr(jt, name)(_var(a), _var(b)).numpy()
                        want = getattr(np, name)(a, b)
                        _assert_matches_numpy(
                            got, want, "%s(%r, %r) %s" % (name, left, right, dtype))

    # ---- integers must not move ------------------------------------------

    def test_integer_max_and_min_are_unchanged(self):
        for dtype in ("int8", "int16", "int32", "int64", "uint8"):
            info = np.iinfo(dtype)
            host_a = np.array([3, info.min, 0, info.max, 7], dtype=dtype)
            host_b = np.array([1, info.max, info.min, 9, 7], dtype=dtype)
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=self.device_flag):
                    a, b = _var(host_a), _var(host_b)
                    got_max = jt.maximum(a, b).numpy()
                    got_min = jt.minimum(a, b).numpy()
                    red_max = jt.max(a).numpy()
                    red_min = jt.min(a).numpy()
                np.testing.assert_array_equal(got_max, np.maximum(host_a, host_b))
                np.testing.assert_array_equal(got_min, np.minimum(host_a, host_b))
                np.testing.assert_array_equal(red_max, host_a.max())
                np.testing.assert_array_equal(red_min, host_a.min())


class TestMinMaxNanPropagationCpu(_NanPropagationContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestMinMaxNanPropagationCuda(_NanPropagationContract, unittest.TestCase):
    device_flag = 1


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestMinMaxDeviceParity(unittest.TestCase):
    """The device-parity case KI-BACKEND-004 asked for.

    The two backends lowered ``maximum`` through different functions and the
    disagreement was invisible because no parity case fed a NaN. This is that
    case: the same operands on both devices, compared to each other as well as
    to NumPy, so a future divergence cannot pass unnoticed even if each device
    looks individually plausible.
    """

    def _both_devices(self, run):
        with jt.flag_scope(use_cuda=0):
            cpu = run()
        with jt.flag_scope(use_cuda=1):
            cuda = run()
        return cpu, cuda

    def test_elementwise_agrees_across_devices_and_with_numpy(self):
        special = np.array(SPECIAL, dtype="float32")
        zeros = np.zeros(len(SPECIAL), dtype="float32")
        for name in ("maximum", "minimum"):
            for left, right, order in ((special, zeros, "special,zeros"),
                                       (zeros, special, "zeros,special")):
                with self.subTest(op=name, order=order):
                    cpu, cuda = self._both_devices(
                        lambda: getattr(jt, name)(
                            _var(left), _var(right)).numpy())
                    want = getattr(np, name)(left, right)
                    _assert_matches_numpy(cpu, want, "cpu %s %s" % (name, order))
                    _assert_matches_numpy(cuda, want, "cuda %s %s" % (name, order))
                    _assert_matches_numpy(cuda, cpu, "cuda vs cpu %s %s" % (name, order))

    def test_reductions_agree_across_devices_and_with_numpy(self):
        for n in SIZES:
            for sign, label in ((1.0, "+nan"), (-1.0, "-nan")):
                host = np.ones(n, dtype="float32")
                host[n // 2] = np.float32(sign * NAN)
                for name in ("max", "min"):
                    with self.subTest(n=n, nan=label, reduction=name):
                        cpu, cuda = self._both_devices(
                            lambda: getattr(jt, name)(_var(host)).numpy())
                        self.assertTrue(np.isnan(float(np.asarray(cpu))),
                                        "cpu %s n=%d %s gave %r" % (name, n, label, cpu))
                        self.assertTrue(np.isnan(float(np.asarray(cuda))),
                                        "cuda %s n=%d %s gave %r" % (name, n, label, cuda))


if __name__ == "__main__":
    unittest.main()
