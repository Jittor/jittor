# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Generic operator tests driven by ``op_db`` -- modeled on PyTorch's ``test_ops.py``.

One registry (``common_methods_invocations.op_db``), expanded by
``instantiate_device_type_tests`` over every device and dtype, yields the same two
batteries PyTorch runs on every operator:

  * :class:`TestCommon` -- forward output vs the op's INDEPENDENT numpy reference,
    on every device the build has (cpu / cuda / npu). A green test means jittor
    matches numpy, not itself.
  * :class:`TestGradients` -- ``gradcheck`` / ``gradgradcheck`` (numerical vs
    analytical Jacobian) in float64 on CPU. Because the forward is already pinned to
    numpy, a correct gradcheck transitively proves the backward matches torch -- and
    a wrong derivative (stubbed-to-zero, wrong-axis, dropped contribution) fails
    loudly. This is the layer that ends the suite's "forward-only" epidemic: every
    op in the registry gets a backward test for free.

Run::

    python -m pytest tests/ops/test_ops.py                       # all ops, all devices
    python -m pytest tests/ops/test_ops.py -k exp                # just exp
    JITTOR_TEST_DEVICES=cpu python -m pytest tests/ops/test_ops.py
"""
import unittest

import jittor as jt

from _helpers import common as cu
from _helpers.common import JittorTestCase, to_numpy, float64
from _helpers.device_types import (
    instantiate_device_type_tests, ops, OpDTypes, onlyCPU,
)
from opinfo.database import op_db
from _helpers.gradcheck import gradcheck, gradgradcheck, GradcheckError


def _maybe_np(x):
    return to_numpy(x) if isinstance(x, jt.Var) else x


def _is_float_var(v):
    return isinstance(v, jt.Var) and cu.is_floating(str(v.dtype))


def _diff_plan(sample):
    """Return (flat_args, positions_of_differentiable_float_Vars)."""
    flat = [sample.input, *sample.args]
    diff_pos = [i for i, v in enumerate(flat) if _is_float_var(v)]
    return flat, diff_pos


class TestCommon(JittorTestCase):
    """Forward output equals the op's numpy reference, on every device."""

    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_reference(self, device, dtype, op):
        if op.ref is None:
            self.skipTest(f"{op.full_name}: no numpy reference")
        n = 0
        for sample in op.sample_inputs(device, dtype):
            actual = op.op(sample.input, *sample.args, **sample.kwargs)
            # some ops (max/min with dim) return a namedtuple; compare the values
            if hasattr(actual, "values") and not isinstance(actual, jt.Var):
                actual = actual.values
            expected = op.ref(_maybe_np(sample.input),
                              *[_maybe_np(a) for a in sample.args],
                              **sample.kwargs)
            tol = op.reference_tol or (None, None)
            self.assertEqual(actual, expected, atol=tol[0], rtol=tol[1],
                             msg=f"{op.full_name} forward [{device}/{dtype}] sample#{n}")
            n += 1
        self.assertGreater(n, 0, f"{op.full_name}: no samples produced")


class TestGradients(JittorTestCase):
    """Backward correctness via gradcheck/gradgradcheck (float64, CPU)."""

    @onlyCPU
    @ops(op_db, dtypes=OpDTypes.none)
    def test_gradcheck(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest(f"{op.full_name}: supports_autograd=False")
        ran = 0
        for sample in op.sample_inputs(device, float64, requires_grad=True):
            flat, diff_pos = _diff_plan(sample)
            if not diff_pos:
                continue
            diff_inputs = [flat[i] for i in diff_pos]

            def fn(*diffs, flat=flat, diff_pos=diff_pos, sample=sample, op=op):
                full = list(flat)
                for p, v in zip(diff_pos, diffs):
                    full[p] = v
                out = op.op(full[0], *full[1:], **sample.kwargs)
                if hasattr(out, "values") and not isinstance(out, jt.Var):
                    out = out.values
                return out

            try:
                gradcheck(fn, diff_inputs, nondet_tol=op.gradcheck_nondet_tol)
            except GradcheckError as e:
                self.fail(f"{op.full_name} gradcheck failed (sample#{ran}):\n{e}")
            ran += 1
        if ran == 0:
            self.skipTest(f"{op.full_name}: no differentiable samples")

    @onlyCPU
    @ops(op_db, dtypes=OpDTypes.none)
    def test_gradgradcheck(self, device, dtype, op):
        if not op.supports_autograd or not op.supports_gradgrad:
            self.skipTest(f"{op.full_name}: supports_gradgrad=False")
        ran = 0
        for sample in op.sample_inputs(device, float64, requires_grad=True):
            flat, diff_pos = _diff_plan(sample)
            if not diff_pos:
                continue
            diff_inputs = [flat[i] for i in diff_pos]

            def fn(*diffs, flat=flat, diff_pos=diff_pos, sample=sample, op=op):
                full = list(flat)
                for p, v in zip(diff_pos, diffs):
                    full[p] = v
                out = op.op(full[0], *full[1:], **sample.kwargs)
                if hasattr(out, "values") and not isinstance(out, jt.Var):
                    out = out.values
                return out

            try:
                gradgradcheck(fn, diff_inputs, nondet_tol=op.gradcheck_nondet_tol)
            except GradcheckError as e:
                self.fail(f"{op.full_name} gradgradcheck failed (sample#{ran}):\n{e}")
            ran += 1
        if ran == 0:
            self.skipTest(f"{op.full_name}: no differentiable samples")


instantiate_device_type_tests(TestCommon, globals())
instantiate_device_type_tests(TestGradients, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)
