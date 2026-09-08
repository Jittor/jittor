# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Gradients of the CUDA softmax kernels, against an exact reference and CPU.

``_softmax_schedule`` picks between a register-resident kernel and a streaming
one, and the register kernel hands its *backward* to the streaming reduction
above 48K columns. That is three distinct backward routes, and the existing
``test_misc_op.py::test_code_softmax`` reaches only the first: its longest row
is 2049 columns, so both streaming routes shipped without any gradient ever
being compared against a reference.

Two references, because one of them cannot be tight:

* float64 NumPy is exact at these sizes, so the tolerance against it is the
  CUDA kernel's own float32 budget -- small, and the assertion has real power.
* The CPU backend is the independent *implementation* check, but it reduces a
  row sequentially while the kernel reduces it with a CUB block tree. On a
  131072-column row of positive cotangents the CPU sum is the less accurate of
  the two by a factor of some thousands, so its bound is the sequential
  n*eps one and it can only catch a gross error. It is still worth asserting:
  it is what fails if the backward formula, rather than its summation order,
  is wrong.

The cotangent is deliberately all-positive. A standard-normal cotangent makes
the row reduction cancel to far below the sum of magnitudes that sets its
rounding error, so every tolerance wide enough for the honest disagreement
between two float32 reductions is also wide enough for a kernel that is wrong
by a percent -- verified: a 0.1% error injected into the reduction still
passed. Positive cotangents keep the reduction well-conditioned.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt

from jittor.backends.cuda.kernels.nn.softmax_cuda import (
    _softmax_schedule, softmax_v1)

EPS32 = float(np.finfo("float32").eps)

# (length, expected schedule, expected backward route)
CASES = (
    (3, "register", "register"),
    (2049, "register", "register"),
    (16384, "register", "register"),
    # Register forward, streaming backward: above 48K columns with ILP 1 the
    # register backward spills, so ``CodeSoftmax.grad`` delegates. 50001 is not
    # a multiple of its 256 threads, which is what holds ILP at 1.
    (50001, "register", "streaming"),
    (65537, "stream", "streaming"),
    (131072, "stream", "streaming"),
)


def _tree_sum_error(terms):
    """Bound on a float32 sum reduced as a tree (CUB's block reduction).

    Pairwise summation combines partials in a tree of depth log2(n) and each
    level contributes at most one rounding, giving log2(n) * eps * sum|terms|.
    """
    return float(np.log2(max(terms.size, 2)) * EPS32 * np.abs(terms).sum())


def _sequential_sum_error(terms):
    """Bound on a float32 sum accumulated one element at a time.

    Every partial total is rounded once, and with positive terms those
    roundings share a sign instead of cancelling, so the bound is the textbook
    n * eps * sum|terms| rather than the tree's log2(n).
    """
    return float(terms.size * EPS32 * np.abs(terms).sum())


def _exact(x_np, cot_np, log):
    """float64 softmax and the exact gradient of (softmax(x) * cot).sum()."""
    x = x_np.astype("float64")
    g = cot_np.astype("float64")
    shifted = np.exp(x - x.max(-1, keepdims=True))
    prob = shifted / shifted.sum(-1, keepdims=True)
    if log:
        return np.log(prob), g - prob * g.sum(-1, keepdims=True)
    return prob, prob * (g - (prob * g).sum(-1, keepdims=True))


def _grad_tolerance(prob, cot_np, grad_ref, log, sum_error):
    """Budget for |computed - exact| on one softmax backward.

    The backward is a row reduction plus an elementwise combination. Only the
    reduction is order-dependent; ``sum_error`` says how accurate it is, and
    the two terms below carry that error to the gradient through the factor
    the formula multiplies it by, and through the softmax probability, which
    was itself produced by a reduction of the same shape.
    """
    depth = np.log2(max(prob.shape[-1], 2))
    if log:
        # dx = g - prob * sum(g)
        reduced = cot_np
        carried = prob * cot_np.sum(-1, keepdims=True)
    else:
        # dx = prob * (g - sum(prob * g))
        reduced = prob * cot_np
        carried = grad_ref
    rows = range(reduced.shape[0])
    from_reduction = float(prob.max()) * max(sum_error(reduced[i]) for i in rows)
    from_probability = depth * EPS32 * float(np.abs(carried).max())
    elementwise = 8.0 * EPS32 * float(np.abs(grad_ref).max())
    return from_reduction + from_probability + elementwise


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
class TestSoftmaxCudaGrad(unittest.TestCase):
    def _cuda(self, x_np, cot_np, log):
        with jt.flag_scope(use_cuda=1):
            x = jt.array(x_np)
            y = softmax_v1(x, log=log)
            self.assertIsNotNone(y, "softmax_v1 declined a shape it schedules")
            grad = jt.grad((y * jt.array(cot_np)).sum(), x)
            return jt.fetch_sync([y, grad])

    def _cpu(self, x_np, cot_np, log):
        with jt.flag_scope(use_cuda=0):
            x = jt.array(x_np)
            y = jt.nn.softmax(x, dim=-1, log=log)
            grad = jt.grad((y * jt.array(cot_np)).sum(), x)
            return jt.fetch_sync([y, grad])

    def test_every_schedule_matches_an_exact_float64_reference(self):
        rng = np.random.RandomState(20260907)
        for length, kind, route in CASES:
            self.assertEqual(
                _softmax_schedule(length)[0], kind,
                "schedule for %d moved; this case no longer covers the %s "
                "kernel" % (length, kind))
            for log in (False, True):
                x_np = rng.randn(2, length).astype("float32")
                cot_np = (rng.rand(2, length) + 0.5).astype("float32")
                y_ref, grad_ref = _exact(x_np, cot_np, log)
                prob = np.exp(y_ref) if log else y_ref
                y_got, grad_got = self._cuda(x_np, cot_np, log)

                forward_tol = (np.log2(length) * EPS32
                               * max(1.0, float(np.abs(y_ref).max())))
                error = float(np.abs(y_got - y_ref).max())
                self.assertLess(
                    error, forward_tol,
                    "softmax forward len=%d log=%s (%s) is off the exact value "
                    "by %.3g > %.3g" % (length, log, kind, error, forward_tol))

                tol = _grad_tolerance(prob, cot_np, grad_ref, log,
                                      _tree_sum_error)
                error = float(np.abs(grad_got - grad_ref).max())
                self.assertLess(
                    error, tol,
                    "softmax backward len=%d log=%s route=%s is off the exact "
                    "gradient by %.3g, past the %.3g a float32 tree reduction "
                    "can account for" % (length, log, route, error, tol))

    def test_every_schedule_matches_the_cpu_backend(self):
        """Same rows against jittor's CPU softmax, the independent route.

        Loose by construction -- see the module docstring: the CPU sum, not
        the kernel, sets this floor.
        """
        rng = np.random.RandomState(1109)
        for length, _, route in CASES:
            for log in (False, True):
                x_np = rng.randn(2, length).astype("float32")
                cot_np = (rng.rand(2, length) + 0.5).astype("float32")
                prob, grad_ref = _exact(x_np, cot_np, log)
                if log:
                    prob = np.exp(prob)
                _, grad_cpu = self._cpu(x_np, cot_np, log)
                _, grad_cuda = self._cuda(x_np, cot_np, log)

                tol = _grad_tolerance(prob, cot_np, grad_ref, log,
                                      _sequential_sum_error)
                error = float(np.abs(grad_cuda - grad_cpu).max())
                self.assertLess(
                    error, tol,
                    "softmax backward len=%d log=%s route=%s disagrees with "
                    "the CPU backend by %.3g > %.3g"
                    % (length, log, route, error, tol))

    def test_half_precision_gradient_tracks_the_float32_reference(self):
        """float16 rows still reduce in float32 inside the kernel.

        The kernel widens on load, so the gradient carries the error of two
        float16 roundings around a float32-accurate value, not that of a
        float16 accumulation.
        """
        rng = np.random.RandomState(31)
        length = 4096
        x_np = rng.randn(2, length).astype("float32")
        cot_np = (rng.rand(2, length) + 0.5).astype("float32")
        _, grad_ref = _exact(x_np, cot_np, False)

        with jt.flag_scope(use_cuda=1):
            x = jt.array(x_np).float16()
            y = softmax_v1(x)
            self.assertIsNotNone(y)
            grad = jt.grad((y.float32() * jt.array(cot_np)).sum(), x)
            grad_got = grad.float32().numpy()

        tol = 4.0 * float(np.finfo("float16").eps) * float(np.abs(grad_ref).max())
        error = float(np.abs(grad_got - grad_ref).max())
        self.assertLess(error, tol,
                        "float16 softmax backward is off by %.3g > %.3g"
                        % (error, tol))


class TestSoftmaxScheduleCoversBothKernels(unittest.TestCase):
    """The case table reaches every backward route. Needs no GPU.

    Without this, a change to ``_softmax_schedule``'s thresholds could route
    all six lengths to the register kernel while the suite above stayed green
    and tested a third of what it names.
    """

    def test_the_table_exercises_register_and_streaming_backward(self):
        self.assertEqual({route for _, _, route in CASES},
                         {"register", "streaming"})
        self.assertEqual({_softmax_schedule(length)[0] for length, _, _ in CASES},
                         {"register", "stream"})

    def test_the_declared_schedule_is_the_real_one(self):
        for length, kind, _ in CASES:
            self.assertEqual(_softmax_schedule(length)[0], kind, length)


if __name__ == "__main__":
    unittest.main()
