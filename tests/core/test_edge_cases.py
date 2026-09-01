# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Core-op behaviour on PATHOLOGICAL inputs ordinary samples never reach.

``test_ops``/gradcheck exercise well-conditioned, moderately-sized, contiguous
tensors -- exactly the inputs where a stride bug, an empty-dim special case, or an
overflow path stays silent. Training code, by contrast, routinely produces:

  * EMPTY tensors -- a masked-select that selected nothing, a zero-length batch, an
    ``F.cat`` whose first operand is ``zeros(0, C)`` (the transformers padding idiom
    locked in by ``test_torch_compat``'s "cat empty" check). A reduction over an
    empty axis must yield the identity (``sum``->0, not a crash); shapes must track
    numpy.
  * NON-CONTIGUOUS views -- a transpose or a strided slice fed straight into the next
    op. If a kernel assumes C-contiguity, the result is silently wrong *on the
    logical data* while every contiguous test passes.
  * EXTREME magnitudes / non-finite values -- ``exp`` of a large logit, ``inf``
    propagating through ``add``/``mul``, ``clamp`` bounding a runaway value. The
    result must match numpy's IEEE semantics (inf stays inf, inf-inf is nan).
  * Degenerate ranks -- a single-element tensor and a 5-D tensor through the same
    reduce/broadcast machinery the common case uses.

Every expectation here is an INDEPENDENT numpy reference, never jittor-vs-jittor.
Where jittor legitimately differs from numpy by design -- it has no 0-d scalar, so a
full reduction is ``(1,)`` not ``()`` -- the reference is ``np.atleast_1d``'d and the
divergence is spelled out in a comment, not hidden. Differentiable cases also check
the BACKWARD (analytical ``jt.grad`` vs a numpy/analytic gradient); non-differentiable
or index-only cases say so. Anything that genuinely crashes or silently diverges is
marked ``expectedFailure`` with a KNOWN-BUG note rather than deleted.

Run::  python -m pytest tests/core/test_edge_cases.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for,
)


class _EdgeBase(JittorTestCase):
    """Device-loop + grad helpers shared by every edge-case suite below."""

    def _for_devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    def _grad(self, loss, xs):
        g = jt.grad(loss, xs)
        return g if isinstance(g, (list, tuple)) else [g]


# =====================================================================
# (1) EMPTY tensors -- a dimension is 0.
# =====================================================================
class TestEmptyTensors(_EdgeBase):
    """Ops on tensors with a zero-length axis: must not crash, must match numpy.

    numpy defines reductions over an empty axis as the operator identity
    (``sum`` -> 0, ``prod`` -> 1) and elementwise ops as shape-preserving no-ops.
    jittor must agree -- on shape AND value.
    """

    def test_sum_over_empty_axis_is_zero(self):
        # numpy: sum over a length-0 axis is the additive identity 0, for every row.
        x_np = np.zeros((3, 0), dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            got = x.sum(1)                       # reduce the empty axis -> (3,)
            ref = x_np.sum(1)                    # numpy: array of zeros, shape (3,)
            self.assertEqual(got, ref, msg=f"sum over empty axis [{dev}]")
        self._for_devices(body)

    def test_sum_all_empty_is_zero_scalar(self):
        # Full reduction of a wholly-empty tensor. numpy gives a 0-d 0.0; jittor has
        # NO 0-d scalar -> it returns shape (1,). Encode the jittor convention by
        # atleast_1d-ing the numpy reference (documented divergence, not a bug).
        x_np = np.zeros((0,), dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            got = x.sum()                        # jittor: shape (1,), value [0.0]
            ref = np.atleast_1d(x_np.sum())      # numpy 0-d 0.0 -> (1,) for shape parity
            self.assertEqual(got, ref, msg=f"sum-all of empty [{dev}]")
        self._for_devices(body)

    def test_mean_over_empty_axis_is_nan(self):
        # NumPy/Torch define mean over an empty reduction as 0/0 -> NaN.
        def body(dev):
            for dtype in ("float32", "float64"):
                for x_np, dim in (
                    (np.zeros((2, 0), dtype=dtype), 1),
                    (np.zeros((0, 3), dtype=dtype), 0),
                ):
                    got = jt.array(x_np).mean(dim).numpy()
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ref = x_np.mean(dim)
                    self.assertTrue(np.isnan(got).all(),
                                    f"mean over empty axis [{dev}/{dtype}]")
                    self.assertTrue(np.isnan(ref).all(),
                                    f"numpy reference [{dev}/{dtype}]")
                full = jt.array(np.zeros((0,), dtype=dtype)).mean().numpy()
                self.assertEqual(full.shape, (1,))
                self.assertTrue(np.isnan(full).all(), f"mean-all empty [{dev}/{dtype}]")
        self._for_devices(body)

    def test_reshape_empty_preserves_size(self):
        # An empty tensor reshapes freely as long as total size (0) is preserved.
        x_np = np.zeros((0, 4), dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            got = x.reshape(0, 2, 2)             # 0*4 == 0*2*2 == 0
            ref = x_np.reshape(0, 2, 2)
            self.assertEqual(got, ref, msg=f"reshape empty [{dev}]")
        self._for_devices(body)

    def test_cat_with_empty_operand(self):
        # The transformers padding idiom locked in by test_torch_compat's "cat empty":
        # concatenating a (0, C) placeholder with real rows must drop the empty operand
        # and yield exactly the non-empty data (shape (2, 4), identical values).
        empty_np = np.zeros((0, 4), dtype="float32")
        real_np = np.arange(8, dtype="float32").reshape(2, 4)

        def body(dev):
            got = jt.concat([jt.array(empty_np), jt.array(real_np)], dim=0)
            ref = np.concatenate([empty_np, real_np], axis=0)   # == real_np
            self.assertEqual(got, ref, msg=f"cat empty operand [{dev}]")
        self._for_devices(body)

    def test_elementwise_on_empty_is_identity_shape(self):
        # Elementwise on a (0,) tensor must run and stay (0,) -- a kernel that indexes
        # element 0 unconditionally would crash here. Value parity is vacuous (no
        # elements) so the load-bearing assertion is the SHAPE (assertEqual checks it).
        x_np = np.zeros((0,), dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            got = (x * 2.0 + 1.0)
            ref = x_np * 2.0 + 1.0               # still shape (0,)
            self.assertEqual(got, ref, msg=f"elementwise on empty [{dev}]")
        self._for_devices(body)

    def test_empty_backward_is_empty(self):
        # Differentiating through an empty tensor: the grad must exist and be empty,
        # not crash. d/dx sum(x*3) = 3 everywhere, but there are no elements, so the
        # grad is the empty array -- shape (0,), no values.
        x_np = np.zeros((0,), dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            g = self._grad((x * 3.0).sum(), [x])[0]
            self.assertEqual(g, np.full((0,), 3.0, "float32"),
                             msg=f"grad through empty [{dev}]")
        self._for_devices(body)


# =====================================================================
# (2) NON-CONTIGUOUS inputs -- transpose / strided-slice views.
# =====================================================================
class TestNonContiguous(_EdgeBase):
    """Run core ops on a view whose memory layout is NOT C-contiguous.

    A transpose or a step>1 slice changes strides without copying. A kernel that
    assumes contiguity reads the wrong elements -> a result that is wrong on the
    LOGICAL data while every contiguous test stays green. Each check compares jittor
    on the view against numpy on the *same logical view*.
    """

    def test_elementwise_on_transposed_view(self):
        base = np.arange(12, dtype="float32").reshape(3, 4)

        def body(dev):
            xt = jt.array(base).transpose(0, 1)     # (4,3) non-contiguous view
            got = jt.exp(xt)
            ref = np.exp(base.T)                    # numpy on the same logical data
            self.assertEqual(got, ref, atol=1e-5, rtol=1e-5,
                             msg=f"exp on transposed view [{dev}]")
        self._for_devices(body)

    def test_reduce_on_transposed_view(self):
        base = np.arange(24, dtype="float32").reshape(2, 3, 4)

        def body(dev):
            xt = jt.array(base).transpose(2, 0)     # (4,3,2) view, axes permuted
            got = xt.sum(0)                          # reduce the (now-leading) axis
            ref = np.transpose(base, (2, 1, 0)).sum(0)
            self.assertEqual(got, ref, atol=1e-4, rtol=1e-4,
                             msg=f"sum on transposed view [{dev}]")
        self._for_devices(body)

    def test_matmul_on_transposed_views(self):
        a_np = np.arange(6, dtype="float32").reshape(2, 3)
        b_np = np.arange(6, dtype="float32").reshape(2, 3)

        def body(dev):
            # a.T is (3,2) view, b stays (2,3): (3,2)@(2,3) -> (3,3). The lhs operand
            # of the matmul is non-contiguous, which exercises the strided-read path.
            at = jt.array(a_np).transpose(0, 1)
            got = jt.matmul(at, jt.array(b_np))
            ref = a_np.T @ b_np
            self.assertEqual(got, ref, atol=1e-4, rtol=1e-4,
                             msg=f"matmul on transposed lhs [{dev}]")
        self._for_devices(body)

    def test_strided_slice_view(self):
        base = np.arange(20, dtype="float32").reshape(4, 5)

        def body(dev):
            # every-other-column slice -> stride>1, non-contiguous.
            xs = jt.array(base)[:, ::2]              # (4,3) view
            got = (xs * 10.0).sum(1)
            ref = (base[:, ::2] * 10.0).sum(1)
            self.assertEqual(got, ref, atol=1e-4, rtol=1e-4,
                             msg=f"reduce on strided slice [{dev}]")
        self._for_devices(body)

    def test_backward_through_transposed_view(self):
        # Grad must route correctly back through the transpose (the view op must
        # transpose the cotangent on the way back). Closed-form: loss = sum((x.T)^2),
        # d/dx = 2x -- independent of the transpose, which the engine must undo.
        base = np.arange(12, dtype="float32").reshape(3, 4)

        def body(dev):
            x = jt.array(base)
            loss = (x.transpose(0, 1) ** 2).sum()
            g = self._grad(loss, [x])[0]
            self.assertEqual(g, 2 * base, atol=1e-4, rtol=1e-4,
                             msg=f"grad through transposed view [{dev}]")
        self._for_devices(body)


# =====================================================================
# (3) EXTREME values -- huge/tiny magnitudes and +-inf.
# =====================================================================
class TestExtremeValues(_EdgeBase):
    """IEEE-754 edge behaviour: large/small magnitudes and non-finite inputs.

    The reference is numpy's own float32 result, so jittor must reproduce numpy's
    overflow/inf/nan propagation exactly (not merely "be finite").
    """

    def test_exp_overflow_to_inf(self):
        # exp of a large logit overflows to +inf in float32 (~exp(88) is the edge).
        # Match numpy: the big entries become inf, the small ones stay finite.
        x_np = np.array([0.0, 1.0, 100.0, -100.0], dtype="float32")

        def body(dev):
            got = jt.exp(jt.array(x_np))
            with np.errstate(over="ignore"):
                ref = np.exp(x_np)              # [1, e, inf, 0]
            self.assertEqual(got, ref, atol=1e-3, rtol=1e-4,
                             msg=f"exp overflow->inf [{dev}]")
        self._for_devices(body)

    def test_add_inf_propagates(self):
        # inf + finite = inf ; -inf + finite = -inf. A clamping-on-add bug would
        # turn these finite.
        x_np = np.array([np.inf, -np.inf, 1.0, -1.0], dtype="float32")

        def body(dev):
            got = jt.array(x_np) + 1.0
            ref = x_np + 1.0
            self.assertEqual(got, ref, msg=f"add with inf [{dev}]")
        self._for_devices(body)

    def test_inf_minus_inf_is_nan(self):
        # inf - inf = nan (IEEE). Verifies jittor doesn't special-case it to 0.
        a_np = np.array([np.inf, np.inf, 2.0], dtype="float32")
        b_np = np.array([np.inf, 1.0, np.inf], dtype="float32")

        def body(dev):
            got = jt.array(a_np) - jt.array(b_np)
            ref = a_np - b_np                  # [nan, inf, -inf]
            self.assertEqual(got, ref, msg=f"inf - inf == nan [{dev}]")
        self._for_devices(body)

    def test_mul_inf_by_zero_is_nan(self):
        # 0 * inf = nan (IEEE), a classic masking-bug source (a 0 mask times an inf
        # logit must NOT silently become 0).
        x_np = np.array([np.inf, -np.inf, 3.0], dtype="float32")
        z_np = np.array([0.0, 0.0, 0.0], dtype="float32")

        def body(dev):
            got = jt.array(x_np) * jt.array(z_np)
            ref = x_np * z_np                  # [nan, nan, 0]
            self.assertEqual(got, ref, msg=f"inf * 0 == nan [{dev}]")
        self._for_devices(body)

    def test_clamp_bounds_extremes(self):
        # clamp must pull +-inf and huge magnitudes into [lo, hi] exactly like numpy.clip.
        x_np = np.array([np.inf, -np.inf, 1e30, -1e30, 0.5, np.nan], dtype="float32")

        def body(dev):
            got = jt.clamp(jt.array(x_np), -1.0, 1.0)
            ref = np.clip(x_np, -1.0, 1.0)     # inf->1, -inf->-1, 1e30->1, ...
            self.assertEqual(got, ref, msg=f"clamp extremes [{dev}]")
        self._for_devices(body)

    def test_large_magnitude_add_precision(self):
        # Each scalar float32 add/subtract step follows strict IEEE float32 semantics.
        big = np.float32(1e8)
        x_np = np.array([big, 1.0, -big], dtype="float32")

        def body(dev):
            got = (jt.array(x_np) + big) - big
            ref = (x_np + big) - big           # strict float32: the 1.0 is lost -> 0
            self.assertEqual(got, ref, atol=0.0, rtol=0.0,
                             msg=f"float32 large-add precision [{dev}]")
        self._for_devices(body)

    def test_clamp_backward_passes_gradient_in_range(self):
        # PyTorch routes the input gradient through both exact clamp boundaries.
        x_np = np.array(
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype="float32"
        )

        def body(dev):
            x = jt.array(x_np)
            loss = jt.clamp(x, -1.0, 1.0).sum()
            g = self._grad(loss, [x])[0]
            ref = ((x_np >= -1.0) & (x_np <= 1.0)).astype("float32")
            self.assertEqual(g, ref, msg=f"clamp backward mask [{dev}]")
        self._for_devices(body)

    def test_clamp_reversed_bounds_use_upper_bound(self):
        x_np = np.array([-2.0, 0.0, 2.0], dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            output = jt.clamp(x, 1.0, -1.0)
            gradient = self._grad(output.sum(), [x])[0]
            self.assertEqual(
                output,
                np.full_like(x_np, -1.0),
                msg=f"clamp reversed forward [{dev}]",
            )
            self.assertEqual(
                gradient,
                np.zeros_like(x_np),
                msg=f"clamp reversed backward [{dev}]",
            )

        self._for_devices(body)

    def test_clamp_low_precision_scalar_bounds(self):
        x_np = np.array([-2.0, -0.5, 0.0, 1.0, 2.0], dtype="float32")

        def body(dev):
            for dtype in ("float16", "bfloat16"):
                x = jt.array(x_np).cast(dtype)
                for lower, upper in ((-1, 1), (-0.5, 1.5)):
                    output = jt.clamp(x, lower, upper)
                    self.assertEqual(
                        str(output.dtype),
                        dtype,
                        msg=f"clamp {dtype} dtype [{dev}]",
                    )
                    self.assertEqual(
                        output.float32(),
                        np.clip(x_np, lower, upper),
                        msg=f"clamp {dtype} values [{dev}]",
                    )

        self._for_devices(body)

    def test_clamp_integer_float_bounds_promote_float32(self):
        x_np = np.array([-2, -1, 0, 1, 2], dtype="int64")

        def body(dev):
            output = jt.clamp(jt.array(x_np), -0.5, 1.5)
            self.assertEqual(
                str(output.dtype),
                "float32",
                msg=f"clamp integer promotion [{dev}]",
            )
            self.assertEqual(
                output,
                np.clip(x_np, -0.5, 1.5).astype("float32"),
                msg=f"clamp integer values [{dev}]",
            )

        self._for_devices(body)


# =====================================================================
# (4) Degenerate ranks -- single element and high-rank (5-D).
# =====================================================================
class TestDegenerateRanks(_EdgeBase):
    """A single-element tensor and a 5-D tensor through reduce/broadcast.

    The reduce and broadcast machinery is rank-generic; these pin the two ends of the
    rank spectrum the common (2-D/3-D) tests skip.
    """

    def test_single_element_reduce(self):
        # A 1-element tensor reduced to a scalar. jittor keeps it (1,) (no 0-d), so
        # atleast_1d the numpy (1,)->() reduction for shape parity. Value is identity.
        x_np = np.array([3.5], dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            got = x.sum()
            ref = np.atleast_1d(x_np.sum())    # numpy () 3.5 -> (1,) [3.5]
            self.assertEqual(got, ref, msg=f"single-element reduce [{dev}]")
        self._for_devices(body)

    def test_single_element_backward(self):
        # Backward through a 1-element tensor: d/dx (x^2) = 2x at the single point.
        x_np = np.array([4.0], dtype="float32")

        def body(dev):
            x = jt.array(x_np)
            g = self._grad((x * x).sum(), [x])[0]
            self.assertEqual(g, 2 * x_np, atol=1e-5,
                             msg=f"single-element backward [{dev}]")
        self._for_devices(body)

    def test_5d_reduce_over_middle_axis(self):
        # 5-D reduction over an interior axis -- exercises the rank-generic reduce
        # index arithmetic at the high end.
        rng = np.random.RandomState(0)
        x_np = rng.randn(2, 3, 4, 2, 3).astype("float32")

        def body(dev):
            x = jt.array(x_np)
            got = x.sum(2)                      # (2,3,2,3)
            ref = x_np.sum(2)
            self.assertEqual(got, ref, atol=1e-3, rtol=1e-4,
                             msg=f"5-D reduce mid-axis [{dev}]")
        self._for_devices(body)

    def test_5d_broadcast_add(self):
        # Broadcast a 1-D bias across a 5-D tensor along the last axis (the canonical
        # channel-bias pattern, at rank 5).
        rng = np.random.RandomState(1)
        x_np = rng.randn(2, 2, 2, 2, 4).astype("float32")
        b_np = rng.randn(4).astype("float32")

        def body(dev):
            got = jt.array(x_np) + jt.array(b_np)
            ref = x_np + b_np                   # numpy broadcasts the trailing 4
            self.assertEqual(got, ref, atol=1e-5, rtol=1e-5,
                             msg=f"5-D broadcast add [{dev}]")
        self._for_devices(body)

    def test_5d_broadcast_backward_sums_over_broadcast_axes(self):
        # The grad wrt a broadcast 1-D operand must SUM over every axis it was
        # broadcast along -- the classic broadcast-back trap, at rank 5. For
        # loss = sum(x + b), d/db_j = (number of positions sharing channel j) = 2*2*2*2.
        rng = np.random.RandomState(2)
        x_np = rng.randn(2, 2, 2, 2, 4).astype("float32")
        b_np = rng.randn(4).astype("float32")

        def body(dev):
            b = jt.array(b_np)
            loss = (jt.array(x_np) + b).sum()
            g = self._grad(loss, [b])[0]
            ref = np.full(4, 2 * 2 * 2 * 2, "float32")
            self.assertEqual(g, ref, atol=1e-4, rtol=1e-4,
                             msg=f"5-D broadcast-back grad [{dev}]")
        self._for_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
