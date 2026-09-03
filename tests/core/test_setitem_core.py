# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Low-level setitem / scatter KERNEL correctness -- forward AND backward, CPU+CUDA.

This is the highest-risk core kernel in the project: setitem (the write side of
``x[idx] = v``) is also the BACKWARD of getitem, so a bug here silently corrupts the
gradient of every advanced-indexing read.  Several of the audit's §4 "silent-wrong"
bugs lived exactly here, and a generic op_db gradcheck cannot reach them because
setitem mutates in place and returns ``None`` -- it is not an OpInfo op.  Hence this
dedicated ``JittorTestCase`` module, one named check per failure mode, every forward
against an INDEPENDENT numpy reference and every differentiable backward against the
analytic scatter of the cotangent (never jittor-vs-jittor), looped over every device
the build can run (``get_all_device_types`` / ``use_cuda_for``).

Regression locks embedded here (each constructed to FAIL on the buggy behavior):

  * ``58e95b73`` -- negative *advanced* index in setitem was NOT normalized, so the
    backward of ``x[..., [-2], :]`` (= a setitem of the upstream grad) scattered to a
    negative row, outside the buffer: indexed rows got no grad and stray writes
    corrupted memory (presented as "non-deterministic" wrong grads).  Locked by
    ``test_neg_advanced_index_backward`` (grad wrt v AND wrt x).
  * ``880cd6ad`` -- on CUDA the setitem reduce=max/min/multiply RMW was non-atomic, so
    DUPLICATE indices colliding on one output cell dropped all but the last
    contribution (last-writer-wins).  Locked by the duplicate-index max/min/multiply
    forward checks and by ``test_index_add_duplicate_accumulate`` (the add-path dup
    case from the same family).

Backward coverage policy (honest, per the native kernel ``SetitemOp::grad``):
  - reduce in {void(overwrite), add, multiply} HAVE an analytic kernel backward -> tested.
  - reduce in {max, min} have NO setitem backward (the kernel ``LOGf``-fatals); they are
    FORWARD-ONLY here, deliberately, and marked as such.  (The differentiable torch path
    for segment max/min is ``scatter_reduce``/``reindex_reduce``, a different kernel
    covered elsewhere.)
  - overwrite (reduce=void / plain setitem) backward is tested with DISTINCT destinations
    only: torch leaves duplicate-overwrite order (and hence which src gets the grad)
    unspecified, so a dup-overwrite backward has no well-defined analytic reference.

Run::  python -m pytest tests/core/test_setitem_core.py
       python -m pytest tests/core/test_setitem_core.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.assertions import expect_error
from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for,
)


# ---------------------------------------------------------------------------
# Independent numpy references (the "gold" side: forward value AND analytic
# backward of ``loss = out.sum()`` -- i.e. the cotangent is all-ones).
# ---------------------------------------------------------------------------
def np_setitem(x, key, v):
    """Out-of-place ``y = x.copy(); y[key] = v`` -> y (forward reference)."""
    y = x.copy()
    y[key] = v
    return y


def np_setitem_grad_overwrite(x, key, v_shape):
    """Analytic grad of ``sum(setitem(x, key, v))`` for an OVERWRITE (reduce=void).

    Returns ``(gx, gv)`` for the all-ones cotangent:
      * gx = 1 everywhere the value survives, 0 where v overwrote it.
      * gv = the all-ones cotangent gathered at the written positions, reduced
             (summed) onto v's (possibly broadcast) shape.
    Requires DISTINCT destinations -- overwrite-dup backward is order-unspecified.
    """
    cot = np.ones_like(x)
    gx = cot.copy()
    gx[key] = 0.0                                  # overwritten cells contribute nothing to x
    # grad wrt v = cotangent read at the written region, then sum-reduced to v_shape
    gv_full = cot[key]                             # shape of the indexed region
    gv = _reduce_to_shape(gv_full, v_shape)
    return gx, gv


def np_scatter_add_grad(x_shape, dim, idx, src_shape):
    """Analytic grad of ``sum(scatter_add(x, dim, idx, src))`` (all-ones cotangent).

    add is linear: gx = ones(x_shape); each src element's grad = cotangent at its
    destination cell = 1 (so gsrc = ones(src_shape)).  Holds even with DUPLICATE
    indices, since every contribution is independent and additive.
    """
    return np.ones(x_shape, "float64"), np.ones(src_shape, "float64")


def np_index_add(base, dim, index, source, alpha=1.0):
    """torch index_add reference: out[.., index[k], ..] += alpha*source[.., k, ..],
    ACCUMULATING duplicate indices (the torch contract)."""
    out = base.astype(np.float64).copy()
    src = source.astype(np.float64) * alpha
    for k, j in enumerate(index):
        sl_dst = [slice(None)] * out.ndim; sl_dst[dim] = int(j)
        sl_src = [slice(None)] * src.ndim; sl_src[dim] = k
        out[tuple(sl_dst)] += src[tuple(sl_src)]
    return out


def np_scatter_reduce_forward(base, dim, idx, src, op):
    """torch scatter-with-reduce forward over a 2-D index/src (op in max/min/multiply),
    include_self=True semantics (self participates).  Used for the dup-collision
    FORWARD checks (these reduces have no kernel backward)."""
    out = base.astype(np.float64).copy()
    it = np.nditer(idx, flags=["multi_index"])
    for v in it:
        mi = it.multi_index
        dst = list(mi); dst[dim] = int(v)
        val = float(src[mi])
        cell = tuple(dst)
        if op == "max":
            out[cell] = max(out[cell], val)
        elif op == "min":
            out[cell] = min(out[cell], val)
        elif op == "multiply":
            out[cell] = out[cell] * val
        else:
            raise ValueError(op)
    return out


def _reduce_to_shape(arr, shape):
    """Sum ``arr`` down (numpy broadcasting in reverse) to ``shape`` -- the grad of a
    broadcast is a sum over the broadcast axes."""
    arr = np.asarray(arr, dtype="float64")
    shape = tuple(int(s) for s in shape)
    if arr.shape == shape:
        return arr
    # align ranks (broadcasting pads on the left)
    while arr.ndim > len(shape):
        arr = arr.sum(axis=0)
    out = arr
    for ax, s in enumerate(shape):
        if out.shape[ax] != s:
            out = out.sum(axis=ax, keepdims=True)
    return out.reshape(shape)


class _SetitemCore(JittorTestCase):
    """Shared device loop + small backward helper for the whole module."""

    def _for_devices(self, body):
        for d in get_all_device_types():
            with self.subTest(device=d):
                with jt.flag_scope(use_cuda=use_cuda_for(d)):
                    body(d)

    @staticmethod
    def _f32(a):
        """np array -> float32 jittor Var (a differentiable leaf)."""
        return jt.array(np.ascontiguousarray(a).astype("float32"), dtype="float32")

    @staticmethod
    def _i64(a):
        """np array -> int64 jittor Var (a NON-differentiable index/mask carrier)."""
        return jt.array(np.ascontiguousarray(a))


class TestSetitemShapeErrors(unittest.TestCase):
    def test_data_dimension_mismatch_is_a_catchable_user_error(self):
        value = jt.zeros((2, 3))
        expect_error(
            lambda: value.__setitem__(slice(None), jt.ones((1, 2, 3))),
            exc_type=RuntimeError,
            match="Data dimension not match",
        )

    def test_data_shape_mismatch_is_a_catchable_user_error(self):
        value = jt.zeros((2, 3))
        expect_error(
            lambda: value.__setitem__(slice(None), jt.ones((2, 4))),
            exc_type=RuntimeError,
            match="Data shape not match",
        )


# ===========================================================================
# 1. setitem by SLICE / INT / MASK / FANCY index  (overwrite; forward+backward)
# ===========================================================================
class TestSetitemOverwrite(_SetitemCore):
    """``x[key] = v`` overwrite, out-of-place via ``x.clone()``.  Distinct dst only,
    so the overwrite backward has a well-defined analytic reference."""

    def _check(self, x_np, key, v_np, label):
        ref_fwd = np_setitem(x_np, key, v_np)
        gx_ref, gv_ref = np_setitem_grad_overwrite(x_np, key, v_np.shape)

        def body(dev):
            x = self._f32(x_np)
            v = self._f32(v_np)
            y = x.clone()
            y[key] = v                              # in-place write into the clone
            self.assertEqual(y, ref_fwd, msg=f"{label} fwd [{dev}]")
            gx, gv = jt.grad(y.sum(), [x, v])
            self.assertEqual(gx, gx_ref, msg=f"{label} d/dx [{dev}]")
            self.assertEqual(gv, gv_ref, msg=f"{label} d/dv [{dev}]")
        self._for_devices(body)

    def test_slice(self):
        # x[1:3] = v over the leading dim
        x = np.arange(20, dtype="float32").reshape(5, 4)
        v = (np.arange(8, dtype="float32") + 100).reshape(2, 4)
        self._check(x, slice(1, 3), v, "setitem slice")

    def test_int_index(self):
        # x[2] = v -- single integer index drops the indexed dim on the value.
        x = np.arange(20, dtype="float32").reshape(5, 4)
        v = (np.arange(4, dtype="float32") + 50)
        self._check(x, 2, v, "setitem int")

    def test_boolean_mask(self):
        # x[mask] = v -- boolean mask routes through .where() (advanced index of the
        # True positions); 3 Trues, v length 3 (distinct flat positions).
        x = np.arange(12, dtype="float32").reshape(3, 4)
        mask = np.array([[True, False, False, True],
                         [False, False, True, False],
                         [False, False, False, False]])
        v = np.array([-1., -2., -3.], dtype="float32")
        ref_fwd = np_setitem(x, mask, v)
        cot = np.ones_like(x); cot[mask] = 0.0      # gx: masked cells zeroed
        gv_ref = np.ones(3, "float64")              # each written cell feeds 1 to its v elem

        def body(dev):
            xv = self._f32(x)
            vv = self._f32(v)
            mv = jt.array(mask)                      # bool Var (non-diff)
            y = xv.clone()
            y[mv] = vv
            self.assertEqual(y, ref_fwd, msg=f"setitem bool-mask fwd [{dev}]")
            gx, gv = jt.grad(y.sum(), [xv, vv])
            self.assertEqual(gx, cot, msg=f"setitem bool-mask d/dx [{dev}]")
            self.assertEqual(gv, gv_ref, msg=f"setitem bool-mask d/dv [{dev}]")
        self._for_devices(body)

    def test_fancy_index(self):
        # x[[0, 2, 3]] = v -- fancy (advanced) integer index along dim 0, distinct rows.
        x = np.arange(20, dtype="float32").reshape(5, 4)
        v = (np.arange(12, dtype="float32") + 70).reshape(3, 4)
        idx = np.array([0, 2, 3], dtype="int64")
        ref_fwd = x.copy(); ref_fwd[idx] = v
        cot = np.ones_like(x); cot[idx] = 0.0
        gv_ref = np.ones((3, 4), "float64")

        def body(dev):
            xv = self._f32(x)
            vv = self._f32(v)
            iv = self._i64(idx)
            y = xv.clone()
            y[iv] = vv
            self.assertEqual(y, ref_fwd, msg=f"setitem fancy fwd [{dev}]")
            gx, gv = jt.grad(y.sum(), [xv, vv])
            self.assertEqual(gx, cot, msg=f"setitem fancy d/dx [{dev}]")
            self.assertEqual(gv, gv_ref, msg=f"setitem fancy d/dv [{dev}]")
        self._for_devices(body)

    def test_scalar_broadcast_value(self):
        # x[0:] = 1.0 -- python scalar broadcast (the test_setitem inplace_case3 shape).
        # Scalar is non-differentiable; only d/dx is asserted (all-zero: every cell
        # is overwritten).
        x = np.zeros((3,), dtype="float32")
        ref_fwd = x.copy(); ref_fwd[0:] = 1.0

        def body(dev):
            xv = self._f32(x)
            y = xv.clone()
            y[0:] = 1.0
            self.assertEqual(y, ref_fwd, msg=f"setitem scalar-bcast fwd [{dev}]")
            gx = jt.grad(y.sum(), [xv])[0]
            self.assertEqual(gx, np.zeros_like(x), msg=f"setitem scalar-bcast d/dx [{dev}]")
        self._for_devices(body)


# ===========================================================================
# 2. NEGATIVE-index backward  (regression lock: 58e95b73)
# ===========================================================================
class TestNegativeIndexBackward(_SetitemCore):
    """The 58e95b73 backward-corruption case: a negative advanced index into setitem
    (= the backward of a negative-index getitem) must normalize to a valid row, NOT
    scatter to a negative offset (which corrupted memory and dropped the grad)."""

    def test_neg_slice_setitem_backward(self):
        # x[-2:] = v then grad wrt v and wrt x.  Negative-slice ints are host-normalized
        # in infer_slices, but this still guards the negative-slice setitem backward path.
        x = np.arange(20, dtype="float32").reshape(5, 4)
        v = (np.arange(8, dtype="float32") + 200).reshape(2, 4)
        ref_fwd = x.copy(); ref_fwd[-2:] = v
        cot = np.ones_like(x); cot[-2:] = 0.0
        gv_ref = np.ones((2, 4), "float64")

        def body(dev):
            xv = self._f32(x)
            vv = self._f32(v)
            y = xv.clone()
            y[-2:] = vv
            self.assertEqual(y, ref_fwd, msg=f"x[-2:]=v fwd [{dev}]")
            gx, gv = jt.grad(y.sum(), [xv, vv])
            self.assertEqual(gx, cot, msg=f"x[-2:]=v d/dx [{dev}]")
            self.assertEqual(gv, gv_ref, msg=f"x[-2:]=v d/dv [{dev}]")
        self._for_devices(body)

    def test_neg_advanced_index_backward(self):
        # THE 58e95b73 case: a negative VAR/list advanced index reaches the kernel
        # un-normalized.  falcon _split_heads does fused[..., [-2], :] / [..., [-1], :];
        # its backward (a setitem of the upstream grad at [-2]/[-1]) was ~4.6e-2 off
        # because the write went to row -2 of the buffer.  Test both forward and the
        # grad wrt v and wrt x, against the POSITIVE-index analytic equivalent.
        x = np.arange(24, dtype="float32").reshape(4, 3, 2)
        v = (np.arange(6, dtype="float32") + 300).reshape(1, 3, 2)

        # write along dim 0 with a length-1 advanced index list [neg] (the falcon pattern)
        def make_ref(neg):
            ref = x.copy()
            ref[[neg], :, :] = v[0]
            cot = np.ones_like(x); cot[[neg], :, :] = 0.0
            return ref, cot

        def body(dev):
            for neg in (-2, -1):
                ref_fwd, gx_ref = make_ref(neg)
                xv = self._f32(x)
                vv = self._f32(v)
                iv = self._i64(np.array([neg], dtype="int64"))   # negative ADVANCED index
                y = xv.clone()
                y[iv, :, :] = vv
                self.assertEqual(y, ref_fwd, msg=f"x[[{neg}],:,:]=v fwd [{dev}]")
                gx, gv = jt.grad(y.sum(), [xv, vv])
                self.assertEqual(gx, gx_ref, msg=f"x[[{neg}],:,:]=v d/dx [{dev}]")
                # v is broadcast (1,3,2) over the (1,3,2) written region -> grad all ones
                self.assertEqual(gv, np.ones((1, 3, 2), "float64"),
                                 msg=f"x[[{neg}],:,:]=v d/dv [{dev}]")
        self._for_devices(body)


# ===========================================================================
# 3. scatter_ / scatter_add  with reduce=add  (forward + backward, DUP-correct)
# ===========================================================================
class TestScatterAddReduce(_SetitemCore):
    """reduce='add' has a real kernel backward (linear): gx=ones, gsrc=ones, and it is
    correct even with DUPLICATE indices (every contribution is independent)."""

    def test_scatter_add_dim1_backward(self):
        # x.scatter_add(1, idx, src) -- distinct columns, dim 1.
        base = np.zeros((2, 3), dtype="float32")
        idx = np.array([[0, 1, 2], [2, 1, 0]], dtype="int64")
        src = np.arange(1, 7, dtype="float32").reshape(2, 3)
        # forward ref via the index_add-style accumulation per (row, idx) cell
        ref = base.copy()
        for i in range(2):
            for j in range(3):
                ref[i, idx[i, j]] += src[i, j]
        gx_ref, gsrc_ref = np_scatter_add_grad((2, 3), 1, idx, (2, 3))

        def body(dev):
            xv = self._f32(base)
            sv = self._f32(src)
            iv = self._i64(idx)
            out = xv.scatter_add(1, iv, sv)         # out-of-place (clones internally)
            self.assertEqual(out, ref, msg=f"scatter_add fwd [{dev}]")
            gx, gs = jt.grad(out.sum(), [xv, sv])
            self.assertEqual(gx, gx_ref, msg=f"scatter_add d/dx [{dev}]")
            self.assertEqual(gs, gsrc_ref, msg=f"scatter_add d/dsrc [{dev}]")
        self._for_devices(body)

    def test_scatter_add_duplicate_indices_backward(self):
        # DUP collision (880cd6ad family, add path): index col all 0 -> every src in
        # that row accumulates into out[:,0].  add is dup-correct in fwd AND bwd.
        base = np.array([[1., 1., 1., 1.]], dtype="float32")
        idx = np.array([[0, 0, 0, 0]], dtype="int64")
        src = np.array([[1., 2., 3., 4.]], dtype="float32")
        ref = base.copy()
        for j in range(4):
            ref[0, idx[0, j]] += src[0, j]          # out[0,0] = 1 + (1+2+3+4) = 11
        self.assertEqual(ref.tolist(), [[11., 1., 1., 1.]])
        gx_ref, gsrc_ref = np_scatter_add_grad((1, 4), 1, idx, (1, 4))

        def body(dev):
            xv = self._f32(base)
            sv = self._f32(src)
            iv = self._i64(idx)
            out = xv.scatter_add(1, iv, sv)
            self.assertEqual(out, ref, msg=f"scatter_add dup fwd [{dev}]")
            gx, gs = jt.grad(out.sum(), [xv, sv])
            self.assertEqual(gx, gx_ref, msg=f"scatter_add dup d/dx [{dev}]")
            self.assertEqual(gs, gsrc_ref, msg=f"scatter_add dup d/dsrc [{dev}]")
        self._for_devices(body)


# ===========================================================================
# 4. index_add  (out-of-place, DUP-ACCUMULATE; forward + backward)
# ===========================================================================
class TestIndexAdd(_SetitemCore):
    """index_add ACCUMULATES duplicate indices (the torch contract; the 880cd6ad /
    index_add accumulate family).  Differentiable through the scatter_add(reduce=add)
    path: gx=ones, gsource=ones."""

    def test_index_add_duplicate_accumulate_backward(self):
        base = np.zeros((3, 2), dtype="float32")
        index = np.array([0, 0, 1], dtype="int64")     # rows 0,1 of src -> out row 0
        source = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype="float32")
        ref = np_index_add(base, 0, index, source)      # row0=[3,3], row1=[3,3], row2=0
        self.assertEqual(ref.tolist(), [[3., 3.], [3., 3.], [0., 0.]])

        def body(dev):
            xv = self._f32(base)
            sv = self._f32(source)
            iv = self._i64(index)
            out = xv.index_add(0, iv, sv)               # out-of-place
            self.assertEqual(out, ref, msg=f"index_add dup fwd [{dev}]")
            gx, gs = jt.grad(out.sum(), [xv, sv])
            self.assertEqual(gx, np.ones((3, 2), "float64"),
                             msg=f"index_add dup d/dx [{dev}]")
            self.assertEqual(gs, np.ones((3, 2), "float64"),
                             msg=f"index_add dup d/dsource [{dev}]")
        self._for_devices(body)

    def test_index_add_alpha_backward(self):
        # alpha scales source; grad wrt source is therefore alpha (not 1).
        base = np.ones((3, 2), dtype="float32")
        index = np.array([0, 0], dtype="int64")
        source = np.array([[1., 1.], [1., 1.]], dtype="float32")
        alpha = 2.0
        ref = np_index_add(base, 0, index, source, alpha=alpha)   # row0 = 1 + 2 + 2 = 5

        def body(dev):
            xv = self._f32(base)
            sv = self._f32(source)
            iv = self._i64(index)
            out = xv.index_add(0, iv, sv, alpha=alpha)
            self.assertEqual(out, ref, msg=f"index_add alpha fwd [{dev}]")
            gx, gs = jt.grad(out.sum(), [xv, sv])
            self.assertEqual(gx, np.ones((3, 2), "float64"),
                             msg=f"index_add alpha d/dx [{dev}]")
            self.assertEqual(gs, np.full((2, 2), alpha, "float64"),
                             msg=f"index_add alpha d/dsource [{dev}]")
        self._for_devices(body)


# ===========================================================================
# 5. scatter reduce=max / min / multiply  -- FORWARD ONLY (dup-collision lock)
#    Regression: 880cd6ad CUDA non-atomic RMW dropped colliding contributions.
# ===========================================================================
class TestScatterReduceMaxMinForward(_SetitemCore):
    """reduce in {max, min, multiply} via ``x.scatter(dim, idx, src, reduce=...)`` --
    FORWARD ONLY: the native ``SetitemOp::grad`` has no backward for these (it
    ``LOGf``-fatals), so this exercises only the value, with DUPLICATE indices that
    triggered the 880cd6ad CUDA silent-drop.  ``supports_autograd=False`` for this op.
    """

    # honest marker for the harness: these reduces have no analytic kernel backward.
    SUPPORTS_AUTOGRAD = False

    def _check(self, op, base, idx, src, label, atol=1e-5):
        # x.scatter(dim=0, idx, src, reduce=op) overwrites base AT idx cells, reducing
        # collisions with op; reduce='max'/'min'/'multiply' include the existing self.
        ref = np_scatter_reduce_forward(base, 0, idx, src, op)

        def body(dev):
            xv = self._f32(base)
            sv = self._f32(src)
            iv = self._i64(idx)
            # reduce strings: 'add'/'multiply' are torch-facing; 'max'/'min' map to
            # ns_maximum/ns_minimum in the kernel (string_to_ns).
            out = xv.scatter(0, iv, sv, reduce=op)
            self.assertEqual(out, ref, atol=atol, msg=f"{label} [{dev}]")
        self._for_devices(body)

    def test_scatter_max_duplicate(self):
        # heavy collision: all index entries target row 0 -> out[0,c] = max(self, all src).
        base = np.zeros((3, 4), dtype="float32")
        idx = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype="int64")
        src = np.array([[1., 5., 3., 9.], [4., 2., 8., 6.]], dtype="float32")
        self._check("max", base, idx, src, "scatter reduce=max dup")

    def test_scatter_min_duplicate(self):
        # base seeded ABOVE every src so min must pull each cell down to the src min;
        # the 880cd6ad bug dropped colliding contributions -> a too-high (stale) value.
        base = np.full((3, 4), 100.0, dtype="float32")
        idx = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype="int64")
        src = np.array([[7., 5., 3., 9.], [4., 2., 8., 6.]], dtype="float32")
        self._check("min", base, idx, src, "scatter reduce=min dup")

    def test_scatter_multiply_duplicate(self):
        # multiply was the third non-atomic RMW in 880cd6ad; product over colliding src.
        base = np.full((2, 3), 2.0, dtype="float32")
        idx = np.array([[0, 0, 0], [0, 0, 0]], dtype="int64")
        src = np.array([[1.5, 0.5, 2.0], [3.0, 4.0, 0.25]], dtype="float32")
        self._check("multiply", base, idx, src, "scatter reduce=multiply dup", atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
