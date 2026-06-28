# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Indexing / gather-scatter OpInfos.

Covers the data-movement family: ``gather``/``scatter``/``scatter_add``,
``index_select``/``index_add``, ``take``, ``masked_select``, ``masked_fill`` and
``where`` (the 3-arg ``ternary`` selector). These ops carry an integer (or boolean)
*addressing* operand that is **not** differentiable -- the gradient flows only to the
float source/destination operands. The generic ``test_ops`` gradcheck driver
differentiates exactly the floating-dtype Vars in ``[input, *args]`` (see
``_diff_plan``), so index/mask tensors are built as ``int64``/``bool`` Vars and the
scatter *value* / selection *index* are passed as positional ``args`` to keep them
out of the differentiated set. Index ranges are constructed to be in-bounds, and
scatter destinations are made unique so the (order-unspecified) overwrite has a
deterministic forward to compare against the numpy reference.

numpy oracles: gather -> ``np.take_along_axis``; index_select -> fancy index;
scatter (overwrite) / scatter_add -> per-element loop on a copy (``np.add.at``
semantics for the accumulating variant); index_add -> accumulating per-slice add;
take -> flat fancy index; masked_select -> boolean index; masked_fill / where ->
``np.where``.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo

# Resolve the jittor callables explicitly (all native, always importable: gather /
# scatter / scatter_add / index_add come from misc via `from .misc import *`;
# index_select is jt.index_select; masked_fill is defined in jittor/__init__.py;
# `where` is the core ternary selector). take / masked_select are expressed through
# always-present native primitives rather than the torch_compat aliases so the op_db
# does not depend on torch_compat being activated.
_take = lambda x, index: x.reshape((-1,))[index]
_masked_select = lambda x, mask: x.reshape((-1,))[jt.where(mask.reshape((-1,)))[0]]
_where = lambda cond, a, b: jt.ternary(cond, a, b)


# ------------------------------------------------------------------- numpy refs

def gather_ref(x, dim, index):
    return np.take_along_axis(x, index, axis=dim)


def scatter_ref(x, dim, index, src):
    """torch scatter (overwrite): out[.. index[m] @dim ..] = src[m] for every m.
    Forward is only well defined when destinations are unique (the sample builder
    guarantees this), matching jittor's setitem semantics."""
    out = np.array(x, copy=True)
    it = np.nditer(index, flags=["multi_index"])
    for v in it:
        mi = it.multi_index
        dst = list(mi)
        dst[dim] = int(v)
        out[tuple(dst)] = src[mi]
    return out


def scatter_add_ref(x, dim, index, src):
    """torch scatter_add (out-of-place): accumulate src into a COPY of x at index
    along dim (np.add.at semantics -- duplicate destinations accumulate)."""
    out = np.array(x, copy=True)
    it = np.nditer(index, flags=["multi_index"])
    for v in it:
        mi = it.multi_index
        dst = list(mi)
        dst[dim] = int(v)
        out[tuple(dst)] += src[mi]
    return out


def index_select_ref(x, dim, index):
    sl = (slice(None),) * dim + (index,)
    return x[sl]


def index_add_ref(x, dim, index, source, alpha=1):
    """torch index_add (out-of-place): out[.., index[k], ..] += alpha*source[.., k, ..],
    ACCUMULATING duplicate indices."""
    out = np.array(x, copy=True)
    src = source * alpha
    for k, j in enumerate(index):
        sl_dst = [slice(None)] * out.ndim; sl_dst[dim] = int(j)
        sl_src = [slice(None)] * src.ndim; sl_src[dim] = k
        out[tuple(sl_dst)] += src[tuple(sl_src)]
    return out


def take_ref(x, index):
    return x.reshape(-1)[index]


def masked_select_ref(x, mask):
    # jittor selects in row-major (C) order; numpy boolean index does the same.
    return x.reshape(-1)[mask.reshape(-1).astype(bool)]


def masked_fill_ref(x, mask, value):
    return np.where(mask.astype(bool), np.array(value, dtype=x.dtype), x)


def where_ref(cond, a, b):
    return np.where(cond.astype(bool), a, b)


# --------------------------------------------------------------- sample builders
#
# All index/mask operands are placed in positional ``args`` as int64 / bool Vars (or
# python scalars), so the gradcheck driver differentiates ONLY the float Var(s) among
# [input, *args]. Tensors are kept small (<= 24 elements per differentiated operand)
# because gradcheck cost is O(numel) forward passes.

def _idx(*shape, low, high, seed):
    """Deterministic int64 index Var with values in [low, high)."""
    return make_tensor(*shape, dtype="int64", low=low, high=high, seed=seed)


def sample_gather(op_info, device, dtype, requires_grad):
    out = []
    # 2-D, gather along dim 1 (size 4): index shape (3, 2), values in [0, 4)
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=600),
        1, _idx(3, 2, low=0, high=4, seed=601)))
    # 2-D, gather along dim 0 (size 3): index shape (2, 4), values in [0, 3)
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=602),
        0, _idx(2, 4, low=0, high=3, seed=603)))
    # 3-D, gather along dim 2 (size 4): index shape (2, 3, 2), values in [0, 4)
    out.append(SampleInput(
        make_tensor(2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=604),
        2, _idx(2, 3, 2, low=0, high=4, seed=605)))
    return out


def _unique_col_index(n_rows, n_cols, seed):
    """An (n_rows, n_cols) int64 index whose entries are DISTINCT per (row, col)
    cell-destination so a scatter overwrite is order-independent. Built by taking,
    per source row, a random permutation prefix of the destination-axis range."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_rows):
        perm = rng.permutation(n_cols)[:n_cols]
        rows.append(perm)
    a = np.ascontiguousarray(np.stack(rows).astype("int64"))
    return jt.array(a, dtype="int64")


def sample_scatter(op_info, device, dtype, requires_grad):
    out = []
    # scatter along dim 1 into a (2, 5) destination; src/index are (2, 3) with
    # DISTINCT columns per row so the overwrite forward is deterministic.
    idx = _unique_col_index(2, 5, seed=610)[:, :3]
    out.append(SampleInput(
        make_tensor(2, 5, dtype=dtype, requires_grad=requires_grad, seed=611),
        1, idx,
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=612)))
    return out


def sample_scatter_add(op_info, device, dtype, requires_grad):
    out = []
    # heavy collision along dim 1: all of row's src lands in column 0 -> exercises
    # the ACCUMULATE path and its broadcast-back backward.
    idx = jt.zeros((2, 3), dtype="int64")
    out.append(SampleInput(
        make_tensor(2, 4, dtype=dtype, requires_grad=requires_grad, seed=620),
        1, idx,
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=621)))
    # scatter_add along dim 0 with mixed (some-colliding) rows.
    idx0 = jt.array(np.array([[0, 1, 2, 0], [2, 0, 1, 2]], dtype="int64"), dtype="int64")
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=622),
        0, idx0,
        make_tensor(2, 4, dtype=dtype, requires_grad=requires_grad, seed=623)))
    return out


def sample_index_select(op_info, device, dtype, requires_grad):
    out = []
    # 1-D index (length 3, repeats allowed) selecting along each dim of a (3, 4) Var.
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=630),
        0, _idx(3, low=0, high=3, seed=631)))
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=632),
        1, _idx(3, low=0, high=4, seed=633)))
    # 3-D along dim 1.
    out.append(SampleInput(
        make_tensor(2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=634),
        1, _idx(2, low=0, high=3, seed=635)))
    return out


def sample_index_add(op_info, device, dtype, requires_grad):
    out = []
    # index_add along dim 0: 1-D index (length 2, may collide), source rows match.
    idx = jt.array(np.array([0, 0], dtype="int64"), dtype="int64")  # collide -> accumulate
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=640),
        0, idx,
        make_tensor(2, 4, dtype=dtype, requires_grad=requires_grad, seed=641)))
    # along dim 1, non-colliding.
    idx1 = jt.array(np.array([1, 3], dtype="int64"), dtype="int64")
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=642),
        1, idx1,
        make_tensor(3, 2, dtype=dtype, requires_grad=requires_grad, seed=643)))
    return out


def sample_take(op_info, device, dtype, requires_grad):
    out = []
    # flat indices into a (3, 4)=12-element Var (repeats allowed).
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=650),
        _idx(5, low=0, high=12, seed=651)))
    # 2-D index shape preserved on output.
    out.append(SampleInput(
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=652),
        _idx(2, 2, low=0, high=6, seed=653)))
    return out


def _fixed_mask(*shape, seed):
    rng = np.random.RandomState(seed)
    a = np.ascontiguousarray((rng.uniform(size=shape) > 0.4))
    return jt.array(a, dtype="bool")


def sample_masked_select(op_info, device, dtype, requires_grad):
    out = []
    # mask is a FIXED bool Var (independent of x) so the selected SHAPE is stable
    # across the gradcheck finite-difference perturbations.
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=660),
        _fixed_mask(3, 4, seed=661)))
    out.append(SampleInput(
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=662),
        _fixed_mask(2, 3, seed=663)))
    return out


def sample_masked_fill(op_info, device, dtype, requires_grad):
    out = []
    # masked_fill(x, mask, value): x is differentiated; mask (bool) and value (scalar)
    # are passed as args so they are not.
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=670),
        _fixed_mask(3, 4, seed=671), 1.5))
    out.append(SampleInput(
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=672),
        _fixed_mask(2, 3, seed=673), -2.0))
    return out


def sample_where(op_info, device, dtype, requires_grad):
    out = []
    # where(cond, a, b): cond is the (non-differentiated bool) `input`; a, b are the
    # differentiated float operands passed as args. Same shape (no broadcast) keeps
    # the gradient mapping a straight elementwise mask.
    out.append(SampleInput(
        _fixed_mask(3, 4, seed=680),
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=681),
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=682)))
    out.append(SampleInput(
        _fixed_mask(2, 3, seed=683),
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=684),
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=685)))
    return out


op_db = [
    # ---- gather / take (pull) ----
    OpInfo("gather", op=jt.gather, ref=gather_ref,
           sample_inputs_func=sample_gather),
    OpInfo("take", op=_take, ref=take_ref,
           sample_inputs_func=sample_take),

    # ---- scatter / scatter_add (push) ----
    # Both float operands (destination x and source src) are differentiated; the
    # int64 index between them is auto-skipped by the gradcheck driver. Overwrite uses
    # unique destinations so its (order-unspecified) forward is deterministic.
    OpInfo("scatter", op=jt.scatter, ref=scatter_ref,
           sample_inputs_func=sample_scatter),
    OpInfo("scatter_add", op=jt.scatter_add, ref=scatter_add_ref,
           sample_inputs_func=sample_scatter_add),

    # ---- index_select / index_add ----
    OpInfo("index_select", op=jt.index_select, ref=index_select_ref,
           sample_inputs_func=sample_index_select),
    OpInfo("index_add", op=jt.index_add, ref=index_add_ref,
           sample_inputs_func=sample_index_add),

    # ---- masked ops ----
    # masked_select: output shape is data-dependent but the mask is FIXED (independent
    # of x), so the selected shape is stable under gradcheck perturbations and the
    # gradient (a scatter-back of the cotangent into the selected positions) is exact.
    OpInfo("masked_select", op=_masked_select, ref=masked_select_ref,
           sample_inputs_func=sample_masked_select),
    OpInfo("masked_fill", op=jt.masked_fill, ref=masked_fill_ref,
           sample_inputs_func=sample_masked_fill),

    # ---- where (3-arg ternary select) ----
    # cond (bool) is the non-differentiated input; a, b are the differentiated float
    # branches. Linear in a/b -> gradgrad is the trivial zero second derivative.
    OpInfo("where", op=_where, ref=where_ref,
           sample_inputs_func=sample_where),
]
