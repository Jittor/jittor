# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reindex / fusion-primitive OpInfos -- the meta-operator bedrock.

This module pins the *fusion primitives* every higher op (conv, pool, interpolate,
pad, tile) lowers to. Two of them are jittor-specific meta-operators with no torch
analogue:

  * ``jt.reindex(x, shape, indexes)`` -- a one-to-many **PULL** (gather): for each
    output coordinate ``(i0,i1,...)`` it reads ``x`` at the coordinate given by the
    C++-style integer expressions ``indexes`` (one per *input* axis), substituting
    ``overflow_value`` when the computed source coordinate is out of bounds. Its
    backward is ``reindex_reduce(add)`` (scatter-add of the cotangent back into x).
  * ``jt.reindex_reduce(y, op, shape, indexes)`` -- a many-to-one **PUSH**
    (scatter-reduce): for each *source* coordinate it computes an output coordinate
    from ``indexes`` (one per *output* axis) and folds ``y`` into ``x`` with ``op``.
    Its add-backward is a plain ``reindex`` (gather) of the cotangent.

The remaining ops here lock the shape-movement primitives the task calls out
(``broadcast_to``, ``repeat``, ``unsqueeze``-style) **expressed as reindex
lowerings** -- the exact index patterns conv/pool emit. They are registered under
``*_via_reindex`` names (not bare ``broadcast_to``/``repeat``/``unsqueeze``, which
``shape.py`` already locks against ``jt.broadcast``/``jt.repeat``/``jt.unsqueeze``):
duplicating a bare ``full_name`` would silently clobber the other entry's generated
test method (``instantiate_device_type_tests`` ``setattr``s by ``full_name``). Here
they instead pin that the **reindex meta-op itself** reproduces broadcast / tile /
unsqueeze, so the fusion primitive's forward+backward is covered on every device
without re-testing the already-covered ``shape.py`` callables.

Design notes (see test_ops.py::_diff_plan):
  * The generic driver differentiates exactly the floating-dtype Vars in
    ``[input, *args]``. So every ``shape`` list, integer ``reps`` list, ``dim`` int
    and ``indexes`` expression-string list is passed as a NON-Var positional arg
    (or kwarg): ``_maybe_np`` passes such python objects through unchanged to the
    numpy ref, and they are skipped by the differentiated set. Only ``x`` (the
    float Var) is differentiated -- which is exactly what reindex/reindex_reduce
    define a gradient for (the C++ ``grad`` returns ``nullptr`` for every non-x
    input).

  * The reindex index expressions are chosen **division-free** (``i1``, ``i0``,
    constants, ``i0-{p}``) so the SAME string is a valid C++ integer expression for
    jittor AND a valid python expression for the numpy oracle -- the oracle
    ``eval``s each expression over a coordinate meshgrid, mirroring the documented
    pseudo-implementation exactly (including the overflow -> overflow_value branch).
    (jittor's ``/`` is C++ integer division but python's ``/`` is float; avoiding
    ``/`` keeps the two interpreters bit-identical, so the oracle stays independent
    yet exact.)

PRESERVED assets: the conv/conv_transpose/pool numpy references and their semantics
live in test_reindex_op.py / test_reindex_reduce_op.py (kept as regression locks).
Here we add the *primitive-level* forward+backward coverage those higher-level tests
assume; we do not duplicate or "fix" them.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo

# ------------------------------------------------------------------- op callables
#
# reindex / reindex_reduce are always-present native meta-ops. The shape-movement
# entries below are deliberately routed THROUGH reindex (not jt.broadcast / jt.repeat
# / jt.unsqueeze, which shape.py already covers) so this module pins the reindex
# lowering of those primitives. Every callable forwards its non-Var shape/reps/dim
# arguments verbatim so the same value reaches the numpy oracle unchanged.

_reindex = lambda x, shape, indexes, **kw: jt.reindex(x, shape, indexes, **kw)
_reindex_reduce = lambda y, op, shape, indexes, **kw: jt.reindex_reduce(y, op, shape, indexes, **kw)


def _broadcast_to_via_reindex(x, shape):
    """broadcast_to as a reindex PULL. Output axes right-align with x's; an x axis of
    size 1 (broadcast) maps to source index 0, otherwise to its aligned output coord;
    new leading output axes simply do not appear in the (per-x-axis) index list."""
    out_shape = [int(s) for s in shape]
    n, m = len(x.shape), len(out_shape)
    off = m - n
    idx = ["0" if x.shape[k] == 1 else f"i{off + k}" for k in range(n)]
    return jt.reindex(x, out_shape, idx)


def _repeat_via_reindex(x, reps):
    """repeat/tile as a reindex PULL: x-axis k is read modulo its own length, so the
    tile wraps. len(reps) == x.ndim here (each axis tiled in place)."""
    reps = [int(r) for r in reps]
    out_shape = [x.shape[k] * reps[k] for k in range(len(reps))]
    idx = [f"i{k}%{x.shape[k]}" for k in range(len(x.shape))]
    return jt.reindex(x, out_shape, idx)


def _unsqueeze_via_reindex(x, dim):
    """unsqueeze as a reindex PULL: insert a size-1 output axis at ``dim``; every x
    axis reads the output coord shifted past the inserted axis."""
    n = len(x.shape)
    if dim < 0:
        dim += n + 1
    out_shape = list(x.shape[:dim]) + [1] + list(x.shape[dim:])
    idx = [f"i{k if k < dim else k + 1}" for k in range(n)]
    return jt.reindex(x, out_shape, idx)


# ------------------------------------------------------------------- numpy refs

def _eval_index_exprs(indexes, out_shape):
    """Evaluate each C++-style integer index expression over the output coordinate
    grid, returning one int64 ndarray of shape ``out_shape`` per expression.

    Mirrors the reindex pseudo-implementation: ``i0,i1,...`` range over ``out_shape``.
    The expressions used in this module are division-free, so python ``eval`` matches
    jittor's C++ evaluation exactly. ``np.intp`` arithmetic stays integral throughout.
    """
    grids = np.meshgrid(*[np.arange(s, dtype=np.int64) for s in out_shape],
                        indexing="ij")
    env = {f"i{d}": grids[d] for d in range(len(out_shape))}
    coords = []
    for expr in indexes:
        val = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 (trusted literals)
        coords.append(np.asarray(val, dtype=np.int64) + np.zeros(out_shape, np.int64))
    return coords


def reindex_ref(x, shape, indexes, overflow_value=0, overflow_conditions=(), extras=()):
    """Independent numpy oracle for ``jt.reindex`` (the PULL / gather meta-op).

    ``y[I] = x[indexes[0](I), ..., indexes[m](I)]`` when every computed source
    coordinate is in bounds, else ``overflow_value``. ``len(indexes) == x.ndim``.
    """
    x = np.asarray(x)
    out_shape = tuple(int(s) for s in shape)
    assert len(indexes) == x.ndim, (len(indexes), x.ndim)
    src = _eval_index_exprs(list(indexes), out_shape)  # one (out_shape) array per x-axis
    overflow = np.zeros(out_shape, dtype=bool)
    clamped = []
    for d, c in enumerate(src):
        overflow |= (c < 0) | (c >= x.shape[d])
        clamped.append(np.clip(c, 0, x.shape[d] - 1))
    gathered = x[tuple(clamped)]
    out = np.where(overflow, np.asarray(overflow_value, dtype=x.dtype), gathered)
    return out.astype(x.dtype, copy=False)


def reindex_reduce_ref(y, op, shape, indexes, overflow_conditions=(), extras=()):
    """Independent numpy oracle for ``jt.reindex_reduce`` (the PUSH / scatter-reduce).

    For each source coordinate ``J`` over ``y.shape``, the output coordinate is
    ``(indexes[0](J), ..., indexes[m](J))`` and ``x[coord] = op(x[coord], y[J])`` when
    in bounds. ``len(indexes) == len(shape)``. Only ``op="add"`` is registered for
    gradcheck (its backward is a plain gather and is smooth); ``np.add.at`` gives the
    exact accumulate-with-collisions forward.
    """
    y = np.asarray(y)
    out_shape = tuple(int(s) for s in shape)
    assert len(indexes) == len(out_shape), (len(indexes), len(out_shape))
    # index expressions range over the SOURCE (y) coordinate grid.
    coords = _eval_index_exprs(list(indexes), y.shape)
    overflow = np.zeros(y.shape, dtype=bool)
    clamped = []
    for d, c in enumerate(coords):
        overflow |= (c < 0) | (c >= out_shape[d])
        clamped.append(np.clip(c, 0, out_shape[d] - 1))
    if op != "add":
        raise NotImplementedError(f"reindex_reduce_ref only models op='add' (got {op!r})")
    out = np.zeros(out_shape, dtype=y.dtype)
    src = np.where(overflow, np.zeros_like(y), y)  # overflowing sources contribute 0
    np.add.at(out, tuple(c[~overflow] for c in clamped), src[~overflow])
    return out


def broadcast_to_ref(x, shape):
    # independent oracle: numpy broadcast (copy off the read-only view for clean grad).
    return np.array(np.broadcast_to(np.asarray(x), tuple(int(s) for s in shape)))


def repeat_ref(x, reps):
    # len(reps) == x.ndim -> the reindex modulo-tile equals np.tile axis-for-axis.
    return np.tile(np.asarray(x), tuple(int(r) for r in reps))


def unsqueeze_ref(x, dim):
    return np.expand_dims(np.asarray(x), dim)


# --------------------------------------------------------------- sample builders
#
# Every non-float operand (shape lists, reps, dims, index-expression strings) is a
# plain python object passed positionally, so the gradcheck driver differentiates
# ONLY ``x``. Differentiated tensors are kept <= 24 elements (gradcheck is O(numel)).

def sample_reindex(op_info, device, dtype, requires_grad):
    out = []
    # (a) 2-D transpose y[i,j] = x[j,i]  -- a pure permutation (no overflow), the
    #     canonical "data move expressed via reindex"; backward is the transpose.
    out.append(SampleInput(
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=700),
        [3, 2], ["i1", "i0"]))
    # (b) broadcast a row: x is 1-D (4,), y[i,j] = x[j] -> shape (3, 4). len(indexes)
    #     == x.ndim == 1. Backward sums the cotangent over the broadcast axis.
    out.append(SampleInput(
        make_tensor(4, dtype=dtype, requires_grad=requires_grad, seed=701),
        [3, 4], ["i1"]))
    # (c) 3-D axis permutation y[i,j,k] = x[k,i,j] (shape (2,3,4) -> (4,2,3)).
    out.append(SampleInput(
        make_tensor(2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=702),
        [4, 2, 3], ["i1", "i2", "i0"]))
    # (d) pad with overflow: 1-D (6,) -> (10,), y[i] = x[i-2] else overflow_value.
    #     Exercises the overflow branch AND its (zero-on-overflow) backward; the
    #     reindex grad is reindex_reduce(add) which drops the out-of-range cotangent.
    out.append(SampleInput(
        make_tensor(6, dtype=dtype, requires_grad=requires_grad, seed=703),
        [10], ["i0-2"], overflow_value=0.0))
    return out


def sample_reindex_reduce(op_info, device, dtype, requires_grad):
    out = []
    # (a) column sum: y (4, 3) -> x (1, 3) via push to row 0. Heavy collision along
    #     axis 0 -> exercises the ACCUMULATE path; backward is broadcast-back (gather).
    out.append(SampleInput(
        make_tensor(4, 3, dtype=dtype, requires_grad=requires_grad, seed=710),
        "add", [1, 3], ["0", "i1"]))
    # (b) transpose-scatter (collision-free permutation): y (2, 3) -> x (3, 2).
    out.append(SampleInput(
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=711),
        "add", [3, 2], ["i1", "i0"]))
    # (c) total reduce to a single bin: y (3, 4) -> x (1, 1) (every source folds into
    #     x[0,0]) -- the maximal-collision add; backward is the all-ones broadcast.
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=712),
        "add", [1, 1], ["0", "0"]))
    return out


def sample_broadcast_to(op_info, device, dtype, requires_grad):
    out = []
    # in-place broadcast of a unit axis (3,1)->(3,4); a brand-new leading axis
    # (4,)->(2,4); and a mixed case (1,3,1)->(2,3,4). Backward sums the cotangent
    # back over every replicated axis (the broadcast-back hole).
    out.append(SampleInput(
        make_tensor(3, 1, dtype=dtype, requires_grad=requires_grad, seed=720),
        [3, 4]))
    out.append(SampleInput(
        make_tensor(4, dtype=dtype, requires_grad=requires_grad, seed=721),
        [2, 4]))
    out.append(SampleInput(
        make_tensor(1, 3, 1, dtype=dtype, requires_grad=requires_grad, seed=722),
        [2, 3, 4]))
    return out


def sample_repeat(op_info, device, dtype, requires_grad):
    out = []
    # len(reps) == x.ndim so the modulo-tile equals np.tile (each axis tiled in place).
    out.append(SampleInput(
        make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=740),
        [2, 1]))
    out.append(SampleInput(
        make_tensor(3, dtype=dtype, requires_grad=requires_grad, seed=741),
        [3]))
    out.append(SampleInput(
        make_tensor(2, 2, dtype=dtype, requires_grad=requires_grad, seed=742),
        [1, 3]))
    return out


def sample_unsqueeze(op_info, device, dtype, requires_grad):
    out = []
    for i, (shape, dim) in enumerate([((4,), 0), ((4,), 1), ((2, 3), 1), ((2, 3), -1)]):
        out.append(SampleInput(
            make_tensor(*shape, dtype=dtype, requires_grad=requires_grad, seed=750 + i),
            dim))
    return out


op_db = [
    # ---- reindex (one-to-many PULL / gather meta-op) ----
    # Forward pinned to an independent eval-the-expression numpy oracle; backward
    # (reindex_reduce(add) scatter-back) gradchecked. Samples cover transpose,
    # broadcast, 3-D permutation and the overflow/pad branch. The mapping is a pure
    # gather of x, so it is linear in x -> gradgrad is the trivial zero 2nd derivative.
    OpInfo("reindex", op=_reindex, ref=reindex_ref,
           sample_inputs_func=sample_reindex),

    # ---- reindex_reduce (many-to-one PUSH / scatter-reduce meta-op) ----
    # Only op="add" is registered: its backward is a plain gather (smooth, exact),
    # and np.add.at gives the exact collision-accumulating forward oracle. The
    # maximum/minimum/multiply reduces are NON-smooth (max/min) or have an
    # other-operand-dependent backward (product) -- those variants are regression-
    # locked forward+grad in the preserved test_reindex_reduce_op.py and are
    # deliberately kept out of this generic gradcheck (see module docstring).
    OpInfo("reindex_reduce", op=_reindex_reduce, ref=reindex_reduce_ref,
           sample_inputs_func=sample_reindex_reduce),

    # ---- shape-movement primitives, expressed AS reindex lowerings ----
    # These pin that the reindex meta-op reproduces broadcast / tile / unsqueeze (the
    # patterns conv & friends emit). Registered under `*_via_reindex` full_names so
    # they do NOT collide with shape.py's bare `broadcast_to`/`repeat`/`unsqueeze`
    # (which test jt.broadcast/jt.repeat/jt.unsqueeze directly) -- a colliding
    # full_name would silently overwrite the other's generated test method. The
    # forward is pinned to an INDEPENDENT numpy oracle (broadcast_to / tile /
    # expand_dims); the backward (reindex_reduce(add) sum-back over the replicated
    # axes) is gradchecked. All three are linear in x -> gradgrad is trivially zero.
    OpInfo("broadcast_to", variant_test_name="via_reindex",
           op=_broadcast_to_via_reindex, ref=broadcast_to_ref,
           sample_inputs_func=sample_broadcast_to),
    OpInfo("repeat", variant_test_name="via_reindex",
           op=_repeat_via_reindex, ref=repeat_ref,
           sample_inputs_func=sample_repeat),
    OpInfo("unsqueeze", variant_test_name="via_reindex",
           op=_unsqueeze_via_reindex, ref=unsqueeze_ref,
           sample_inputs_func=sample_unsqueeze),
]
