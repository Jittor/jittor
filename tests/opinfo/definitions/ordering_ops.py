# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Ordering OpInfos: sort / topk (differentiable .values) + argsort / kthvalue /
median (the index-routing family the audit found under-tested).

WHY THIS FILE EXISTS (the coverage gap it closes)
-------------------------------------------------
The ordering ops were previously tested ONLY with the tautological oracle
``grad((key)^2) == 2*x`` (see ``test_argsort_op.py::check_backward``). That oracle is
blind to *where* the gradient lands: ``sum(sorted^2)`` has the same value no matter
which original position each sorted element came from, so a backward that scattered
the cotangent to the WRONG original index would still pass. The real backward of
``sort``/``topk`` is an *index-routing* scatter -- the upstream cotangent on sorted
position ``r`` must be deposited on the ORIGINAL position ``argsort[r]``. That routing
is exactly what these OpInfos exercise: gradcheck differentiates ``.values`` against
finite differences, so a mis-routed scatter diverges loudly.

NO-TIE INPUTS (why we build from a shuffled arange, not ``make_tensor``)
-----------------------------------------------------------------------
gradcheck perturbs each element by ``+/- eps`` (1e-6) and re-runs the forward. If two
input values are within ``2*eps`` of each other, that perturbation can *reorder* them,
which changes the permutation and makes the numerical Jacobian meaningless (the
gradient appears to jump between positions). A uniform ``make_tensor`` fill produces
near-ties with non-trivial probability. We instead feed a *random permutation of a
well-separated arange* (min gap == the scale, ~0.37 >> 1e-6), so the sort order is
locally constant under the FD step and the index-routing backward is unambiguous and
well-defined. This is the condition under which gradcheck of a sort is even valid.

SIGNATURES (verified against jittor source -- NOT guessed)
----------------------------------------------------------
The ``jittor.compat.torch`` installer, applied to the
jittor module at import) overrides these:
  * ``jt.sort(x, dim=-1, descending=False)``  -> ``_Sort(values, indices)`` namedtuple.
    test_ops compares ``.values``; ``.values`` is differentiable (argsort-gather).
  * ``jt.topk(x, k, dim=-1, largest=True, sorted=True)`` -> ``_TopK(values, indices)``.
    Built as ``gather(x, dim, argsort(...)[:k])`` -> values are differentiable.
  * ``jt.argsort(x, dim=-1, descending=False)`` -> indices-only int64 Var
    (NB: jittor's *native* argsort returns ``(index, value)``; the torch-compat
    module-level override returns indices only -- the form ``jt.argsort`` resolves to).
``jt.kthvalue(input, k, dim=None, keepdim=False)`` (``jittor.misc``) ->
``(values, indices)`` plain tuple. ``jt.median(x)`` returns the global value;
``jt.median(x, dim=..., keepdim=...)`` returns Torch-style ``(values, indices)``
under the compatibility installer, and this harness compares ``.values``.

WHAT IS MARKED NON-DIFFERENTIABLE / WHY
---------------------------------------
  * ``argsort`` is integer-valued (a permutation) -> ``supports_autograd=False`` (no
    meaningful derivative; forward-only battery vs ``np.argsort``).
  * ``sort`` / ``topk`` / ``kthvalue`` / ``median`` are gradchecked on their VALUES,
    but with ``supports_gradgrad=False``: the values backward is a piecewise-constant
    *selection/permutation* (linear in the cotangent for fixed indices) routed through
    jittor's getitem/reindex scatter, whose second derivative is not reliably
    available -- the same conservative stance the core registry takes for
    ``max``/``amax``/``min`` (all ``supports_gradgrad=False``). gradcheck of the
    first-order backward -- the actual coverage gap -- still runs.

cumsum / cumprod are covered in ``reductions_extra.py`` and are intentionally NOT
repeated here.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo


# ------------------------------------------------------------------- numpy refs

def sort_ref(x, dim=-1, descending=False):
    """torch sort(input, dim) -> (values, indices); test_ops compares .values, so the
    ref returns the sorted VALUES only. descending=True reverses along ``dim``."""
    s = np.sort(x, axis=dim)
    if descending:
        s = np.flip(s, axis=dim)
    return np.ascontiguousarray(s)


def topk_ref(x, k, dim=-1, largest=True, sorted=True):
    """torch topk -> (values, indices); compare .values. largest=True takes the k
    biggest (descending), largest=False the k smallest (ascending). Implemented as
    full sort then slice the leading k along ``dim`` -- matching the torch-compat op,
    which is argsort(descending=largest) then gather of the first k."""
    s = np.sort(x, axis=dim)
    if largest:
        s = np.flip(s, axis=dim)            # descending
    # slice the first k along ``dim``
    nd = x.ndim
    d = dim if dim >= 0 else dim + nd
    sl = [slice(None)] * nd
    sl[d] = slice(0, k)
    return np.ascontiguousarray(s[tuple(sl)])


def argsort_ref(x, dim=-1, descending=False):
    """torch argsort -> indices (a permutation) that sort ``x`` along ``dim``. Non-diff;
    forward-only. Distinct (no-tie) inputs make the permutation unique, so jittor's
    indices match numpy's exactly (np.argsort is stable/ascending; descending reverses
    the order, which on distinct values is unambiguous)."""
    if descending:
        # argsort of the negated values gives the descending permutation; on distinct
        # values this is exactly the reverse-order permutation with no tie ambiguity.
        return np.argsort(-x, axis=dim)
    return np.argsort(x, axis=dim)


def kthvalue_ref(x, k, dim=None, keepdim=False):
    """torch kthvalue(input, k) -> (values, indices) for the k-th SMALLEST (1-indexed);
    we wrap the op to return values only, so the ref returns the k-th smallest VALUE.
    dim=None reduces the last axis (jittor's kthvalue maps None -> -1).

    Matches a jittor quirk: ``jittor.misc.tensor_ops.kthvalue`` only squeezes the
    reduced axis when ``indices.ndim > 1``, so for a 1-D INPUT the result stays (1,)-shaped
    even with keepdim=False. The ref reproduces that (squeeze only when x.ndim > 1)."""
    d = -1 if dim is None else dim
    s = np.sort(x, axis=d)
    nd = x.ndim
    dd = d if d >= 0 else d + nd
    sl = [slice(None)] * nd
    sl[dd] = slice(k - 1, k)                 # k is 1-indexed
    out = s[tuple(sl)]
    if not keepdim and nd > 1:               # 1-D input: op leaves the (1,) axis intact
        out = np.squeeze(out, axis=dd)
    return np.ascontiguousarray(out)


def median_ref(x, dim=None, keepdim=False):
    """``jittor.misc.tensor_ops.median`` takes the LOWER median element at index
    (n-1)//2 of the
    sorted axis (it selects an actual element, not the mean of the two middle ones).
    dim=None flattens first. Returns the median VALUE (matches jittor's value output)."""
    if dim is None:
        xf = x.reshape(-1)
        s = np.sort(xf)
        kpos = (xf.shape[0] - 1) // 2
        return np.ascontiguousarray(np.atleast_1d(s[kpos]))
    s = np.sort(x, axis=dim)
    nd = x.ndim
    dd = dim if dim >= 0 else dim + nd
    kpos = (x.shape[dd] - 1) // 2
    sl = [slice(None)] * nd
    sl[dd] = slice(kpos, kpos + 1) if keepdim else kpos
    return np.ascontiguousarray(s[tuple(sl)])


# ---------------------------------------------------------------- op wrappers
# kthvalue returns a bare (values, indices) tuple (NOT a namedtuple), so the harness'
# ``hasattr(out, "values")`` check would not fire and it would try to compare the
# 2-tuple against a single ref array. Wrap to return VALUES only -- the differentiable
# half -- so both the forward compare and gradcheck see a single Var.

def _kthvalue_values(x, k, dim=None, keepdim=False):
    return jt.kthvalue(x, k, dim=dim, keepdim=keepdim)[0]


# --------------------------------------------------------------- sample builders
# Small tensors only (<= 24 elems on the differentiated operand): gradcheck is
# O(numel) forward passes. All ordering samples use DISTINCT, well-separated values
# (a shuffled arange) so the sort order is locally constant under the FD perturbation.

_SCALE = 0.37   # min gap between consecutive sorted values; >> gradcheck eps (1e-6)


def _perm_tensor(shape, dtype, requires_grad, seed):
    """A Var of ``shape`` whose entries are a random permutation of a well-separated
    arange (centered, scaled by _SCALE). Distinct by construction -> NO TIES, so
    sort/topk index routing is unambiguous and gradcheck is well-defined."""
    n = 1
    for s in shape:
        n *= s
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    vals = (np.arange(n, dtype="float64") - (n - 1) / 2.0) * _SCALE
    rng.shuffle(vals)
    a = np.ascontiguousarray(vals.reshape(shape).astype(np.float64))
    v = jt.array(a, dtype=str(a.dtype))
    if str(v.dtype) != dtype:
        v = v.cast(dtype)
    if requires_grad:
        try:
            v.requires_grad = True
        except Exception:
            pass
    return v


# Shapes kept <= 24 elements so gradcheck (central differences over every element)
# stays cheap. 1-D and 2-D cover the contiguous-axis and the strided-axis paths.
_ORD_SHAPES = [(6,), (3, 4), (2, 3, 4)]   # 6, 12, 24 elements


def sample_sort(op_info, device, dtype, requires_grad):
    out = []
    seed = 800
    for shape in _ORD_SHAPES:
        dims = list(range(len(shape))) + [-1]
        for dim in dims:
            for descending in (False, True):
                out.append(SampleInput(
                    _perm_tensor(shape, dtype, requires_grad, seed),
                    dim=dim, descending=descending))
                seed += 1
    return out


def sample_topk(op_info, device, dtype, requires_grad):
    out = []
    seed = 840
    for shape in _ORD_SHAPES:
        dims = list(range(len(shape))) + [-1]
        for dim in dims:
            d = dim if dim >= 0 else dim + len(shape)
            axis_len = shape[d]
            # k strictly between 1 and axis_len so the slice is a proper top-k (and the
            # gather-backward must scatter to a SUBSET of original positions -- the
            # dropped positions must get a zero gradient, a thing the tautological
            # key^2 oracle never checked).
            for k in sorted({1, max(1, axis_len // 2), axis_len}):
                for largest in (True, False):
                    out.append(SampleInput(
                        _perm_tensor(shape, dtype, requires_grad, seed),
                        k, dim=dim, largest=largest))
                    seed += 1
    return out


def sample_argsort(op_info, device, dtype, requires_grad):
    # Non-differentiable (integer permutation output): requires_grad is ignored by the
    # forward-only battery. Distinct values -> unique permutation -> jittor indices
    # match numpy exactly.
    out = []
    seed = 880
    for shape in _ORD_SHAPES:
        dims = list(range(len(shape))) + [-1]
        for dim in dims:
            for descending in (False, True):
                out.append(SampleInput(
                    _perm_tensor(shape, dtype, requires_grad, seed),
                    dim=dim, descending=descending))
                seed += 1
    return out


def sample_kthvalue(op_info, device, dtype, requires_grad):
    out = []
    seed = 920
    for shape in _ORD_SHAPES:
        # a required dim avoids the dim=None flatten path's ambiguity; sweep k over the
        # axis (k is 1-indexed, 1..axis_len).
        dims = list(range(len(shape))) + [-1]
        for dim in dims:
            d = dim if dim >= 0 else dim + len(shape)
            axis_len = shape[d]
            for k in sorted({1, (axis_len + 1) // 2, axis_len}):
                for keepdim in (False, True):
                    out.append(SampleInput(
                        _perm_tensor(shape, dtype, requires_grad, seed),
                        k, dim=dim, keepdim=keepdim))
                    seed += 1
    return out


def sample_median(op_info, device, dtype, requires_grad):
    out = []
    seed = 960
    for shape in _ORD_SHAPES:
        # full (flatten) median -- jittor takes the lower-median element, differentiable
        out.append(SampleInput(
            _perm_tensor(shape, dtype, requires_grad, seed)))
        seed += 1
        dims = list(range(len(shape))) + [-1]
        for dim in dims:
            for keepdim in (False, True):
                out.append(SampleInput(
                    _perm_tensor(shape, dtype, requires_grad, seed),
                    dim=dim, keepdim=keepdim))
                seed += 1
    return out


op_db = [
    # ---- sort -> (values, indices) namedtuple; differentiate .values ----------
    # The index-routing backward: the cotangent on sorted position r must land on the
    # ORIGINAL position argsort[r]. gradcheck (vs finite differences on a no-tie input)
    # catches a wrong-position scatter -- the gap the key^2 oracle could not see.
    # supports_gradgrad=False: the values backward is a permutation/selection (scatter
    # via getitem/reindex) whose 2nd derivative is not reliably available in jittor
    # (mirrors the core registry's max/amax/min stance).
    OpInfo("sort", op=jt.sort, ref=sort_ref, sample_inputs_func=sample_sort,
           supports_gradgrad=False),

    # ---- topk -> (values, indices); differentiate .values ---------------------
    # Backward gathers k positions then scatters back; the DROPPED positions must
    # receive a zero gradient. gradcheck verifies both the routing and the zeros.
    OpInfo("topk", op=jt.topk, ref=topk_ref, sample_inputs_func=sample_topk,
           supports_gradgrad=False),

    # ---- argsort -> integer permutation (NON-differentiable) ------------------
    # int64 output; supports_autograd=False -> forward-only vs np.argsort. Distinct
    # inputs make the permutation unique so jittor == numpy exactly.
    OpInfo("argsort", op=jt.argsort, ref=argsort_ref,
           sample_inputs_func=sample_argsort,
           dtypes=cu.floating_types(), supports_autograd=False),

    # ---- kthvalue -> (values, indices); differentiate VALUES ------------------
    # Wrapped to return values only (jittor returns a bare tuple, not a namedtuple, so
    # the harness' .values extraction would not fire). The value is gather(input,
    # argsort_idx)[k-1] -> a single-position select whose backward routes the cotangent
    # to the original index of the k-th smallest.
    OpInfo("kthvalue", op=_kthvalue_values, ref=kthvalue_ref,
           sample_inputs_func=sample_kthvalue, supports_gradgrad=False),

    # ---- median -> value (jittor selects the LOWER-median element) ------------
    # Differentiable single-element select; the cotangent routes to the original
    # position of the lower-median element -- same select-backward family as kthvalue.
    #
    OpInfo("median", op=jt.median, ref=median_ref,
           sample_inputs_func=sample_median, supports_gradgrad=False),
]
