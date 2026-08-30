# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Integer- and bool-dtype reduction OpInfos (the int-reduce kernel hole).

The core registry (``core_ops.py``) and ``reductions_extra.py`` test reductions
over *floating* dtypes only -- which is exactly where the int8/int16 CUDA reduce
miscompile (``eb3c8bee``) hid: a float-only forward/gradcheck never instantiates
the integer reduce kernel, and ``gradcheck`` runs CPU-only so it could not have
seen a device-specific miscompile even if it had. These OpInfos close that hole by
driving ``sum``/``prod``/``max``/``min`` over EVERY integer dtype
(``cu.integral_types()`` = uint8/int8/int16/int32/int64) with NEGATIVE values, plus
``all``/``any`` over ``bool``.

How this catches the bug (two independent layers):
  * ``test_ops.TestCommon.test_reference`` compares the forward against an
    INDEPENDENT numpy oracle. (It uses the ``any_one`` dtype policy, which prefers
    ``int64`` here, so the forward oracle runs on the int64 kernel.)
  * ``test_device_parity.TestDeviceParity`` re-materializes each SampleInput's
    numpy and runs the op on CPU *and* on the accelerator, asserting they agree.
    That driver builds its tensors from ``sample.input``'s OWN dtype (it ignores
    the dtype argument and re-arrays from ``str(np.asarray(...).dtype)``), so the
    sample builders below HARDCODE each integer dtype -- one SampleInput per
    dtype -- which is what actually instantiates the int8/int16/... reduce kernel on
    both devices and pits CUDA against the (serial, trusted) CPU oracle. That is the
    precise mechanism by which ``eb3c8bee`` originally surfaced.

Differentiability: integer and boolean reductions are NOT differentiable in this
context, so every OpInfo here is ``supports_autograd=False`` -- the forward battery
runs, ``gradcheck``/``gradgradcheck`` are skipped, and the device-parity driver
only compares forwards (it gates the backward on float diff-vars).

Binding (torch_compat independence -- mirrors ``indexing.py``'s policy): the ops are
bound to the NATIVE reduce primitives ``jt.reduce_add`` / ``jt.reduce_multiply`` /
``jt.reduce_maximum`` / ``jt.reduce_minimum`` and ``jt.all_`` / ``jt.any_`` (the C++
``@pybind`` reduce ops). The user-facing ``jt.sum`` / ``Var.sum`` / ``jt.max`` are
re-wrapped when ``torch_compat`` is active -- ``Var.sum`` even UPCASTS int8/uint8 to
int32 before summing (through ``jittor.compat.torch``), a workaround that would *mask* the
very int8 reduce kernel this file is meant to exercise, and ``jt.max(x, dim)`` then
returns a (values, indices) namedtuple. Binding to the native primitives keeps the
test on the raw integer reduce kernel regardless of whether torch_compat is loaded.

Signatures (verified against ``__init__.pyi`` / ``src/ops/reduce_op.cc``): every
native reduce primitive is ``op(x, dim:int, keepdims=False)`` /
``op(x, dims:Tuple=(), keepdims=False)``; a no-arg call is the full reduce and yields
a ``(1,)``-shaped Var (jittor has no 0-d scalar -> the numpy refs ``atleast_1d``).
The reduce kernel keeps the INPUT integer dtype on output (``reduce_dtype_infer``: no
integer promotion), except a ``bool`` input always yields ``int32``
(reduce_op.cc L289) -- so the all/any refs below emit int32 0/1 to match exactly.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F, cu)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo, skip


# ------------------------------------------------------------------- numpy refs
#
# A reduction full-reduces when no ``dim`` kwarg is given, else reduces ``dim``
# honoring ``keepdims``. ``np.atleast_1d`` lifts the (0-d) numpy scalar of a full
# reduce to the ``(1,)`` shape jittor produces. The integer refs operate in the
# input dtype so wraparound (if any) matches the integer reduce kernel; the chosen
# value ranges (see the sample builders) keep every forwarded reduction in-range so
# the int64 oracle is exact and the sign behaviour of ``prod`` over negatives is the
# discriminating signal.

def _arith_reduce_ref(npfn):
    """sum/prod/max/min reference, matching jittor's (dim, keepdims) kwargs and its
    ``(1,)``-shaped full-reduce result. ``npfn`` is np.sum/np.prod/np.max/np.min.

    The result is cast back to the input dtype: numpy's ``np.prod`` upcasts small
    int dtypes to the platform int, whereas jittor's reduce keeps the input dtype
    (reduce_dtype_infer -> no promotion), so we must narrow to compare exactly."""
    def ref(x, dim=None, keepdims=False):
        out = np.atleast_1d(npfn(x, axis=dim, keepdims=keepdims))
        return out.astype(x.dtype)
    return ref


sum_ref = _arith_reduce_ref(np.sum)
prod_ref = _arith_reduce_ref(np.prod)
max_ref = _arith_reduce_ref(np.max)
min_ref = _arith_reduce_ref(np.min)


def all_ref(x, dim=None, keepdims=False):
    """jittor ``all_`` over bool: reduce truthiness; output dtype is int32 (a bool
    input forces an int32 reduce output, reduce_op.cc L289), so emit int32 0/1."""
    return np.atleast_1d(np.all(x != 0, axis=dim, keepdims=keepdims)).astype("int32")


def any_ref(x, dim=None, keepdims=False):
    return np.atleast_1d(np.any(x != 0, axis=dim, keepdims=keepdims)).astype("int32")


# --------------------------------------------------------------- sample builders
#
# Non-differentiable ops: ``requires_grad`` is ignored (the forward-only battery
# never differentiates these). To make the DEVICE-PARITY driver -- which rebuilds
# each tensor from its own materialized dtype and ignores the dtype argument --
# actually exercise every integer reduce kernel, each builder emits ONE SampleInput
# per integer dtype, for both the full reduce and a (dim, keepdims) sweep. Negative
# values are included for the signed dtypes (the int reduce miscompile only shows on
# a real, sign-bearing integer kernel; uint8 stays non-negative by definition).
#
# Value ranges are bounded so even the narrowest dtype (int8: [-128, 127]) does NOT
# overflow the forwarded reduction, keeping the numpy oracle exact:
#   * sum  : 24 elems x |v|<=4  -> |sum| <= 96  < 127.
#   * prod : <=6 elems x |v|<=2 -> |prod| <= 64 < 127  (sign-bearing: negatives flip
#            the product sign, the discriminating behaviour a miscompile drops).
# uint8 ranges stay within [0, ...] (its default low is 0); the signed dtypes draw
# from a symmetric window so negatives are present.

_INT_DTYPES = cu.integral_types()   # (uint8, int8, int16, int32, int64)

# 24-element block for sum/max/min (well within int8 range for the bounded sum).
_RED_SHAPE = (2, 3, 4)
# Small block for prod so int8 products cannot overflow.
_PROD_SHAPE = (2, 3)
_PROD_DIMS = (0, 1, -1)


def _int_range(dtype, lo_signed, hi):
    """[low, high) for ``make_tensor``: signed dtypes draw from a symmetric window
    (so negatives appear); uint8 keeps low=0 (it cannot be negative)."""
    return (0 if dtype == cu.uint8 else lo_signed), hi


def _full_and_dim_sweep(seed0, shape, dims, lo_signed, hi):
    """One full-reduce SampleInput + a (dim, keepdims) sweep, replicated once per
    integer dtype so each integer reduce kernel is instantiated by the parity test."""
    out = []
    for di, dtype in enumerate(_INT_DTYPES):
        lo, hi_ = _int_range(dtype, lo_signed, hi)
        # full reduce (no dim kwarg -> kernel full-reduces, yields a (1,)-Var)
        out.append(SampleInput(
            make_tensor(*shape, dtype=dtype, low=lo, high=hi_, seed=seed0 + di)))
        # (dim, keepdims) sweep
        for ki, dim in enumerate(dims):
            for keepdims in (False, True):
                out.append(SampleInput(
                    make_tensor(*shape, dtype=dtype, low=lo, high=hi_,
                                seed=seed0 + 100 + di * 16 + ki * 2 + int(keepdims)),
                    dim=dim, keepdims=keepdims))
    return out


def sample_int_sum(op_info, device, dtype, requires_grad):
    # |v| <= 4 over 24 elems -> |sum| <= 96 < int8 max (127). Negatives present.
    return _full_and_dim_sweep(800, _RED_SHAPE, (0, 1, 2, -1), lo_signed=-4, hi=5)


def sample_int_prod(op_info, device, dtype, requires_grad):
    # |v| in {1, 2} over <=6 elems -> |prod| <= 64 < 127; signs flip the product.
    # Exclude 0 so the product magnitude (and its sign) is a non-trivial check.
    out = []
    for di, dt in enumerate(_INT_DTYPES):
        lo, hi = _int_range(dt, -2, 3)            # signed: [-2,3) -> {-2,-1,0,1,2}
        out.append(SampleInput(
            make_tensor(*_PROD_SHAPE, dtype=dt, low=lo, high=hi,
                        exclude_zero=True, seed=820 + di)))
        for ki, dim in enumerate(_PROD_DIMS):
            for keepdims in (False, True):
                out.append(SampleInput(
                    make_tensor(*_PROD_SHAPE, dtype=dt, low=lo, high=hi,
                                exclude_zero=True,
                                seed=820 + 100 + di * 16 + ki * 2 + int(keepdims)),
                    dim=dim, keepdims=keepdims))
    return out


def sample_int_max(op_info, device, dtype, requires_grad):
    # Wide signed window so the max is sometimes negative (all-negative slices) --
    # a case the float-only tests never reach and a sign-handling miscompile fails.
    return _full_and_dim_sweep(840, _RED_SHAPE, (0, 1, 2, -1), lo_signed=-9, hi=10)


def sample_int_min(op_info, device, dtype, requires_grad):
    return _full_and_dim_sweep(860, _RED_SHAPE, (0, 1, 2, -1), lo_signed=-9, hi=10)


def _bool_full_and_dim_sweep(seed0):
    """all/any over bool: full reduce + a per-dim sweep. NO ``keepdims`` is passed --
    the native ``all_``/``any_`` accept it, but a mix of True/False per slice is what
    we want; keepdims is covered by the int builders. bool ``make_tensor`` draws
    uniform 0/1 so each reduced slice has a non-trivial truth mix."""
    out = [SampleInput(make_tensor(*_RED_SHAPE, dtype=cu.bool_, seed=seed0))]
    for ki, dim in enumerate((0, 1, 2, -1)):
        for keepdims in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype=cu.bool_, seed=seed0 + 1 + ki),
                dim=dim, keepdims=keepdims))
    return out


def sample_bool_all(op_info, device, dtype, requires_grad):
    return _bool_full_and_dim_sweep(880)


def sample_bool_any(op_info, device, dtype, requires_grad):
    return _bool_full_and_dim_sweep(890)


# ------------------------------------------------------------------------ op_db
#
# Distinct ``variant_test_name`` on every entry so the generated test-method names
# (``test_<full_name>`` in the device-parity driver; ``..._<dtype>`` in test_ops) do
# not collide with the float reductions already registered as ``sum``/``prod`` in
# core_ops.py or ``max``/``min``/``all``/``any`` in reductions_extra.py.
#
# All entries: supports_autograd=False (integer/bool reductions are non-diff here) ->
# only the forward battery (vs numpy) and the CPU-vs-accelerator parity run; no
# gradcheck. dtypes=cu.integral_types() makes the test_ops dtype sweep cover every
# integer width, and the sample builders additionally pin all of them for parity.

# CUDA uses packed 32-bit CAS overloads for uint8/int8/int16 add, multiply,
# maximum, and minimum. CANN product now covers every integer dtype in this
# matrix; NPU sum/max/min remain explicit skips until their real-device matrix
# can run without aborting the test process.
_NPU_INT_REDUCE_SKIP = (
    skip("test_reference", device_type="npu",
         reason="KI-BACKEND-001: narrow sum/max/min are unavailable on NPU"),
)

op_db = [
    # ---- integer sum / prod / max / min (full + dim, with negatives) -----------
    OpInfo("sum", op=jt.reduce_add, ref=sum_ref,
           sample_inputs_func=sample_int_sum,
           dtypes=cu.integral_types(), supports_autograd=False,
           variant_test_name="int_reduce", skips=_NPU_INT_REDUCE_SKIP),
    OpInfo("prod", op=jt.reduce_multiply, ref=prod_ref,
           sample_inputs_func=sample_int_prod,
           dtypes=cu.integral_types(), supports_autograd=False,
           variant_test_name="int_reduce"),
    OpInfo("max", op=jt.reduce_maximum, ref=max_ref,
           sample_inputs_func=sample_int_max,
           dtypes=cu.integral_types(), supports_autograd=False,
           variant_test_name="int_reduce", skips=_NPU_INT_REDUCE_SKIP),
    OpInfo("min", op=jt.reduce_minimum, ref=min_ref,
           sample_inputs_func=sample_int_min,
           dtypes=cu.integral_types(), supports_autograd=False,
           variant_test_name="int_reduce", skips=_NPU_INT_REDUCE_SKIP),

    # ---- bool all / any --------------------------------------------------------
    # Bound to the native C++ reduces ``all_``/``any_`` (torch_compat re-wraps the
    # ``Var.all``/``Var.any`` METHODS, not these top-level funcs). A bool input forces
    # an int32 reduce output (reduce_op.cc L289), which the refs above mirror.
    #
    # CUDA logical reductions write an int32 output, so the generated kernels use
    # CUDA's native atomicAnd/atomicOr on that output rather than bool atomics. NPU
    # remains unverified and keeps an explicit skip until the same device matrix is
    # available there.
    OpInfo("all", op=jt.all_, ref=all_ref,
           sample_inputs_func=sample_bool_all,
           dtypes=(cu.bool_,), supports_autograd=False,
           variant_test_name="bool_reduce",
           skips=(skip("test_reference", device_type="npu",
                       reason="logical reduce is not verified on NPU"),)),
    OpInfo("any", op=jt.any_, ref=any_ref,
           sample_inputs_func=sample_bool_any,
           dtypes=(cu.bool_,), supports_autograd=False,
           variant_test_name="bool_reduce",
           skips=(skip("test_reference", device_type="npu",
                       reason="logical reduce is not verified on NPU"),)),
]
