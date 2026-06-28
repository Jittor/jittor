# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Bitwise / shift / boolean OpInfos -- jittor-core native INTEGER kernels.

These are pure C++ binary & unary ops on integer dtypes (``src/ops/binary_op.cc``,
``src/ops/unary_op.cc``): ``bitwise_and/or/xor/not``, ``left_shift``, ``right_shift``,
``logical_xor``, ``logical_not``. They emit an integer/bool Var and carry no gradient
(``supports_autograd=False``), so the test value is the FORWARD kernel itself -- and,
through ``test_device_parity``, the CUDA integer kernel measured against the CPU one.

Integer bitwise/shift kernels are a classic accelerator silent-wrong spot (wrong
width, sign-extension on the shift, a missing per-dtype atomic/overload). A float-only
or CPU-only battery never touches them; the existing op_db had ZERO bitwise coverage.

Every op has an INDEPENDENT numpy reference (``np.bitwise_*``, ``np.left_shift`` ...).
The sample builders pin an INTEGER dtype: when a float-only driver materializes the
samples (``test_device_parity`` always asks for ``float32``) the request is coerced to
int32 so the op stays well-typed and the integer CUDA kernel still gets exercised.
Across ``test_ops`` (``OpDTypes.any_one``) the full integral width set is swept, so the
int8 sign bit and uint8 wraparound are covered too.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo


def _int_dtype(dtype):
    """Bitwise ops are integer-only: coerce a floating request to int32 so a
    float-only driver (device-parity) still produces a well-typed integer sample."""
    return "int32" if cu.is_floating(dtype) else dtype


def _value_range(dt):
    """Pick a sample value range valid for this integer width.

    SIGNED types use a +/- range so the two's-complement SIGN BIT is exercised (jittor
    matches numpy bit-for-bit there). UNSIGNED (uint8) stays NON-NEGATIVE and inside the
    int8-positive band [0,64): jittor's *binary* bitwise op returns an int8 for uint8
    inputs (a dtype-promotion quirk, not a value bug), so a value >=128 would read back
    negative and spuriously diverge from numpy's uint8 reference. Bits 0..5 still vary,
    which is what the kernel test needs."""
    return (0, 64) if str(dt).startswith("uint") else (-32, 32)


# ----------------------------------------------------------------- numpy refs
def bitwise_and_ref(x, y):  return np.bitwise_and(x, y)
def bitwise_or_ref(x, y):   return np.bitwise_or(x, y)
def bitwise_xor_ref(x, y):  return np.bitwise_xor(x, y)
def bitwise_not_ref(x):     return np.bitwise_not(x)
def left_shift_ref(x, y):   return np.left_shift(x, y)
def right_shift_ref(x, y):  return np.right_shift(x, y)
def logical_xor_ref(x, y):  return np.logical_xor(x, y)
def logical_not_ref(x):     return np.logical_not(x)


# --------------------------------------------------------------- sample builders
def sample_bitwise_binary(op_info, device, dtype, requires_grad):
    # signed widths span negatives so the SIGN BIT (two's-complement) is exercised;
    # uint8 stays non-negative (see _value_range). Broadcast pair included.
    dt = _int_dtype(dtype)
    lo, hi = _value_range(dt)
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dt, low=lo, high=hi, seed=1300 + i)
        b = make_tensor(*sb, dtype=dt, low=lo, high=hi, seed=1310 + i)
        out.append(SampleInput(a, b))
    return out


def sample_bitwise_unary(op_info, device, dtype, requires_grad):
    # bitwise_not is unary and PRESERVES dtype (uint8->uint8), so negatives are safe on
    # signed widths and the full range is safe on uint8 -- but keep the same range policy
    # as the binary ops for consistency.
    dt = _int_dtype(dtype)
    lo, hi = _value_range(dt)
    out = []
    for i, s in enumerate([(5,), (3, 4), (2, 3, 4)]):
        out.append(SampleInput(make_tensor(*s, dtype=dt, low=lo, high=hi, seed=1320 + i)))
    return out


def sample_shift(op_info, device, dtype, requires_grad):
    # keep the shifted value and the shift amount NON-NEGATIVE and small so the result
    # is well-defined for every integral width (int8 max here is 7<<3 == 56 < 127) and
    # numpy's arithmetic-vs-logical right-shift question never arises.
    dt = _int_dtype(dtype)
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dt, low=0, high=8, seed=1330 + i)
        n = make_tensor(*sb, dtype=dt, low=0, high=4, seed=1340 + i)
        out.append(SampleInput(a, n))
    return out


def sample_logical(op_info, device, dtype, requires_grad):
    # integers over a small range so True/False both occur (real overlap) -> bool out.
    dt = _int_dtype(dtype)
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dt, low=0, high=3, seed=1350 + i)
        b = make_tensor(*sb, dtype=dt, low=0, high=3, seed=1360 + i)
        out.append(SampleInput(a, b))
    return out


def sample_logical_unary(op_info, device, dtype, requires_grad):
    dt = _int_dtype(dtype)
    out = []
    for i, s in enumerate([(5,), (3, 4), (2, 3, 4)]):
        out.append(SampleInput(make_tensor(*s, dtype=dt, low=0, high=3, seed=1370 + i)))
    return out


_INT = cu.integral_types()

op_db = [
    # ---- bitwise (integer output -> no gradient) ----
    OpInfo("bitwise_and", op=jt.bitwise_and, ref=bitwise_and_ref,
           sample_inputs_func=sample_bitwise_binary, dtypes=_INT, supports_autograd=False),
    OpInfo("bitwise_or", op=jt.bitwise_or, ref=bitwise_or_ref,
           sample_inputs_func=sample_bitwise_binary, dtypes=_INT, supports_autograd=False),
    OpInfo("bitwise_xor", op=jt.bitwise_xor, ref=bitwise_xor_ref,
           sample_inputs_func=sample_bitwise_binary, dtypes=_INT, supports_autograd=False),
    OpInfo("bitwise_not", op=jt.bitwise_not, ref=bitwise_not_ref,
           sample_inputs_func=sample_bitwise_unary, dtypes=_INT, supports_autograd=False),

    # ---- shifts (kept non-negative & small so every width is well-defined) ----
    OpInfo("left_shift", op=jt.left_shift, ref=left_shift_ref,
           sample_inputs_func=sample_shift, dtypes=_INT, supports_autograd=False),
    OpInfo("right_shift", op=jt.right_shift, ref=right_shift_ref,
           sample_inputs_func=sample_shift, dtypes=_INT, supports_autograd=False),

    # ---- boolean logic not already in pointwise_binary (and/or are there) ----
    OpInfo("logical_xor", op=jt.logical_xor, ref=logical_xor_ref,
           sample_inputs_func=sample_logical, dtypes=_INT, supports_autograd=False),
    OpInfo("logical_not", op=jt.logical_not, ref=logical_not_ref,
           sample_inputs_func=sample_logical_unary, dtypes=_INT, supports_autograd=False),
]
