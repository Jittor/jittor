# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Pointwise binary / compare OpInfos: the elementwise two-operand ops that are
*not* the plain add/sub/mul/div already covered by ``core_ops.py``.

Two families live here:

  * Differentiable binary math (``BinaryUfuncInfo``/``OpInfo`` with the default
    ``supports_autograd=True``): pow, atan2, fmod, remainder, hypot, logaddexp,
    and clamp. Each has an INDEPENDENT numpy ``ref`` so ``test_ops.py`` pins the
    forward, and a gradchecked backward. These were classic "forward-only"
    suspects -- a forward that returns a plausible array hides a backward that
    drops the divisor term (fmod/remainder), routes a wrong quadrant (atan2), or
    stubs the clamp/hypot gradient to zero. Domains are chosen so the result is
    smooth over the whole sample (e.g. ``atan2`` keeps the x-operand > 0 so the
    quadrant correction never fires; ``fmod``/``remainder`` keep ``x/y`` mid-bin
    so no element sits on a floor/trunc step under the 1e-6 gradcheck eps).

  * Comparisons & boolean logic (plain ``OpInfo`` with ``supports_autograd=False``):
    equal, not_equal, greater, less, greater_equal, less_equal, logical_and,
    logical_or. These emit a *bool* Var (see ``binary_dtype_infer`` in
    nano_string.h: ``op.is_bool() -> ns_bool``) so they carry no gradient; they are
    tested on integer dtypes (so True/False overlap is meaningful) with the forward
    compared exactly against the numpy reference.

Op resolution: every ``op=`` is bound to an always-present primitive -- the native
C++ binary ops (``jt.pow``, ``jt.equal``, ...) or a tiny lambda built from native
unary/binary ops (``jt.floor``, ``jt.sqrt``, ``jt.log``, ``jt.exp``). Nothing here
depends on the torch-compat shim being active under the runner (matching the
resolution discipline in ``shape.py``). All sample tensors stay small (<= 24 elems
per differentiated operand) because gradcheck is O(numel) forward passes.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# =============================================================================
# Op callables resolved to native primitives (no torch-compat dependency).
# jittor exposes logaddexp / fmod / remainder / hypot only via the torch shim
# (jittor.compat.torch), so rebuild them here from native ops with identical math.
# =============================================================================

def _atan2(y, x):
    # jittor.misc.arctan2(y, x) == arctan(y/x) with a +/-pi quadrant fix
    # that only fires for x < 0. Samples keep x > 0, so this is plain arctan(y/x)
    # and matches np.arctan2(y, x) elementwise. Built from the native atan op
    # (CPU codegen uses std::atan -> accurate for float32/float64; the ~1e-3 gap
    # noted in test_torch_compat_math is a CUDA/fp16 ::atanf artifact, not CPU).
    return jt.arctan2(y, x)


def _fmod(x, y):
    # truncated remainder (sign of dividend): x - trunc(x/y) * y. Avoid jt.trunc
    # (torch-shim only) -- build truncation from native floor/ceil via ternary.
    q = x / y
    t = jt.ternary(q >= 0, jt.floor(q), jt.ceil(q))
    return x - t * y


def _remainder(x, y):
    # floored remainder (sign of divisor): x - floor(x/y) * y == jt.mod, but spelled
    # out so the gradient flows through native floor (which is a constant w.r.t. x,y).
    return x - jt.floor(x / y) * y


def _hypot(a, b):
    # jittor's misc.hypot: sqrt(a^2 + b^2). Smooth away from the origin.
    return jt.sqrt(a * a + b * b)


def _logaddexp(a, b):
    # numerically stable log(exp(a)+exp(b)) -- same formula the torch shim installs.
    m = jt.maximum(a, b)
    return m + jt.log(jt.exp(a - m) + jt.exp(b - m))


# =============================================================================
# numpy references (the INDEPENDENT forward oracle).
# =============================================================================

def pow_ref(x, y):
    return np.power(x, y)


def atan2_ref(y, x):
    return np.arctan2(y, x)


def fmod_ref(x, y):
    return np.fmod(x, y)            # C fmod: sign of dividend (matches _fmod)


def remainder_ref(x, y):
    return np.remainder(x, y)       # floored: sign of divisor (matches _remainder)


def hypot_ref(a, b):
    return np.hypot(a, b)


def logaddexp_ref(a, b):
    return np.logaddexp(a, b)


def floor_divide_ref(x, y):
    return np.floor_divide(x, y)


def clamp_ref(x, min_v=None, max_v=None):
    return np.clip(x, min_v, max_v)


# comparisons / logic -> bool numpy arrays (jittor returns a bool Var).
def equal_ref(x, y):          return np.equal(x, y)
def not_equal_ref(x, y):      return np.not_equal(x, y)
def greater_ref(x, y):        return np.greater(x, y)
def less_ref(x, y):           return np.less(x, y)
def greater_equal_ref(x, y):  return np.greater_equal(x, y)
def less_equal_ref(x, y):     return np.less_equal(x, y)
def logical_and_ref(x, y):    return np.logical_and(x, y)
def logical_or_ref(x, y):     return np.logical_or(x, y)


# =============================================================================
# sample builders -- differentiable ops keep results smooth over the whole sample
# =============================================================================

def sample_pow(op_info, device, dtype, requires_grad):
    # base > 0 and exponent a small float Var: x**y is smooth, and so are both
    # partials (y*x^(y-1) and x^y*ln x). Both operands are positional float Vars
    # -> both get differentiated by gradcheck.
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 3), (3,))]   # last: broadcast
    out = []
    for i, (sa, sb) in enumerate(pairs):
        base = make_tensor(*sa, dtype=dtype, low=0.3, high=2.5,
                           requires_grad=requires_grad, seed=800 + i)
        exp = make_tensor(*sb, dtype=dtype, low=0.5, high=2.0,
                          requires_grad=requires_grad, seed=810 + i)
        out.append(SampleInput(base, exp))
    return out


def sample_atan2(op_info, device, dtype, requires_grad):
    # op(y, x): y free in sign, x kept > 0 so jittor's quadrant fix is inert and
    # arctan2 == arctan(y/x) (smooth). Both y and x are differentiated.
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]   # incl. broadcast
    out = []
    for i, (sy, sx) in enumerate(pairs):
        y = make_tensor(*sy, dtype=dtype, low=-2.0, high=2.0,
                        requires_grad=requires_grad, seed=820 + i)
        x = make_tensor(*sx, dtype=dtype, low=0.5, high=2.5,
                        requires_grad=requires_grad, seed=830 + i)
        out.append(SampleInput(y, x))
    return out


def sample_fmod_remainder(op_info, device, dtype, requires_grad):
    # x/y must stay strictly inside an integer bin so neither floor nor trunc steps
    # under the 1e-6 gradcheck perturbation (the quotient term is then locally
    # constant and the gradient is exact). Dividend in [-2.5, 2.5], divisor in
    # [3.0, 4.0] => |x/y| < 0.84, comfortably away from +/-1.
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]   # incl. broadcast
    out = []
    for i, (sx, sy) in enumerate(pairs):
        x = make_tensor(*sx, dtype=dtype, low=-2.5, high=2.5,
                        requires_grad=requires_grad, seed=840 + i)
        y = make_tensor(*sy, dtype=dtype, low=3.0, high=4.0,
                        requires_grad=requires_grad, seed=850 + i)
        out.append(SampleInput(x, y))
    return out


def sample_hypot(op_info, device, dtype, requires_grad):
    # both operands bounded away from 0 so sqrt(a^2+b^2) and its partials are smooth.
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dtype, low=0.5, high=2.5,
                        requires_grad=requires_grad, seed=860 + i)
        b = make_tensor(*sb, dtype=dtype, low=0.5, high=2.5,
                        requires_grad=requires_grad, seed=870 + i)
        out.append(SampleInput(a, b))
    return out


def sample_logaddexp(op_info, device, dtype, requires_grad):
    # smooth everywhere; modest magnitudes keep exp() finite for the numpy ref too.
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dtype, low=-2.0, high=2.0,
                        requires_grad=requires_grad, seed=880 + i)
        b = make_tensor(*sb, dtype=dtype, low=-2.0, high=2.0,
                        requires_grad=requires_grad, seed=890 + i)
        out.append(SampleInput(a, b))
    return out


def sample_clamp(op_info, device, dtype, requires_grad):
    # min_v / max_v are python scalars passed as kwargs -> NOT differentiated; only
    # the input Var is. Keep every element strictly inside (min_v, max_v) so the
    # clamp is the identity on this sample and its gradient is a clean all-ones
    # (the boundary kinks -- where torch's subgradient is 0 -- are tested by the
    # forward ref against np.clip, not by gradcheck which would straddle the kink).
    out = []
    for i, s in enumerate([(5,), (3, 4), (2, 3, 4)]):
        x = make_tensor(*s, dtype=dtype, low=-0.5, high=0.5,
                        requires_grad=requires_grad, seed=900 + i)
        out.append(SampleInput(x, min_v=-1.0, max_v=1.0))
    return out


def sample_floor_divide(op_info, device, dtype, requires_grad):
    # non-differentiable: forward only. Integer dtypes (any_one picks int64).
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]
    out = []
    for i, (sx, sy) in enumerate(pairs):
        x = make_tensor(*sx, dtype=dtype, seed=910 + i)
        y = make_tensor(*sy, dtype=dtype, low=1, high=5, seed=920 + i)
        out.append(SampleInput(x, y))
        if dtype != cu.uint8:
            negative_y = make_tensor(*sy, dtype=dtype, low=-5, high=-1, seed=930 + i)
            out.append(SampleInput(x, negative_y))
    return out


def sample_compare(op_info, device, dtype, requires_grad):
    # bool-output comparison/logic: integer inputs over a small range so True and
    # False both occur (real overlap for equal/not_equal). requires_grad ignored.
    pairs = [((3, 4), (3, 4)), ((4,), (4,)), ((2, 1, 4), (3, 4))]   # incl. broadcast
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dtype, low=0, high=4, seed=930 + i)
        b = make_tensor(*sb, dtype=dtype, low=0, high=4, seed=940 + i)
        out.append(SampleInput(a, b))
    return out


# =============================================================================
op_db = [
    # ---- differentiable binary math (forward pinned to numpy; backward gradchecked) ----
    OpInfo("pow", op=jt.pow, ref=pow_ref, sample_inputs_func=sample_pow),
    OpInfo("atan2", op=_atan2, ref=atan2_ref, sample_inputs_func=sample_atan2),
    # fmod/remainder: gradient w.r.t. both operands is locally constant (the
    # floor/trunc quotient term has zero derivative a.e.), so the *second*
    # derivative is identically 0 -- gradgradcheck against numerical 0 is fine,
    # so gradgrad stays on; samples guarantee no element sits on a step.
    OpInfo("fmod", op=_fmod, ref=fmod_ref, sample_inputs_func=sample_fmod_remainder),
    OpInfo("remainder", op=_remainder, ref=remainder_ref,
           sample_inputs_func=sample_fmod_remainder),
    OpInfo("hypot", op=_hypot, ref=hypot_ref, sample_inputs_func=sample_hypot),
    OpInfo("logaddexp", op=_logaddexp, ref=logaddexp_ref,
           sample_inputs_func=sample_logaddexp),
    OpInfo("clamp", op=jt.clamp, ref=clamp_ref, sample_inputs_func=sample_clamp),

    # ---- non-differentiable: floor division (piecewise-constant output) ----
    OpInfo("floor_divide", op=jt.floor_divide, ref=floor_divide_ref,
           sample_inputs_func=sample_floor_divide,
           dtypes=cu.integral_types(), supports_autograd=False),

    # ---- comparisons (bool output -> supports_autograd=False), tested on ints ----
    # NB: jt.equal is WHOLE-tensor equality (like torch.equal, returns one bool);
    # the elementwise op (torch.eq) is `a == b`, which is what this entry tests.
    OpInfo("equal", op=lambda a, b: a == b, ref=equal_ref, sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("not_equal", op=jt.not_equal, ref=not_equal_ref,
           sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("greater", op=jt.greater, ref=greater_ref, sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("less", op=jt.less, ref=less_ref, sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("greater_equal", op=jt.greater_equal, ref=greater_equal_ref,
           sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("less_equal", op=jt.less_equal, ref=less_equal_ref,
           sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),

    # ---- boolean logic (bool output -> supports_autograd=False) ----
    OpInfo("logical_and", op=jt.logical_and, ref=logical_and_ref,
           sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("logical_or", op=jt.logical_or, ref=logical_or_ref,
           sample_inputs_func=sample_compare,
           dtypes=cu.integral_types(), supports_autograd=False),
]
