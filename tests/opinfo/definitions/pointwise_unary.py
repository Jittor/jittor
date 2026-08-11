# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Pointwise-unary OpInfos: math unary ufuncs + activation functions.

The audit's #1 gap was the activation family (leaky_relu, elu, softplus, ...),
whose extra parameters mean they cannot ride the default elementwise sample path
-- each gets an ``OpInfo`` with a ``sample_inputs_func`` that threads the param.
The plain elementwise math ops use ``UnaryUfuncInfo`` (forward pinned to numpy,
backward gradchecked on CPU float64). Concatenated into the registry's ``op_db``.

Resolution notes (verified against jittor source, not guessed):
  * ``jt.{floor,ceil,round,erf,asin,acos,atan,sinh,cosh,asinh,acosh,atanh}`` are
    C++ unary ops (src/ops/unary_op.cc).  ``jt.{rsqrt,expm1,log2}`` come from
    ``jittor.misc`` via ``from .misc import *``.
  * ``reciprocal, exp2, square, trunc, log1p, sign`` are only patched in lazily by
    ``jittor.compat.torch`` (and only as ``F.*`` / ``Var.*``), so they are NOT reliably
    importable here -- we build them from guaranteed primitives instead.
  * floor/ceil/round/trunc/sign have NO backward (unary_op.cc returns nullptr),
    so they are ``supports_autograd=False``.
  * jittor's ``round`` uses C ``roundf`` (half-away-from-zero), NOT numpy's
    round-half-to-even; the ref matches the C semantics.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# --------------------------------------------------------------- numpy refs (math)

def reciprocal_ref(x):
    return 1.0 / x


def rsqrt_ref(x):
    return 1.0 / np.sqrt(x)


def exp2_ref(x):
    return np.exp2(x)


def square_ref(x):
    return x * x


def round_ref(x):
    # jittor round == C roundf: round half AWAY from zero (not numpy's half-to-even).
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _trunc(x):
    # matches torch_compat Var.trunc; jt has no top-level trunc op.
    return jt.ternary(x >= 0, jt.floor(x), jt.ceil(x))


# --------------------------------------------------------- numpy refs (activations)

def leaky_relu_ref(x, scale=0.01):
    return np.where(x > 0, x, x * scale)


def elu_ref(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


def relu6_ref(x):
    return np.minimum(np.maximum(x, 0.0), 6.0)


def softplus_ref(x, beta=1.0, threshold=20.0):
    # mirrors nn.softplus: 1/beta*log(1+exp(min(beta*x,threshold))) + max(x-threshold/beta,0)
    return (1.0 / beta) * np.log(1.0 + np.exp(np.minimum(beta * x, threshold))) \
        + np.maximum(x - threshold / beta, 0.0)


def hardswish_ref(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def hardsigmoid_ref(x):
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


def mish_ref(x):
    return x * np.tanh(softplus_ref(x))


def selu_ref(x):
    a = 1.6732632423543772848170429916717
    s = 1.0507009873554804934193349852946
    return s * (np.maximum(x, 0.0) + np.minimum(0.0, a * (np.exp(x) - 1.0)))


def celu_ref(x, alpha=1.0):
    return np.maximum(x, 0.0) + np.minimum(0.0, alpha * (np.exp(x / alpha) - 1.0))


def tanhshrink_ref(x):
    return x - np.tanh(x)


def softsign_ref(x):
    return x / (1.0 + np.abs(x))


def prelu_ref(x, weight):
    # functional prelu with a scalar (numel==1) slope: max(0,x) + w*min(0,x)
    w = np.asarray(weight).reshape(())
    return np.maximum(0.0, x) + w * np.minimum(0.0, x)


# ----------------------------------------------------------- jittor callables

def _square(x):
    return x * x


def _reciprocal(x):
    return 1.0 / x


def _exp2(x):
    return jt.exp(x * 0.6931471805599453)   # 2**x = exp(x*ln2)


def _selu(x):
    a = 1.6732632423543772848170429916717
    s = 1.0507009873554804934193349852946
    return s * (jt.maximum(x, 0.0) + jt.minimum(0.0, a * (jt.exp(x) - 1.0)))


def _celu(x, alpha=1.0):
    return jt.maximum(x, 0.0) + jt.minimum(0.0, alpha * (jt.exp(x / alpha) - 1.0))


def _tanhshrink(x):
    return x - jt.tanh(x)


# ------------------------------------------------------------ sample builders
# (math ufuncs ride UnaryUfuncInfo's default elementwise sampler via `domain=`;
#  the activations below need a sampler that threads their extra param.)

def sample_leaky_relu(op_info, device, dtype, requires_grad):
    out = []
    for i, scale in enumerate([0.01, 0.2]):
        out.append(SampleInput(
            make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=600 + i),
            scale=scale))
    return out


def sample_elu(op_info, device, dtype, requires_grad):
    out = []
    for i, alpha in enumerate([1.0, 0.5]):
        out.append(SampleInput(
            make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=610 + i),
            alpha=alpha))
    return out


def sample_celu(op_info, device, dtype, requires_grad):
    out = []
    for i, alpha in enumerate([1.0, 0.7]):
        out.append(SampleInput(
            make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=620 + i),
            alpha=alpha))
    return out


def sample_softplus(op_info, device, dtype, requires_grad):
    out = []
    for i, beta in enumerate([1.0, 2.0]):
        out.append(SampleInput(
            make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=630 + i),
            beta=beta))
    return out


def sample_prelu(op_info, device, dtype, requires_grad):
    # scalar (numel==1) weight Var, passed positionally so gradcheck differentiates
    # it too (prelu is genuinely differentiable w.r.t. its slope). numel==1 hits the
    # scalar branch of nn.prelu (no per-channel broadcast).
    out = []
    for i in range(2):
        out.append(SampleInput(
            make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=640 + i),
            make_tensor(1, dtype=dtype, low=0.1, high=0.5,
                        requires_grad=requires_grad, seed=645 + i)))
    return out


op_db = [
    # ---- math unary ufuncs (forward pinned to numpy; backward gradchecked) ----
    UnaryUfuncInfo("reciprocal", ref=reciprocal_ref, domain=(0.2, 4.0), op=_reciprocal),
    UnaryUfuncInfo("rsqrt", ref=rsqrt_ref, domain=(0.2, 4.0), op=jt.rsqrt),
    UnaryUfuncInfo("exp2", ref=exp2_ref, domain=(-3.0, 3.0), op=_exp2),
    UnaryUfuncInfo("expm1", ref=np.expm1, domain=(-3.0, 3.0), op=jt.expm1),
    UnaryUfuncInfo("log1p", ref=np.log1p, domain=(-0.5, 4.0), op=lambda x: jt.log(1.0 + x)),
    UnaryUfuncInfo("log2", ref=np.log2, domain=(0.2, 4.0), op=jt.log2),
    UnaryUfuncInfo("square", ref=square_ref, op=_square),
    UnaryUfuncInfo("erf", ref=erf_ref, op=jt.erf),

    # trig / hyperbolic (1st & 2nd derivatives are composed of differentiable ops)
    UnaryUfuncInfo("asin", ref=np.arcsin, domain=(-0.9, 0.9), op=jt.asin),
    UnaryUfuncInfo("acos", ref=np.arccos, domain=(-0.9, 0.9), op=jt.acos),
    UnaryUfuncInfo("atan", ref=np.arctan, op=jt.atan),
    UnaryUfuncInfo("sinh", ref=np.sinh, domain=(-3.0, 3.0), op=jt.sinh),
    UnaryUfuncInfo("cosh", ref=np.cosh, domain=(-3.0, 3.0), op=jt.cosh),
    UnaryUfuncInfo("asinh", ref=np.arcsinh, domain=(-4.0, 4.0), op=jt.asinh),
    UnaryUfuncInfo("acosh", ref=np.arccosh, domain=(1.1, 4.0), op=jt.acosh),
    UnaryUfuncInfo("atanh", ref=np.arctanh, domain=(-0.9, 0.9), op=jt.atanh),

    # ---- non-differentiable rounding / sign (unary_op.cc backward == nullptr) ----
    UnaryUfuncInfo("floor", ref=np.floor, op=jt.floor, supports_autograd=False),
    UnaryUfuncInfo("ceil", ref=np.ceil, op=jt.ceil, supports_autograd=False),
    UnaryUfuncInfo("round", ref=round_ref, op=jt.round, supports_autograd=False),
    UnaryUfuncInfo("trunc", ref=np.trunc, op=_trunc, supports_autograd=False),
    UnaryUfuncInfo("sign", ref=np.sign, op=nn.sign, supports_autograd=False),

    # ---- activations (no extra param: plain UnaryUfuncInfo) ----
    # supports_gradgrad=False on the piecewise-mask activations: their 1st derivative
    # carries a ternary/clamp step mask, so the 2nd derivative differentiates a
    # comparison (degenerate / not represented by jittor's autodiff). The smooth
    # ones (mish, tanhshrink, softsign) keep gradgrad on.
    UnaryUfuncInfo("relu6", ref=relu6_ref, op=nn.relu6, supports_gradgrad=False),
    UnaryUfuncInfo("hardswish", ref=hardswish_ref, op=nn.hardswish, supports_gradgrad=False),
    UnaryUfuncInfo("hardsigmoid", ref=hardsigmoid_ref, op=nn.hardsigmoid, supports_gradgrad=False),
    UnaryUfuncInfo("selu", ref=selu_ref, op=_selu, supports_gradgrad=False),
    UnaryUfuncInfo("mish", ref=mish_ref, op=nn.mish),
    UnaryUfuncInfo("tanhshrink", ref=tanhshrink_ref, op=_tanhshrink),
    UnaryUfuncInfo("softsign", ref=softsign_ref, op=nn.softsign),

    # ---- activations with an extra (non-differentiated) python scalar param ----
    OpInfo("leaky_relu", op=nn.leaky_relu, ref=leaky_relu_ref,
           sample_inputs_func=sample_leaky_relu, supports_gradgrad=False),
    OpInfo("elu", op=nn.elu, ref=elu_ref, sample_inputs_func=sample_elu,
           supports_gradgrad=False),
    OpInfo("celu", op=_celu, ref=celu_ref, sample_inputs_func=sample_celu,
           supports_gradgrad=False),
    OpInfo("softplus", op=nn.softplus, ref=softplus_ref,
           sample_inputs_func=sample_softplus),

    # ---- prelu: input + a differentiated scalar slope Var ----
    # supports_gradgrad=False: the mixed 2nd derivative d2/dx dw is the min(0,x) mask.
    OpInfo("prelu", op=nn.prelu, ref=prelu_ref, sample_inputs_func=sample_prelu,
           supports_gradgrad=False),
]
