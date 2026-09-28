"""Softmax-family implementations exposed through :mod:`jittor.nn`."""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name


def _get_softmax_dim(ndim):
    # Mirrors torch.nn.functional._get_softmax_dim: when ``dim`` is not given,
    # torch softmaxes over dim 0 for 0/1/3-D inputs and dim 1 otherwise.
    if ndim == 0 or ndim == 1 or ndim == 3:
        return 0
    return 1


def softmax(x, dim=None, log=False):
    # torch-compatible default: ``dim=None`` selects a single axis via
    # ``_get_softmax_dim`` (NOT a reduction over all elements). Passing an
    # explicit ``dim`` keeps the previous behavior unchanged.
    if dim is None:
        dim = _get_softmax_dim(x.ndim)
    if isinstance(dim, int) and not -max(x.ndim, 1) <= dim < max(x.ndim, 1):
        raise IndexError("softmax dimension out of range for input rank {}".format(x.ndim))
    # ``jittor.backends.cuda.kernels.nn.softmax_cuda`` owns the dispatch entry
    # for every backend's fused softmax, ACL's included, so it cannot be
    # imported at module scope -- the backend packages import ``jittor.nn``.
    # The import statement below walks five package objects to hand back a
    # module that is already in ``sys.modules``, and it used to run on every
    # single softmax, so the module object is kept on this function. The memo
    # lives on the function rather than in a module global because
    # ``tests/nn/test_acl_registry_routing.py`` executes this definition on its
    # own, in a namespace built from the ``FunctionDef`` nodes of this file, and
    # a module-level assignment is not one of those. The *attribute* is still
    # read per call, so replacing ``_softmax_v1`` is still honoured.
    softmax_cuda = getattr(softmax, "_backend", None)
    if softmax_cuda is None:
        from jittor.backends.cuda.kernels.nn import softmax_cuda
        softmax._backend = softmax_cuda

    fused = softmax_cuda._softmax_v1(x, log=log, dim=dim)
    if fused is not None:
        return fused
    dtype, x = x.dtype, x._to_float()
    if log:
        a = x - jt.max(x, dim, keepdims=True)
        ret = a - a.exp().sum(dim, keepdims=True).log()
    else:
        x = (x - jt.max(x, dim, keepdims=True)).exp()
        ret = x / x.sum(dim, keepdims=True)
    return ret.cast(dtype)


def log_softmax(x,dim=None):
    # Both spellings share the same native parameter checks and backend table.
    return jt.nn.softmax(x,dim=dim, log=True)


def log_sigmoid(x):
    return jt.log(jt.sigmoid(x))


def logsumexp(x, dim, keepdims=False, keepdim=False):
    keep = keepdim or keepdims
    maximum = jt.max(x, dim, keepdims=True)
    result = (x - maximum).exp().sum(dim, keepdims=True).log() + maximum
    # The shift-exp-sum-log chain runs in float32 for a half input -- `exp` is
    # a white-list op, so it widens whatever it is handed -- and that is the
    # right place to compute it. But the *result* has to come back at the
    # input's dtype: torch 2.13's `logsumexp` answers float16 for a float16
    # input and bfloat16 for a bfloat16 one, on CPU and CUDA, and this returned
    # a float32, which then infects everything downstream of it.
    if _jittor_dtype_name(x.dtype) in ("float16", "bfloat16") and \
            _jittor_dtype_name(result.dtype) != _jittor_dtype_name(x.dtype):
        result = result.cast(x.dtype)
    if keep:
        return result
    # `dim` may name several axes -- torch takes a tuple here, and einops
    # reduces over all of them at once. `Var.squeeze` takes one axis, so
    # handing it the tuple raised `TypeError: '<' not supported between
    # instances of 'tuple' and 'int'`. Drop them highest-first so the
    # remaining indices do not shift under each other.
    if isinstance(dim, (tuple, list)):
        for axis in sorted((a if a >= 0 else a + x.ndim for a in dim),
                           reverse=True):
            result = result.squeeze(axis)
        return result
    return result.squeeze(dim)
