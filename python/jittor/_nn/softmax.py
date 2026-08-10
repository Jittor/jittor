"""Softmax-family implementations exposed through :mod:`jittor.nn`."""

from .runtime import jt, preserve_facade_origins


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
        dim = jt.nn._get_softmax_dim(x.ndim)
    import jittor.other.code_softmax as code_softmax
    if code_softmax.can_softmax_v1(x, dim) and jt.compiler.is_cuda:
        return code_softmax.softmax_v1(x, log)
    dtype, x = x.dtype, x._to_float()
    if log:
        a = x - jt.max(x, dim, keepdims=True)
        ret = a - a.exp().sum(dim, keepdims=True).log()
    else:
        x = (x - jt.max(x, dim, keepdims=True)).exp()
        ret = x / x.sum(dim, keepdims=True)
    return ret.cast(dtype)


def log_softmax(x,dim=None):
    # Backend integrations replace the public softmax symbol at runtime.  Keep
    # this dependency routed through the facade, as it was when both functions
    # lived in the same module namespace.
    return jt.nn.softmax(x,dim=dim, log=True)


def log_sigmoid(x):
    return jt.log(jt.sigmoid(x))


def logsumexp(x, dim, keepdims=False, keepdim=False):
    return x.exp().sum(dim, keepdim or keepdims).log()


_FACADE_SYMBOLS = (
    _get_softmax_dim, softmax, log_softmax, log_sigmoid, logsumexp,
)
preserve_facade_origins(_FACADE_SYMBOLS)
