"""Public pooling functions with shared parameter normalization."""

from .core_2d import _pool2d, _pool2d_parameters
from .core_3d import _pool3d, _pool3d_parameters

def _no_dilation(dilation):
    # torch's default dilation=1 (or (1,1)) means *no* dilation, which jittor's
    # Pool expresses as None. Normalize int/tuple/list all-ones -> None.
    if dilation is None or dilation == 1: return True
    if isinstance(dilation, (tuple, list)): return all(d == 1 for d in dilation)
    return False

def pool(x, kernel_size, op, padding=0, stride=None):
    return _pool2d(x, **_pool2d_parameters(kernel_size, stride, padding, op=op))

def pool3d(x, kernel_size, op, padding=0, stride=None):
    return _pool3d(x, **_pool3d_parameters(kernel_size, stride, padding, op=op))

def argmax_pool(x, size, stride, padding=0):
    if stride <= 0:
        raise RuntimeError("stride must be > 0, but got {}".format(stride))
    return pool(x, size, "maximum", padding, stride)

def max_pool2d(x=None, kernel_size=None, stride=None, padding=0, dilation=None,
               return_indices=None, ceil_mode=False, input=None):
    if x is None:
        x = input
    if _no_dilation(dilation):
        dilation = None
    return _pool2d(x, **_pool2d_parameters(
        kernel_size, stride, padding, dilation, return_indices, ceil_mode,
        op="maximum",
    ))

def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=None,
               return_indices=None, ceil_mode=False):
    if _no_dilation(dilation):
        dilation = None
    return _pool3d(x, **_pool3d_parameters(
        kernel_size, stride, padding, dilation, return_indices, ceil_mode,
        op="maximum",
    ))

pool2d = pool
