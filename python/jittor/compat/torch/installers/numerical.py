"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt
from jittor import nn
import numpy as np
from builtins import all as _py_all, any as _py_any
from collections import namedtuple as _namedtuple

from ..functional import (
    _diff, _isin, _repeat_interleave, _trapz,
)
from ..grad import (
    _AutocastContext,
)
from ..nested import (
    _NestedTensor,
)
from ..types import (
    _dtype_to_str,
)
from ..fidelity import Fidelity, register_fidelity
from ..context import getitem_transform_active
from ...diagnostics import EXPECTED, swallowed

_vmap_runtime_impl = None


def vmap(func, in_dims=0, out_dims=0, *args, **kwargs):
    if _vmap_runtime_impl is None:
        raise RuntimeError("torch.vmap runtime owner is not installed")
    return _vmap_runtime_impl(func, in_dims, out_dims, *args, **kwargs)


register_fidelity(
    "torch.vmap", vmap, Fidelity.APPROXIMATE,
    "delegates Torch vmap batching to the compatibility runtime; device, "
    "randomness, and unsupported kwargs follow the installed backend policy",
)

autocast = _AutocastContext
register_fidelity(
    "torch.autocast",
    autocast,
    Fidelity.APPROXIMATE,
    "matches Torch context/decorator enable semantics on supported CPU/CUDA "
    "paths; cache, device-specific dtype, and unsupported dtype diagnostics "
    "remain compatibility-layer limitations",
)

_native_all = jt.all
_native_any = jt.any


def _reduce_alias(native, input, dim=None, keepdim=False, *, axis=None,
                  keepdims=None, out=None):
    d = axis if axis is not None else dim
    kd = keepdims if keepdims is not None else keepdim
    if d is None or d == ():
        return native(input)
    result = native(input, d)
    if kd:
        dims = (d,) if isinstance(d, int) else tuple(d)
        for dd in sorted(x % input.ndim for x in dims):
            result = result.unsqueeze(dd)
    return result


def all(input, dim=None, keepdim=False, *, axis=None, keepdims=None, out=None):
    return _reduce_alias(_native_all, input, dim, keepdim,
                         axis=axis, keepdims=keepdims, out=out)


def any(input, dim=None, keepdim=False, *, axis=None, keepdims=None, out=None):
    return _reduce_alias(_native_any, input, dim, keepdim,
                         axis=axis, keepdims=keepdims, out=out)


for _reduce_name, _reduce_impl in (("all", all), ("any", any)):
    register_fidelity(
        "torch." + _reduce_name,
        _reduce_impl,
        Fidelity.APPROXIMATE,
        "matches Torch boolean reduction values and keepdims/axis shape for "
        "CPU tensors; device, dtype, named-dimension, and out semantics are "
        "not implemented",
    )
del _reduce_name, _reduce_impl


_COMPLEX_FIDELITY_DETAIL = (
    "matches Torch complex construction and real/imag round-trips for CPU "
    "real tensors; device, layout, out, and complex128 semantics are not "
    "implemented by the Jittor complex owner"
)


def complex(real, imag, **kwargs):
    """Construct a native complex tensor from real and imaginary parts."""
    return jt.nn.view_as_complex(jt.stack([real, imag], dim=-1))


def view_as_complex(input):
    """Interpret the trailing size-two dimension as a complex tensor."""
    return jt.nn.view_as_complex(input)


def view_as_real(input):
    """Expose native complex values as a trailing size-two real dimension."""
    return jt.nn.view_as_real(input)


for _complex_name, _complex_impl in (
    ("complex", complex),
    ("view_as_complex", view_as_complex),
    ("view_as_real", view_as_real),
):
    register_fidelity(
        "torch." + _complex_name,
        _complex_impl,
        Fidelity.APPROXIMATE,
        _COMPLEX_FIDELITY_DETAIL,
    )
del _complex_name, _complex_impl


def _is_complex_value(value):
    complex_type = jt.nn.ComplexNumber
    return isinstance(value, complex_type) or (
        isinstance(value, jt.Var) and "complex" in str(value.dtype)
    )


def is_complex(input):
    return _is_complex_value(input)


def real(input):
    return input.real if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else input


def imag(input):
    return input.imag if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else jt.zeros_like(input)


def conj(input):
    return input.conj() if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else input


def angle(input):
    return input.angle() if isinstance(input, (jt.nn.ComplexNumber, jt.Var)) else jt.zeros_like(input)


_native_abs = jt.abs


def abs(input):
    return input.abs() if isinstance(input, jt.nn.ComplexNumber) else _native_abs(input)


for _accessor_name, _accessor_impl in (
    ("is_complex", is_complex), ("real", real), ("imag", imag),
    ("conj", conj), ("angle", angle), ("abs", abs),
):
    register_fidelity(
        "torch." + _accessor_name,
        _accessor_impl,
        Fidelity.APPROXIMATE,
        "matches Torch complex accessor values for CPU tensors; device, "
        "layout, out, and complex128 semantics are not implemented",
    )
del _accessor_name, _accessor_impl


def polar(abs, angle, **kwargs):
    """Construct a native complex tensor from magnitude and phase."""
    return jt.nn.polar(abs, angle)


register_fidelity(
    "torch.polar",
    polar,
    Fidelity.APPROXIMATE,
    "matches Torch magnitude/phase values for CPU real tensors; device, "
    "layout, out, and dtype keyword semantics are not implemented",
)


def eye(n, m=None, dtype=None, **kwargs):
    """Create a square or rectangular identity matrix."""
    shape = (int(n), int(n)) if m is None else (int(n), int(m))
    import jittor.init as _init
    return _init.eye(shape, _dtype_to_str(dtype) or "float32")


_PAIRWISE_DISTANCE_FIDELITY_DETAIL = (
    "matches Torch p-norm distance values and keepdim shape through Jittor's "
    "native nn implementation but omits device, layout, and dtype keyword semantics"
)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """Compute p-norm distances between corresponding rows of two tensors."""
    return nn.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)


register_fidelity(
    "torch.pairwise_distance",
    pairwise_distance,
    Fidelity.APPROXIMATE,
    _PAIRWISE_DISTANCE_FIDELITY_DETAIL,
)


_COSINE_SIMILARITY_FIDELITY_DETAIL = (
    "matches Torch cosine similarity values, dim reduction shape, and the eps "
    "denominator floor through Jittor's native nn implementation but omits "
    "device, layout, and dtype keyword semantics"
)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Compute cosine similarity along a tensor dimension."""
    return nn.cosine_similarity(x1, x2, dim=dim, eps=eps)


register_fidelity(
    "torch.cosine_similarity",
    cosine_similarity,
    Fidelity.APPROXIMATE,
    _COSINE_SIMILARITY_FIDELITY_DETAIL,
)


_SVD_FIDELITY_DETAIL = (
    "matches Torch real-matrix decomposition values through Jittor's native "
    "linalg.svd but omits some/compute_uv/driver, device, and dtype keyword semantics"
)


def svd(x, some=True, compute_uv=True, **kwargs):
    """Compute a singular value decomposition via Jittor's native linalg owner."""
    return jt.linalg.svd(x)


register_fidelity(
    "torch.svd",
    svd,
    Fidelity.APPROXIMATE,
    _SVD_FIDELITY_DETAIL,
)


_SVD_LOWRANK_FIDELITY_DETAIL = (
    "matches Torch low-rank decomposition outputs through Jittor's native SVD "
    "for supported real matrices but omits niter, device, and dtype semantics"
)


def svd_lowrank(A, q=6, niter=2, M=None):
    """Return a rank-``q`` approximation using Jittor's native SVD."""
    if M is not None:
        A = A - M
    u, s, v = jt.linalg.svd(A)
    q = min(q, s.shape[0])
    return u[:, :q], s[:q], v[:, :q]


register_fidelity(
    "torch.svd_lowrank",
    svd_lowrank,
    Fidelity.APPROXIMATE,
    _SVD_LOWRANK_FIDELITY_DETAIL,
)


_PCA_LOWRANK_FIDELITY_DETAIL = (
    "matches Torch centered low-rank decomposition through the compatibility "
    "SVD owner for supported real matrices but omits niter, device, and dtype semantics"
)


def pca_lowrank(A, q=6, center=True, niter=2):
    """Compute a low-rank decomposition after optional feature centering."""
    centered = A - (A.mean(0, keepdims=True) if center else 0)
    return svd_lowrank(centered, q=q, niter=niter)


register_fidelity(
    "torch.pca_lowrank",
    pca_lowrank,
    Fidelity.APPROXIMATE,
    _PCA_LOWRANK_FIDELITY_DETAIL,
)


class _SparseCOO:
    def __init__(self, dense): self._dense = dense
    def to_dense(self): return self._dense
    @property
    def shape(self): return self._dense.shape
    @property
    def dtype(self): return self._dense.dtype
    def t(self): return _SparseCOO(self._dense.t())
    def sum(self, dim=None):
        return _SparseCOO(self._dense.sum(dim) if dim is not None else self._dense.sum())


def sparse_coo_tensor(indices, values, size=None, dtype=None, device=None,
                      requires_grad=False, **kwargs):
    """Materialize a dense-backed COO compatibility tensor."""
    if not isinstance(indices, jt.Var): indices = jt.array(indices)
    if not isinstance(values, jt.Var): values = jt.array(values)
    rank = int(indices.shape[0])
    nnz = int(indices.shape[1]) if indices.ndim == 2 else int(indices.shape[0])
    tail = [int(d) for d in values.shape[1:]]
    idx_np = indices.numpy().astype("int64").reshape(rank, -1)
    if size is not None:
        full = [int(s) for s in size]
    else:
        full = [int(idx_np[s].max()) + 1 if nnz > 0 else 0 for s in range(rank)] + tail
    sparse_shape, tail_shape = full[:rank], full[rank:]
    prod = 1
    for d in sparse_shape: prod *= int(d)
    linear = np.zeros(nnz, dtype="int64")
    stride = 1
    for s in range(rank - 1, -1, -1):
        linear = linear + idx_np[s] * stride
        stride *= int(sparse_shape[s])
    flat = jt.zeros([prod] + tail_shape, dtype=str(values.dtype))
    if nnz > 0:
        flat.index_add_(0, jt.array(linear), values.reshape([nnz] + tail_shape))
    return _SparseCOO(flat.reshape(sparse_shape + tail_shape))


_SPARSE_COO_TENSOR_FIDELITY_DETAIL = (
    "matches Torch COO indices/values materialization through a dense-backed "
    "compatibility object but omits sparse storage, device, and dtype semantics"
)
register_fidelity(
    "torch.sparse_coo_tensor",
    sparse_coo_tensor,
    Fidelity.APPROXIMATE,
    _SPARSE_COO_TENSOR_FIDELITY_DETAIL,
)


_RANDINT_LIKE_FIDELITY_DETAIL = (
    "matches Torch integer bounds and shape, with optional dtype casting, for "
    "supported tensors but omits device and requires_grad keyword semantics"
)


def randint_like(input, low, high=None, dtype=None, device=None,
                 requires_grad=False, **kwargs):
    """Sample integer values with the shape of ``input``."""
    if high is None:
        low, high = 0, low
    result = jt.randint(int(low), int(high), tuple(int(s) for s in input.shape))
    return result.cast(_dtype_to_str(dtype)) if dtype is not None else result


register_fidelity(
    "torch.randint_like",
    randint_like,
    Fidelity.APPROXIMATE,
    _RANDINT_LIKE_FIDELITY_DETAIL,
)


_DET_FIDELITY_DETAIL = (
    "matches Torch determinant values through Jittor's native linalg owner for "
    "supported square real tensors but omits device, layout, and dtype semantics"
)


def det(input):
    """Compute a matrix determinant via Jittor's native linalg owner."""
    return jt.linalg.det(input)


register_fidelity(
    "torch.det",
    det,
    Fidelity.APPROXIMATE,
    _DET_FIDELITY_DETAIL,
)


_INVERSE_FIDELITY_DETAIL = (
    "matches Torch matrix inverse values through Jittor's native linalg owner for "
    "supported square real tensors but omits device, layout, and dtype semantics"
)


def inverse(input):
    """Compute a matrix inverse via Jittor's native linalg owner."""
    return jt.linalg.inv(input)


register_fidelity(
    "torch.inverse",
    inverse,
    Fidelity.APPROXIMATE,
    _INVERSE_FIDELITY_DETAIL,
)


_TAKE_ALONG_DIM_FIDELITY_DETAIL = (
    "matches Torch gather values and broadcasted index shape for supported "
    "integer indices but omits out, device, layout, and dtype keyword semantics"
)


def take_along_dim(input, indices, dim=None):
    """Gather values after broadcasting indices outside the gather dimension."""
    if dim is None:
        return jt.gather(input.reshape(-1), 0, indices.reshape(-1))
    d = dim % input.ndim
    target = list(input.shape)
    target[d] = indices.shape[d]
    if list(indices.shape) != target:
        indices = jt.broadcast(indices, target)
    return jt.gather(input, d, indices)


register_fidelity(
    "torch.take_along_dim",
    take_along_dim,
    Fidelity.APPROXIMATE,
    _TAKE_ALONG_DIM_FIDELITY_DETAIL,
)


_LOG1P_FIDELITY_DETAIL = (
    "matches Torch elementwise log1p values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)


def log1p(x):
    """Compute ``log(1 + x)`` elementwise."""
    return jt.log(1.0 + x)


register_fidelity(
    "torch.log1p",
    log1p,
    Fidelity.APPROXIMATE,
    _LOG1P_FIDELITY_DETAIL,
)


_RECIPROCAL_FIDELITY_DETAIL = (
    "matches Torch elementwise reciprocal values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)


def reciprocal(x):
    """Compute the elementwise multiplicative reciprocal."""
    return 1.0 / x


register_fidelity(
    "torch.reciprocal",
    reciprocal,
    Fidelity.APPROXIMATE,
    _RECIPROCAL_FIDELITY_DETAIL,
)


_LERP_FIDELITY_DETAIL = (
    "matches Torch linear interpolation values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)


def lerp(input, end, weight):
    """Linearly interpolate between ``input`` and ``end``."""
    return input + weight * (end - input)


register_fidelity(
    "torch.lerp",
    lerp,
    Fidelity.APPROXIMATE,
    _LERP_FIDELITY_DETAIL,
)


_SOFTMAX_FIDELITY_DETAIL = (
    "matches Torch softmax values along an explicit dimension through Jittor's "
    "native nn owner but omits dtype, device, and layout keyword semantics"
)


def softmax(input, dim=None, **kwargs):
    """Compute softmax values through Jittor's native nn owner."""
    return jt.nn.softmax(input, dim=dim)


register_fidelity(
    "torch.softmax",
    softmax,
    Fidelity.APPROXIMATE,
    _SOFTMAX_FIDELITY_DETAIL,
)


_LOG_SOFTMAX_FIDELITY_DETAIL = (
    "matches Torch log-softmax values along an explicit dimension through Jittor's "
    "native nn owner but omits dtype, device, and layout keyword semantics"
)


def log_softmax(input, dim=None, **kwargs):
    """Compute log-softmax values through Jittor's native nn owner."""
    return jt.nn.log_softmax(input, dim=dim)


register_fidelity(
    "torch.log_softmax",
    log_softmax,
    Fidelity.APPROXIMATE,
    _LOG_SOFTMAX_FIDELITY_DETAIL,
)


_RELU_FIDELITY_DETAIL = (
    "matches Torch elementwise ReLU values through Jittor's native nn owner but "
    "omits inplace, device, layout, and dtype keyword semantics"
)


def relu(input, **kwargs):
    """Compute elementwise rectified linear activation."""
    return jt.nn.relu(input)


register_fidelity(
    "torch.relu",
    relu,
    Fidelity.APPROXIMATE,
    _RELU_FIDELITY_DETAIL,
)


_SHAPE_AS_TENSOR_FIDELITY_DETAIL = (
    "matches Torch int64 shape materialization for supported tensors but omits "
    "device, layout, and dynamic-shape keyword semantics"
)


def _shape_as_tensor(input):
    """Return the static tensor shape as an int64 Jittor array."""
    return jt.array(np.asarray(input.shape, dtype=np.int64))


register_fidelity(
    "torch._shape_as_tensor",
    _shape_as_tensor,
    Fidelity.APPROXIMATE,
    _SHAPE_AS_TENSOR_FIDELITY_DETAIL,
)


# outer / tensordot / repeat_interleave are natively owned by jittor.misc, and
# their native signature already is the Torch one. A forwarding wrapper here
# would be a second function object for the same API: `torch.repeat_interleave
# is jittor.repeat_interleave` then stops holding, which is exactly what
# tests/structure/test_misc_structure.py pins (it also pins that the object
# still pickles back to the misc owner, which a compat wrapper cannot do).
# So these three are re-exported, not wrapped, and the fidelity record points at
# the native implementation.
_NATIVE_OWNER_FIDELITY_DETAIL = (
    "re-exports Jittor's native implementation, whose signature and values "
    "already match Torch for supported real tensors; device, layout, and dtype "
    "keyword semantics are not implemented, and out is not accepted"
)

outer = jt.outer

register_fidelity(
    "torch.outer",
    outer,
    Fidelity.APPROXIMATE,
    _NATIVE_OWNER_FIDELITY_DETAIL,
)


_NATIVE_ISIN = jt.isin
_ISIN_FIDELITY_DETAIL = (
    "re-exports Jittor's native isin implementation for supported tensors but "
    "omits device, layout, and dtype keyword semantics"
)


def isin(elements, test_elements, assume_unique=False, invert=False, **kwargs):
    """Test element membership using the captured native owner."""
    return _NATIVE_ISIN(
        elements, test_elements, assume_unique=assume_unique, invert=invert)


register_fidelity(
    "torch.isin",
    isin,
    Fidelity.APPROXIMATE,
    _ISIN_FIDELITY_DETAIL,
)


tensordot = jt.tensordot

register_fidelity(
    "torch.tensordot",
    tensordot,
    Fidelity.APPROXIMATE,
    _NATIVE_OWNER_FIDELITY_DETAIL,
)


repeat_interleave = jt.repeat_interleave

register_fidelity(
    "torch.repeat_interleave",
    repeat_interleave,
    Fidelity.APPROXIMATE,
    _NATIVE_OWNER_FIDELITY_DETAIL,
)


_NAN_TO_NUM_INPLACE_FIDELITY_DETAIL = (
    "matches Torch in-place NaN/Inf replacement and return identity for supported "
    "real tensors but omits device, layout, dtype, and narrow custom-bound semantics"
)


def nan_to_num_(input, nan=0.0, posinf=None, neginf=None):
    """Replace non-finite values in-place and return the input tensor."""
    result = input.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
    try:
        input.assign(result)
        return input
    except EXPECTED as exc:
        swallowed(
            "torch/installers/numerical.py nan_to_num_: input.assign(result); return input",
            exc,
        )
        return result


register_fidelity(
    "torch.nan_to_num_",
    nan_to_num_,
    Fidelity.APPROXIMATE,
    _NAN_TO_NUM_INPLACE_FIDELITY_DETAIL,
)


_STACKING_FIDELITY_DETAIL = (
    "matches Torch values and shapes for tensor inputs but omits Torch "
    "device, dtype, layout, pin-memory, and out keyword semantics"
)


def _vstack_impl(tensors):
    tensors = list(tensors)
    return jt.concat(
        [t if t.ndim >= 2 else t.reshape((1, -1)) for t in tensors], dim=0)


def vstack(tensors):
    """Stack tensors vertically using the Torch-compatible shape rules."""
    return _vstack_impl(tensors)


def row_stack(tensors):
    """Alias of :func:`vstack` with Torch's historical spelling."""
    return _vstack_impl(tensors)


def hstack(tensors):
    """Stack one-dimensional tensors along 0 and higher-rank tensors along 1."""
    tensors = list(tensors)
    dim = 0 if _py_all(t.ndim == 1 for t in tensors) else 1
    return jt.concat(tensors, dim=dim)


def dstack(tensors):
    """Stack tensors along the third dimension."""
    out = []
    for t in list(tensors):
        out.append(t.reshape((1, -1, 1)) if t.ndim == 1
                   else (t.unsqueeze(-1) if t.ndim == 2 else t))
    return jt.concat(out, dim=2)


def column_stack(tensors):
    """Stack one-dimensional tensors as columns."""
    tensors = list(tensors)
    return jt.concat(
        [t.reshape((-1, 1)) if t.ndim == 1 else t for t in tensors], dim=1)


for _stacking_name in ("vstack", "row_stack", "hstack", "dstack", "column_stack"):
    register_fidelity(
        "torch." + _stacking_name,
        globals()[_stacking_name],
        Fidelity.APPROXIMATE,
        _STACKING_FIDELITY_DETAIL,
    )
del _stacking_name


_MOVEDIM_FIDELITY_DETAIL = (
    "matches Torch axis permutation for valid tensor inputs but omits Torch "
    "layout, device, out, and named-dimension semantics"
)


def _movedim_impl(x, source, destination):
    nd = x.ndim
    src = [s % nd for s in (
        source if isinstance(source, (list, tuple)) else [source])]
    dst = [d % nd for d in (
        destination if isinstance(destination, (list, tuple)) else [destination])]
    order = [d for d in range(nd) if d not in src]
    for d, s in sorted(zip(dst, src)):
        order.insert(d, s)
    return x.permute(order)


def movedim(x, source, destination):
    """Move tensor dimensions using Torch-compatible axis numbering."""
    return _movedim_impl(x, source, destination)


def moveaxis(x, source, destination):
    """Alias of :func:`movedim` with NumPy-compatible naming."""
    return _movedim_impl(x, source, destination)


for _movedim_name in ("movedim", "moveaxis"):
    register_fidelity(
        "torch." + _movedim_name,
        globals()[_movedim_name],
        Fidelity.APPROXIMATE,
        _MOVEDIM_FIDELITY_DETAIL,
    )
del _movedim_name


_SHAPE_HELPER_FIDELITY_DETAIL = (
    "matches Torch values and shapes for valid tensor inputs but omits Torch "
    "device, layout, named-dimension, and out keyword semantics"
)


def _unflatten_impl(input, dim, sizes):
    d = dim % input.ndim
    return input.reshape(
        list(input.shape[:d]) + list(sizes) + list(input.shape[d + 1:]))


def unflatten(input, dim, sizes):
    """Unflatten one tensor dimension according to Torch shape rules."""
    return _unflatten_impl(input, dim, sizes)


def _swapaxes_impl(input, axis0, axis1):
    perm = list(range(input.ndim))
    a, b = axis0 % input.ndim, axis1 % input.ndim
    perm[a], perm[b] = perm[b], perm[a]
    return input.permute(perm)


def swapaxes(input, axis0, axis1):
    """Swap two tensor dimensions."""
    return _swapaxes_impl(input, axis0, axis1)


def swapdims(input, axis0, axis1):
    """Alias of :func:`swapaxes`."""
    return _swapaxes_impl(input, axis0, axis1)


def _ravel_impl(input):
    return input.reshape((-1,))


def ravel(input):
    """Flatten a tensor to one dimension."""
    return _ravel_impl(input)


for _shape_helper_name in ("unflatten", "swapaxes", "swapdims", "ravel"):
    register_fidelity(
        "torch." + _shape_helper_name,
        globals()[_shape_helper_name],
        Fidelity.APPROXIMATE,
        _SHAPE_HELPER_FIDELITY_DETAIL,
    )
del _shape_helper_name


_ELEMENTWISE_FIDELITY_DETAIL = (
    "matches Torch values for supported real tensor inputs but omits Torch "
    "device, layout, and out keyword semantics"
)


def _copysign_impl(input, other):
    sign = (other >= 0).float32() * 2 - 1
    return jt.abs(input) * sign


def copysign(input, other):
    """Copy the sign of ``other`` onto the magnitude of ``input``."""
    return _copysign_impl(input, other)


def _xlogy_impl(input, other):
    return jt.ternary(
        input == 0, jt.zeros_like(input), input * jt.log(other))


def xlogy(input, other):
    """Return ``input * log(other)`` with the Torch ``xlogy(0, y) == 0`` rule."""
    return _xlogy_impl(input, other)


def _heaviside_impl(input, values):
    return (input > 0).float32() + (input == 0).float32() * values


def heaviside(input, values):
    """Return the elementwise Heaviside step function."""
    return _heaviside_impl(input, values)


def _signbit_impl(input):
    return input < 0


def signbit(input):
    """Return a boolean tensor identifying negative values."""
    return _signbit_impl(input)


for _elementwise_name in ("copysign", "xlogy", "heaviside", "signbit"):
    register_fidelity(
        "torch." + _elementwise_name,
        globals()[_elementwise_name],
        Fidelity.APPROXIMATE,
        _ELEMENTWISE_FIDELITY_DETAIL,
    )
del _elementwise_name


_FLOAT_POWER_FIDELITY_DETAIL = (
    "computes supported real inputs in float64 like Torch float_power but "
    "omits device, layout, and out keyword semantics"
)


def _float_power_impl(input, exponent):
    if isinstance(exponent, jt.Var):
        exponent = exponent.float64()
    return input.float64() ** exponent


def float_power(input, exponent):
    """Raise tensors to a power using Torch's float64 computation policy."""
    return _float_power_impl(input, exponent)


register_fidelity(
    "torch.float_power",
    float_power,
    Fidelity.APPROXIMATE,
    _FLOAT_POWER_FIDELITY_DETAIL,
)


_MATRIX_FIDELITY_DETAIL = (
    "matches Torch values for supported real tensor inputs but omits Torch "
    "offset, dimension, device, layout, and out keyword semantics"
)


def _trace_impl(input):
    size = min(int(input.shape[0]), int(input.shape[1]))
    diagonal = jt.arange(size)
    return input[diagonal, diagonal].sum()


def trace(input):
    """Return the sum of a tensor's main matrix diagonal."""
    return _trace_impl(input)


def _diag_embed_impl(input, offset=0, dim1=-2, dim2=-1):
    size = int(input.shape[-1])
    return input.unsqueeze(-1) * jt.init.eye(size)


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """Embed the final dimension of a tensor along a matrix diagonal."""
    return _diag_embed_impl(input, offset=offset, dim1=dim1, dim2=dim2)


def _diagflat_impl(input, offset=0):
    return _diag_embed_impl(input.reshape((-1,)), offset=offset)


def diagflat(input, offset=0):
    """Flatten an input and embed it along a matrix diagonal."""
    return _diagflat_impl(input, offset=offset)


for _matrix_name in ("trace", "diag_embed", "diagflat"):
    register_fidelity(
        "torch." + _matrix_name,
        globals()[_matrix_name],
        Fidelity.APPROXIMATE,
        _MATRIX_FIDELITY_DETAIL,
    )
del _matrix_name


_CLOSE_FIDELITY_DETAIL = (
    "matches Torch finite-value comparisons and equal_nan on CPU-backed "
    "tensors but omits device, layout, and out keyword semantics"
)


def _isclose_impl(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    out = jt.abs(a - b) <= (atol + rtol * jt.abs(b))
    if equal_nan:
        out = out | (jt.isnan(a) & jt.isnan(b))
    return out


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    """Return an elementwise tensor indicating whether values are close."""
    return _isclose_impl(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs)


def _allclose_impl(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    return bool(_isclose_impl(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs
    ).all().item())


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **kwargs):
    """Return a Python bool indicating whether all values are close."""
    return _allclose_impl(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs)


for _close_name in ("isclose", "allclose"):
    register_fidelity(
        "torch." + _close_name,
        globals()[_close_name],
        Fidelity.APPROXIMATE,
        _CLOSE_FIDELITY_DETAIL,
    )
del _close_name


_PAIRWISE_SEARCH_FIDELITY_DETAIL = (
    "matches Torch values for supported tensor inputs but omits compute-mode, "
    "device, layout, and out keyword semantics"
)


def _cdist_impl(x1, x2, p=2.0, compute_mode=None, **kwargs):
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    if p == 2:
        return jt.sqrt((diff * diff).sum(-1))
    if p == 1:
        return jt.abs(diff).sum(-1)
    return (jt.abs(diff) ** p).sum(-1) ** (1.0 / p)


def cdist(x1, x2, p=2.0, compute_mode=None, **kwargs):
    """Return pairwise distances between the rows of two tensors."""
    return _cdist_impl(
        x1, x2, p=p, compute_mode=compute_mode, **kwargs)


def _bucketize_impl(
        input, boundaries, out_int32=False, right=False, **kwargs):
    flattened = boundaries.reshape((-1,))
    comparison = ((input.unsqueeze(-1) >= flattened)
                  if right else (input.unsqueeze(-1) > flattened))
    result = comparison.int32().sum(-1)
    return result if out_int32 else result.int64()


def bucketize(input, boundaries, out_int32=False, right=False, **kwargs):
    """Return insertion indices for values in sorted boundaries."""
    return _bucketize_impl(
        input, boundaries, out_int32=out_int32, right=right, **kwargs)


for _pairwise_search_name in ("cdist", "bucketize"):
    register_fidelity(
        "torch." + _pairwise_search_name,
        globals()[_pairwise_search_name],
        Fidelity.APPROXIMATE,
        _PAIRWISE_SEARCH_FIDELITY_DETAIL,
    )
del _pairwise_search_name


_NAN_REDUCTION_FIDELITY_DETAIL = (
    "matches Torch NaN-ignoring reductions and NaN counts for supported real "
    "tensor inputs but omits device, layout, and out keyword semantics"
)


def _nansum_impl(input, dim=None, keepdim=False, **kwargs):
    values = jt.nan_to_num(input, nan=0.0)
    return (values.sum() if dim is None
            else values.sum(dim, keepdims=keepdim))


def nansum(input, dim=None, keepdim=False, **kwargs):
    """Sum tensor values while treating NaNs as zero."""
    return _nansum_impl(input, dim=dim, keepdim=keepdim, **kwargs)


def _nanmean_impl(input, dim=None, keepdim=False, **kwargs):
    count = 1.0 - jt.isnan(input).float32()
    values = jt.nan_to_num(input, nan=0.0)
    if dim is None:
        return values.sum() / count.sum()
    return (values.sum(dim, keepdims=keepdim)
            / count.sum(dim, keepdims=keepdim))


def nanmean(input, dim=None, keepdim=False, **kwargs):
    """Mean tensor values while ignoring NaNs in the denominator."""
    return _nanmean_impl(input, dim=dim, keepdim=keepdim, **kwargs)


for _nan_reduction_name in ("nansum", "nanmean"):
    register_fidelity(
        "torch." + _nan_reduction_name,
        globals()[_nan_reduction_name],
        Fidelity.APPROXIMATE,
        _NAN_REDUCTION_FIDELITY_DETAIL,
    )
del _nan_reduction_name


_QUANTILE_FIDELITY_DETAIL = (
    "uses a NumPy CPU fallback for supported real tensors; dtype is returned "
    "as float32 and device, layout, interpolation, and out semantics are not "
    "implemented"
)


def _quantile_impl(input, q, dim=None, keepdim=False,
                   interpolation="linear", **kwargs):
    values = input.numpy()
    quantile = q.numpy() if isinstance(q, jt.Var) else q
    result = np.quantile(
        values, quantile, axis=dim, keepdims=keepdim)
    return jt.array(result.astype("float32"))


def quantile(input, q, dim=None, keepdim=False,
             interpolation="linear", **kwargs):
    """Return a NumPy-backed quantile result for CPU-compatible tensors."""
    return _quantile_impl(
        input, q, dim=dim, keepdim=keepdim,
        interpolation=interpolation, **kwargs)


register_fidelity(
    "torch.quantile",
    quantile,
    Fidelity.APPROXIMATE,
    _QUANTILE_FIDELITY_DETAIL,
)


_NANQUANTILE_FIDELITY_DETAIL = (
    "uses a NumPy CPU fallback for supported real tensors; dtype is returned "
    "as float32 and device, layout, interpolation, and out semantics are not "
    "implemented"
)


def _nanquantile_impl(input, q, dim=None, keepdim=False,
                      interpolation="linear", **kwargs):
    values = input.numpy()
    quantile = q.numpy() if isinstance(q, jt.Var) else q
    result = np.nanquantile(
        values, quantile, axis=dim, keepdims=keepdim)
    return jt.array(result.astype("float32"))


def nanquantile(input, q, dim=None, keepdim=False,
                interpolation="linear", **kwargs):
    """Return a NumPy-backed NaN-ignoring quantile for CPU tensors."""
    return _nanquantile_impl(
        input, q, dim=dim, keepdim=keepdim,
        interpolation=interpolation, **kwargs)


register_fidelity(
    "torch.nanquantile",
    nanquantile,
    Fidelity.APPROXIMATE,
    _NANQUANTILE_FIDELITY_DETAIL,
)


_STD_MEAN_FIDELITY_DETAIL = (
    "matches the current Jittor mean/std values for supported real tensors; "
    "correction is ignored and dim std does not preserve keepdim, while "
    "device, layout, and out semantics are omitted"
)


def _std_mean_impl(input, dim=None, unbiased=True, keepdim=False,
                   correction=None, **kwargs):
    mean = (input.mean() if dim is None
            else input.mean(dim, keepdims=keepdim))
    std = input.std() if dim is None else input.std(dim)
    return std, mean


def std_mean(input, dim=None, unbiased=True, keepdim=False,
             correction=None, **kwargs):
    """Return standard deviation and mean using current Jittor semantics."""
    return _std_mean_impl(
        input, dim=dim, unbiased=unbiased, keepdim=keepdim,
        correction=correction, **kwargs)


def _var_mean_impl(input, dim=None, unbiased=True, keepdim=False,
                   correction=None, **kwargs):
    standard_deviation, mean = _std_mean_impl(
        input, dim=dim, unbiased=unbiased, keepdim=keepdim,
        correction=correction, **kwargs)
    return standard_deviation * standard_deviation, mean


def var_mean(input, dim=None, unbiased=True, keepdim=False,
             correction=None, **kwargs):
    """Return variance and mean using current Jittor semantics."""
    return _var_mean_impl(
        input, dim=dim, unbiased=unbiased, keepdim=keepdim,
        correction=correction, **kwargs)


for _std_mean_name in ("std_mean", "var_mean"):
    register_fidelity(
        "torch." + _std_mean_name,
        globals()[_std_mean_name],
        Fidelity.APPROXIMATE,
        _STD_MEAN_FIDELITY_DETAIL,
    )
del _std_mean_name


_AminMax = _namedtuple("aminmax", ["min", "max"])
_AMINMAX_FIDELITY_DETAIL = (
    "matches Torch min/max values for supported real tensor inputs but omits "
    "device, layout, and out keyword semantics"
)


def _aminmax_impl(input, dim=None, keepdim=False):
    if dim is None:
        return _AminMax(input.min(), input.max())
    return _AminMax(
        input.min(dim, keepdims=keepdim),
        input.max(dim, keepdims=keepdim),
    )


def aminmax(input, dim=None, keepdim=False):
    """Return named minimum and maximum reductions."""
    return _aminmax_impl(input, dim=dim, keepdim=keepdim)


register_fidelity(
    "torch.aminmax",
    aminmax,
    Fidelity.APPROXIMATE,
    _AMINMAX_FIDELITY_DETAIL,
)


_PDIST_FIDELITY_DETAIL = (
    "matches Torch pairwise distances for supported real tensor inputs but "
    "omits device, layout, and out keyword semantics"
)


def _pdist_impl(input, p=2.0):
    size = int(input.shape[0])
    differences = input.unsqueeze(1) - input.unsqueeze(0)
    distances = ((jt.abs(differences) ** p).sum(-1)) ** (1.0 / p)
    rows = [i for i in range(size) for _ in range(i + 1, size)]
    cols = [j for i in range(size) for j in range(i + 1, size)]
    return distances[jt.array(rows), jt.array(cols)]


def pdist(input, p=2.0):
    """Return pairwise p-norm distances between rows of a tensor."""
    return _pdist_impl(input, p=p)


register_fidelity(
    "torch.pdist",
    pdist,
    Fidelity.APPROXIMATE,
    _PDIST_FIDELITY_DETAIL,
)


_LOGCUMSUMEXP_FIDELITY_DETAIL = (
    "matches Torch cumulative log-sum-exp values for supported real tensors "
    "but omits device, layout, and out keyword semantics"
)


def _logcumsumexp_impl(input, dim):
    maximum = input.max(dim, keepdims=True)
    return maximum + jt.log(jt.cumsum(jt.exp(input - maximum), dim))


def logcumsumexp(input, dim):
    """Return cumulative log-sum-exp values along ``dim``."""
    return _logcumsumexp_impl(input, dim)


register_fidelity(
    "torch.logcumsumexp",
    logcumsumexp,
    Fidelity.APPROXIMATE,
    _LOGCUMSUMEXP_FIDELITY_DETAIL,
)


_MV_FIDELITY_DETAIL = (
    "matches Torch matrix-vector values, shape checks, and out identity for "
    "supported real tensors but omits device, layout, and dtype keyword semantics"
)


def _mv_impl(input, vec, out=None):
    if input.ndim != 2 or vec.ndim != 1:
        raise RuntimeError(
            "mv: expected a 2-D matrix and a 1-D vector, got "
            f"{input.ndim}-D and {vec.ndim}-D tensors")
    if input.shape[1] != vec.shape[0]:
        raise RuntimeError(
            "mv: size mismatch, matrix has %s columns but vector has %s elements"
            % (input.shape[1], vec.shape[0]))
    result = jt.matmul(input, vec)
    if out is not None:
        out.assign(result)
        return out
    return result


def mv(input, vec, out=None):
    """Multiply a matrix by a vector, optionally writing into ``out``."""
    return _mv_impl(input, vec, out=out)


register_fidelity(
    "torch.mv",
    mv,
    Fidelity.APPROXIMATE,
    _MV_FIDELITY_DETAIL,
)


_ADDMM_FIDELITY_DETAIL = (
    "matches Torch alpha/beta matrix addition for supported real tensors but "
    "omits device, layout, dtype, and out keyword semantics"
)


def _addmm_impl(input, mat1, mat2, *, beta=1, alpha=1):
    result = jt.matmul(mat1, mat2)
    if alpha != 1:
        result = result * alpha
    if beta == 0:
        return result
    return beta * input + result


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    """Compute ``beta * input + alpha * (mat1 @ mat2)``."""
    return _addmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


register_fidelity(
    "torch.addmm",
    addmm,
    Fidelity.APPROXIMATE,
    _ADDMM_FIDELITY_DETAIL,
)


_MM_FIDELITY_DETAIL = (
    "matches Torch 2-D matrix multiplication values for supported real tensors "
    "but omits out, device, layout, and dtype keyword semantics"
)


def _mm_impl(input, mat2, out=None):
    # Keep the existing compatibility boundary: ``out`` is accepted for API
    # shape compatibility but is not populated by this approximate fallback.
    return jt.matmul(input, mat2)


def mm(input, mat2, out=None):
    """Multiply two 2-D tensors using Jittor's matrix multiplication."""
    return _mm_impl(input, mat2, out=out)


register_fidelity(
    "torch.mm",
    mm,
    Fidelity.APPROXIMATE,
    _MM_FIDELITY_DETAIL,
)


_TRAPZ_FIDELITY_DETAIL = (
    "matches Torch composite trapezoidal integration values and out identity "
    "for supported real tensors but omits device, layout, and dtype keyword semantics"
)


def trapz(y, x=None, dx=1, dim=-1, *, out=None):
    """Integrate along a tensor dimension with the trapezoidal rule."""
    return _trapz(y, x=x, dx=dx, dim=dim, out=out)


def trapezoid(y, x=None, dx=1, dim=-1, *, out=None):
    """Alias of :func:`trapz` using Torch's newer spelling."""
    return _trapz(y, x=x, dx=dx, dim=dim, out=out)


register_fidelity(
    "torch.trapz",
    trapz,
    Fidelity.APPROXIMATE,
    _TRAPZ_FIDELITY_DETAIL,
)
register_fidelity(
    "torch.trapezoid",
    trapezoid,
    Fidelity.APPROXIMATE,
    _TRAPZ_FIDELITY_DETAIL,
)


_MASKED_SELECT_FIDELITY_DETAIL = (
    "matches Torch boolean selection values and flattened shape for supported "
    "real tensors but omits out, device, layout, and dtype keyword semantics"
)


def _masked_select_impl(input, mask, out=None):
    # Keep the compatibility boundary: ``out`` is accepted for API shape
    # compatibility but is not populated by this approximate fallback.
    return input[mask]


def masked_select(input, mask, out=None):
    """Return the flattened elements selected by a boolean mask."""
    return _masked_select_impl(input, mask, out=out)


register_fidelity(
    "torch.masked_select",
    masked_select,
    Fidelity.APPROXIMATE,
    _MASKED_SELECT_FIDELITY_DETAIL,
)


_NARROW_FIDELITY_DETAIL = (
    "matches Torch contiguous slice values and shape for supported real tensors "
    "but omits device, layout, and dtype keyword semantics"
)


def _narrow_impl(input, dim, start, length):
    return input.narrow(dim, start, length)


def narrow(input, dim, start, length):
    """Return a length-sized slice along ``dim`` starting at ``start``."""
    return _narrow_impl(input, dim, start, length)


register_fidelity(
    "torch.narrow",
    narrow,
    Fidelity.APPROXIMATE,
    _NARROW_FIDELITY_DETAIL,
)


_TILE_FIDELITY_DETAIL = (
    "matches Torch repetition values and shape for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)


def _tile_impl(input, *dims):
    return input.tile(*dims)


def tile(input, *dims):
    """Repeat tensor dimensions using Torch-compatible tile semantics."""
    return _tile_impl(input, *dims)


register_fidelity(
    "torch.tile",
    tile,
    Fidelity.APPROXIMATE,
    _TILE_FIDELITY_DETAIL,
)


_DIFF_FIDELITY_DETAIL = (
    "matches Torch finite differences and prepend/append concatenation for "
    "supported real tensors but omits device, layout, and dtype keyword semantics"
)


def _diff_impl(x, n=1, dim=-1, prepend=None, append=None):
    return _diff(x, n=n, dim=dim, prepend=prepend, append=append)


def diff(x, n=1, dim=-1, prepend=None, append=None):
    """Compute consecutive differences along a tensor dimension."""
    return _diff_impl(x, n=n, dim=dim, prepend=prepend, append=append)


register_fidelity(
    "torch.diff",
    diff,
    Fidelity.APPROXIMATE,
    _DIFF_FIDELITY_DETAIL,
)


_SQUARE_FIDELITY_DETAIL = (
    "matches Torch elementwise square values for supported real tensors but "
    "omits device, layout, and dtype keyword semantics"
)


def square(x):
    """Return the elementwise square of a tensor."""
    return x * x


register_fidelity(
    "torch.square",
    square,
    Fidelity.APPROXIMATE,
    _SQUARE_FIDELITY_DETAIL,
)


_SPLIT_WITH_SIZES_FIDELITY_DETAIL = (
    "matches Torch split sizes and values for supported real tensors but omits "
    "device, layout, and dtype keyword semantics"
)


def split_with_sizes(input, split_sizes, dim=0):
    """Split a tensor into chunks with the requested sizes."""
    return input.split(split_sizes, dim)


register_fidelity(
    "torch.split_with_sizes",
    split_with_sizes,
    Fidelity.APPROXIMATE,
    _SPLIT_WITH_SIZES_FIDELITY_DETAIL,
)


def equal(a, b):
    """Return a Python bool for Torch's same-shape, elementwise equality."""
    try:
        if isinstance(a, _NestedTensor) or isinstance(b, _NestedTensor):
            return bool(a.equal(b)) if isinstance(a, _NestedTensor) else False
        if not isinstance(a, jt.Var) or not isinstance(b, jt.Var):
            return bool(a == b)
        if tuple(a.shape) != tuple(b.shape):
            return False
        if a.numel() == 0:
            return True
        return bool((a == b).all().item())
    except EXPECTED as exc:
        swallowed("torch/installers/numerical.py equal", exc)
        return False


register_fidelity(
    "torch.equal",
    equal,
    Fidelity.APPROXIMATE,
    "matches Torch Python-bool shape/value equality for CPU tensors; device, "
    "layout, and named-dimension semantics are not implemented",
)


def tensor_split(input, indices_or_sections, dim=0):
    """Split a tensor into uneven chunks using view slices."""
    d = dim % input.ndim
    length = int(input.shape[d])

    def _slice(start, stop):
        index = [slice(None)] * input.ndim
        index[d] = slice(start, stop)
        return input[tuple(index)]

    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        base, rem = divmod(length, n)
        sizes = [base + 1] * rem + [base] * (n - rem)
        result, start = [], 0
        for size in sizes:
            result.append(_slice(start, start + size))
            start += size
        return result
    points, result, previous = list(indices_or_sections), [], 0
    for point in points + [length]:
        result.append(_slice(previous, point))
        previous = point
    return result


register_fidelity(
    "torch.tensor_split",
    tensor_split,
    Fidelity.APPROXIMATE,
    "matches Torch split shapes and values for CPU tensors; device, layout, "
    "named-dimension, and out semantics are not implemented",
)


def take(input, index):
    """Select flattened elements using Torch's ``take`` semantics."""
    return input.reshape((-1,))[index]


register_fidelity(
    "torch.take",
    take,
    Fidelity.APPROXIMATE,
    "matches Torch flattened indexing values for CPU tensors; device, layout, "
    "dtype, and out semantics are not implemented",
)


def index_copy(input, dim, index, source):
    """Copy rows/slices into a clone along ``dim`` (Torch non-inplace form)."""
    result = input.clone()
    d = dim % result.ndim
    idx = index if isinstance(index, jt.Var) else jt.array(index)
    if d == 0:
        result[idx] = source
    else:
        slices = [slice(None)] * result.ndim
        slices[d] = idx
        result[tuple(slices)] = source
    return result


register_fidelity(
    "torch.index_copy",
    index_copy,
    Fidelity.APPROXIMATE,
    "matches Torch non-inplace index-copy values for CPU tensors; device, "
    "layout, dtype, and out semantics are not implemented",
)


def index_copy_(input, dim, index, source):
    """Copy rows/slices into ``input`` in place and return it."""
    d = dim % input.ndim
    idx = index if isinstance(index, jt.Var) else jt.array(index)
    if d == 0:
        input[idx] = source
    else:
        slices = [slice(None)] * input.ndim
        slices[d] = idx
        input[tuple(slices)] = source
    return input


register_fidelity(
    "torch.index_copy_",
    index_copy_,
    Fidelity.APPROXIMATE,
    "matches Torch in-place indexed assignment for CPU tensors; device, "
    "layout, dtype, and out semantics are not implemented",
)


def index_put(input, indices, values, accumulate=False):
    """Return a clone with indexed values assigned using Torch semantics."""
    result = input.clone()
    idx = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
    if not accumulate:
        result[idx if len(idx) > 1 else idx[0]] = values
        return result
    vals = values if isinstance(values, jt.Var) else jt.array(values)
    if len(idx) == 1:
        index = idx[0] if isinstance(idx[0], jt.Var) else jt.array(idx[0])
        result.assign(result.index_add(0, index.int64().reshape((-1,)), vals))
        return result
    raise NotImplementedError(
        "index_put(accumulate=True) with a partial multi-dim index")


register_fidelity(
    "torch.index_put",
    index_put,
    Fidelity.APPROXIMATE,
    "matches Torch non-inplace indexed assignment for CPU tensors; duplicate "
    "multi-dimensional accumulation, device, layout, and out semantics are "
    "not implemented",
)


def index_put_(input, indices, values, accumulate=False):
    """Assign indexed values in place using Torch's duplicate-safe path."""
    idx = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
    if not accumulate:
        input[idx if len(idx) > 1 else idx[0]] = values
        return input
    vals = values if isinstance(values, jt.Var) else jt.array(values)
    if len(idx) == input.ndim:
        shape = input.shape
        strides = [1] * input.ndim
        for k in range(input.ndim - 2, -1, -1):
            strides[k] = strides[k + 1] * int(shape[k + 1])
        linear = None
        for k, ind in enumerate(idx):
            term = (ind if isinstance(ind, jt.Var) else jt.array(ind)).int64().reshape((-1,)) * strides[k]
            linear = term if linear is None else linear + term
        flat_values = vals.reshape((-1,))
        if int(flat_values.shape[0]) == 1 and int(linear.shape[0]) > 1:
            flat_values = flat_values.broadcast(linear.shape)
        input.assign(input.reshape((-1,)).index_add(0, linear, flat_values).reshape(shape))
        return input
    if len(idx) == 1:
        index = idx[0] if isinstance(idx[0], jt.Var) else jt.array(idx[0])
        input.assign(input.index_add(0, index.int64().reshape((-1,)), vals))
        return input
    raise NotImplementedError("index_put_(accumulate=True) with a partial multi-dim index")


register_fidelity(
    "torch.index_put_",
    index_put_,
    Fidelity.APPROXIMATE,
    "matches Torch in-place indexed assignment and duplicate accumulation for "
    "CPU tensors; device, layout, dtype, and out semantics are not implemented",
)


def kron(a, b):
    """Compute the Kronecker product through broadcasted Jittor views."""
    nd = max(a.ndim, b.ndim)
    a2 = a.reshape((1,) * (nd - a.ndim) + tuple(a.shape))
    b2 = b.reshape((1,) * (nd - b.ndim) + tuple(b.shape))
    aex, bex, fin = [], [], []
    for i in range(nd):
        aex += [int(a2.shape[i]), 1]
        bex += [1, int(b2.shape[i])]
        fin.append(int(a2.shape[i]) * int(b2.shape[i]))
    return (a2.reshape(aex) * b2.reshape(bex)).reshape(fin)


register_fidelity(
    "torch.kron",
    kron,
    Fidelity.APPROXIMATE,
    "matches Torch Kronecker shape and values for CPU tensors; device, "
    "layout, dtype, and out semantics are not implemented",
)


def logsumexp(input, dim, keepdim=False):
    """Compute a numerically stable log-sum-exp reduction."""
    m = input.max(dim, keepdims=True)
    out = m + jt.log(jt.exp(input - m).sum(dim, keepdims=True))
    if keepdim:
        return out
    dims = [dim] if isinstance(dim, int) else list(dim)
    nd = input.ndim
    dims = [d % nd for d in dims]
    target = [s for i, s in enumerate(input.shape) if i not in dims]
    return out.reshape(target) if target else out.reshape(-1)


register_fidelity(
    "torch.logsumexp",
    logsumexp,
    Fidelity.APPROXIMATE,
    "matches Torch reduction values and keepdim shape for CPU tensors; device, "
    "dtype, named-dimension, and out semantics are not implemented",
)


def hann_window(window_length, periodic=True, *, dtype=None, device=None,
                requires_grad=False, **kwargs):
    """Create a Hann window through the CPU NumPy signal owner."""
    length = int(window_length)
    if length <= 1:
        return jt.from_numpy(np.ones(max(length, 0), np.float32))
    denominator = length if periodic else (length - 1)
    index = np.arange(length, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / denominator)
    return jt.from_numpy(window.astype(np.float32))


register_fidelity(
    "torch.hann_window",
    hann_window,
    Fidelity.APPROXIMATE,
    "matches Torch periodic and symmetric CPU window values; device, dtype, "
    "layout, and requires_grad semantics are not implemented",
)


def stft(input, n_fft, hop_length=None, win_length=None, window=None,
         center=True, pad_mode="reflect", normalized=False, onesided=True,
         return_complex=True, **kwargs):
    """Compute a CPU NumPy short-time Fourier transform."""
    samples = np.asarray(input.numpy() if hasattr(input, "numpy") else input)
    n_fft = int(n_fft)
    hop = int(hop_length) if hop_length else n_fft // 4
    win_len = int(win_length) if win_length else n_fft
    if window is None:
        win = np.ones(win_len, np.float64)
    else:
        win = np.asarray(window.numpy() if hasattr(window, "numpy") else window).astype(np.float64)
    if win.shape[0] < n_fft:
        left = (n_fft - win.shape[0]) // 2
        win = np.pad(win, (left, n_fft - win.shape[0] - left))
    squeeze = samples.ndim == 1
    if squeeze:
        samples = samples[None, :]
    if center:
        samples = np.pad(samples, ((0, 0), (n_fft // 2, n_fft // 2)), mode=pad_mode)
    batch, length = samples.shape
    frames = 1 + (length - n_fft) // hop
    transform = np.fft.rfft if onesided else np.fft.fft
    spectra = []
    for row in range(batch):
        windowed = np.stack(
            [samples[row, i * hop:i * hop + n_fft] * win for i in range(frames)],
            axis=-1)
        spectrum = transform(windowed, n=n_fft, axis=0)
        if normalized:
            spectrum = spectrum / np.sqrt(n_fft)
        spectra.append(spectrum)
    out = np.stack(spectra, axis=0)
    if squeeze:
        out = out[0]
    return jt.from_numpy(np.ascontiguousarray(out.astype(np.complex64)))


register_fidelity(
    "torch.stft",
    stft,
    Fidelity.APPROXIMATE,
    "matches Torch CPU NumPy STFT values for supported real waveforms; "
    "gradient, device, window dtype, and return_complex=False semantics are "
    "not implemented",
)


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    import collections as _collections
    def _alias(name, fn):
        if not hasattr(g, name):
            setattr(g, name, fn)
    # complex-dtype API (#3): jittor represents complex via nn.ComplexNumber (real/imag
    # pair); wire the torch entry points onto it. torch.complex(re,im), view_as_complex
    # (last dim of 2 -> complex), view_as_real (complex -> last dim of 2), polar, real/
    # imag/conj/is_complex. The arithmetic (* / + matmul exp conj) is on ComplexNumber.
    _CN = jt.nn.ComplexNumber
    # A complex value is either the legacy ComplexNumber (still produced by torch.complex and
    # consumed by torch.fft.* -- migrated in P3) OR the native complex64 dtype (Phase 6). The
    # accessors below handle both; Var.real/imag/angle are patched in jittor.nn. We force-set
    # (not _alias) the accessors because _alias skips names that already exist as native ops --
    # that is why torch.conj(ComplexNumber) used to fall through to the native conj op and crash.
    _alias("complex", complex)  # native complex64
    _alias("view_as_complex", view_as_complex)   # -> native complex64
    _alias("view_as_real", view_as_real)         # polymorphic
    g.is_complex = is_complex
    g.real = real
    g.imag = imag
    _alias("polar", polar)                                        # -> native complex64
    g.conj = conj
    g.angle = angle
    # torch.abs of a complex tensor is its magnitude; jittor's abs only takes real Vars.
    g.abs = abs
    Var.abs = lambda self: _native_abs(self)

    # ``jittor.fft`` is the native owner. Torch mode publishes that same module
    # object under its historical namespace instead of carrying a duplicate DFT.
    from jittor import fft as _fft_ns
    g.fft = _fft_ns
    _modules["torch.fft"] = _fft_ns
    # torch.softmax / log_softmax / relu top-level function forms (convbert calls
    # torch.softmax(x, dim=...)). jittor exposes these via nn, not the top level.
    _alias("softmax", softmax)
    _alias("log_softmax", log_softmax)
    _alias("relu", relu)
    # elementwise / functional top-level forms missing from jittor's top level
    _alias("log1p", log1p)
    _alias("reciprocal", reciprocal)
    _alias("lerp", lerp)
    _alias("isclose", isclose)
    _alias("allclose", allclose)
    _alias("cosine_similarity", cosine_similarity)
    _alias("pairwise_distance", pairwise_distance)
    # torch.take_along_dim(input, indices, dim): like gather, but torch BROADCASTS
    # indices against input on every dim except `dim` first. transformers' beam search
    # _gather_beams passes indices of shape (batch, k, 1) to gather full sequences of
    # shape (batch, beams, seq_len) along dim=1 -> expects (batch, k, seq_len). A plain
    # jt.gather returns the index's shape (batch, k, 1), collapsing seq_len -> beam
    # search crashed on the next `seq[:, :, cur_len] = ...` setitem. Broadcast first.
    _alias("take_along_dim", take_along_dim)
    _orig_all = getattr(g, "all", None)
    _orig_any = getattr(g, "any", None)
    if callable(_orig_all):
        g.all = all
    if callable(_orig_any):
        g.any = any
    _alias("movedim", movedim)
    _alias("moveaxis", moveaxis)
    # Var.movedim/moveaxis (the functions exist but weren't bound as methods), plus
    # index_put_/index_put (scatter-style assignment), tensor_split (uneven split), take.
    Var.movedim = lambda self, source, destination: _movedim_impl(self, source, destination)
    Var.moveaxis = lambda self, source, destination: _movedim_impl(self, source, destination)
    Var.index_put_ = index_put_
    Var.index_put = lambda self, indices, values, accumulate=False: index_put(
        self, indices, values, accumulate)
    # index_copy_(dim, index, source): self[..,index[i],..] = source[i,..] along dim
    # (overwrite, NOT accumulate -- cf. index_add).
    Var.index_copy_ = index_copy_
    Var.index_copy = lambda self, dim, index, source: index_copy(
        self, dim, index, source)
    g.index_copy = index_copy
    g.index_copy_ = index_copy_
    g.index_put = index_put
    g.index_put_ = index_put_
    Var.tensor_split = lambda self, indices_or_sections, dim=0: tensor_split(
        self, indices_or_sections, dim)
    g.tensor_split = tensor_split
    Var.take = lambda self, index: take(self, index)
    g.take = take
    _alias("eye", eye)
    register_fidelity(
        "torch.eye", eye, Fidelity.APPROXIMATE,
        "Values and dtype are supported; layout, device, out, and pin_memory "
        "arguments are not implemented.")
    # torch.narrow(input, dim, start, length) / torch.tile(input, dims) --
    # function forms mirroring the Var methods (added in _install_tensor_methods).
    _alias("narrow", narrow)
    _alias("tile", tile)
    # torch.equal returns a Python bool (True iff same shape & all elements
    # equal). jittor's native `equal` is elementwise, so force-override.
    g.equal = equal
    Var.equal = lambda self, other: equal(self, other)
    _alias("diff", diff)
    _alias("trapz", trapz)
    _alias("trapezoid", trapezoid)
    g.repeat_interleave = repeat_interleave
    _alias("autocast", autocast)
    # Real loop-based torch.vmap. The old no-op stub (`lambda fn,*a,**k: fn`)
    # ignored in_dims/out_dims, so transformers' vmap-based causal-mask builder
    # (taken when a model passes and_mask/or_mask -- e.g. falcon) collapsed to a
    # single direct call and produced a wrong all-True (seq,) mask instead of the
    # (b,1,q,kv) causal triangle -> bidirectional attention -> ~79% forward error.
    # Map over in_dims and stack along out_dims. jittor has no 0-d tensors, so a
    # scalar leaf is (1,) where torch has (); collapse that spurious trailing
    # singleton so the stacked rank matches torch.vmap.
    def _vectorized_getitem_vmap(func, specs, args):
        # Transformers builds attention masks under TransformGetItemToIndex
        # using nested pointwise vmaps. Materialize their Cartesian batch axes
        # through broadcasting instead of creating one graph per scalar pair.
        if len(specs) < 2 or _py_any(out_dims != 0 for _, out_dims in specs):
            return None
        mapped_by_arg = [[] for _ in args]
        level_sizes = []
        for level, (level_dims, _) in enumerate(specs):
            dims = ((level_dims,) * len(args)
                    if isinstance(level_dims, int) or level_dims is None
                    else tuple(level_dims))
            if len(dims) != len(args):
                return None
            mapped_sizes = []
            for arg_index, dim in enumerate(dims):
                if dim is not None:
                    if dim != 0 or not isinstance(args[arg_index], jt.Var):
                        return None
                    mapped_by_arg[arg_index].append(level)
                    mapped_sizes.append(int(args[arg_index].shape[dim]))
            if not mapped_sizes or _py_any(size != mapped_sizes[0]
                                       for size in mapped_sizes[1:]):
                return None
            level_sizes.append(mapped_sizes[0])
        if _py_any(len(levels) > 1 for levels in mapped_by_arg):
            return None

        level_count = len(specs)
        expanded = []
        for arg, mapped_levels in zip(args, mapped_by_arg):
            if not mapped_levels:
                expanded.append(arg)
                continue
            output_axis = level_count - 1 - mapped_levels[0]
            shape = ([1] * output_axis + [int(arg.shape[0])] +
                     [1] * (level_count - output_axis - 1) +
                     [int(size) for size in arg.shape[1:]])
            expanded.append(arg.reshape(shape))
        result = func(*expanded)
        if (
            not isinstance(result, jt.Var)
            or str(result.dtype) != "bool"
            or result.ndim > level_count
        ):
            return None
        if result.ndim < level_count:
            result = result.reshape([1] * (level_count - result.ndim) +
                                    [int(size) for size in result.shape])
        target_shape = list(reversed(level_sizes)) + [
            int(size) for size in result.shape[level_count:]
        ]
        return result.broadcast(target_shape)

    def _vmap(func, in_dims=0, out_dims=0, *_a, **_k):
        base_func = getattr(func, "_jittor_vmap_base", func)
        specs = getattr(func, "_jittor_vmap_specs", ()) + ((in_dims, out_dims),)

        def wrapped(*args):
            if getitem_transform_active(g):
                vectorized = _vectorized_getitem_vmap(base_func, specs, args)
                if vectorized is not None:
                    return vectorized
            ids = (in_dims,) * len(args) if (isinstance(in_dims, int) or in_dims is None) else tuple(in_dims)
            size = None
            for a, d in zip(args, ids):
                if d is not None:
                    size = int(a.shape[d]); break
            if size is None:
                return func(*args)
            outs = []
            for i in range(size):
                sub = []
                for a, d in zip(args, ids):
                    if d is None:
                        sub.append(a)
                    else:
                        idx = [slice(None)] * a.ndim; idx[d] = i
                        sub.append(a[tuple(idx)])
                r = func(*sub)
                if not isinstance(r, jt.Var):
                    r = jt.array(r)
                outs.append(r)
            if _py_all(o.ndim >= 1 and o.shape[-1] == 1 for o in outs) and _py_all(o.ndim == outs[0].ndim for o in outs):
                outs = [o.reshape(o.shape[:-1]) if o.ndim > 1 else o for o in outs]
            od = out_dims if isinstance(out_dims, int) else (out_dims[0] if out_dims else 0)
            return jt.stack(outs, dim=od)
        wrapped._jittor_vmap_base = base_func
        wrapped._jittor_vmap_specs = specs
        return wrapped
    global _vmap_runtime_impl
    _vmap_runtime_impl = _vmap
    g.vmap = vmap
    g.outer = outer
    g.isin = isin
    # Pairwise distances and sorted-boundary insertion indices.
    _alias("cdist", cdist)
    _alias("bucketize", bucketize)
    # trace / diag_embed / diagflat / kron / logcumsumexp / tensordot / pdist.
    _alias("trace", trace); Var.trace = _trace_impl
    _alias("diag_embed", diag_embed); Var.diag_embed = _diag_embed_impl
    _alias("diagflat", diagflat)
    g.kron = kron; Var.kron = kron
    _alias("logcumsumexp", logcumsumexp); Var.logcumsumexp = _logcumsumexp_impl
    g.tensordot = tensordot
    _alias("pdist", pdist); Var.pdist = _pdist_impl
    # shape ops: unflatten / swapaxes / swapdims / ravel + numpy-style stacking helpers.
    _alias("unflatten", unflatten); Var.unflatten = _unflatten_impl
    _alias("swapaxes", swapaxes); _alias("swapdims", swapdims)
    Var.swapaxes = _swapaxes_impl; Var.swapdims = _swapaxes_impl
    _alias("ravel", ravel); Var.ravel = _ravel_impl
    _alias("vstack", vstack)
    _alias("row_stack", row_stack)
    _alias("hstack", hstack)
    _alias("dstack", dstack)
    _alias("column_stack", column_stack)
    # element-wise ops: copysign / xlogy / heaviside / float_power / signbit.
    _alias("copysign", copysign); Var.copysign = _copysign_impl
    _alias("xlogy", xlogy); Var.xlogy = _xlogy_impl
    _alias("heaviside", heaviside); Var.heaviside = _heaviside_impl
    _alias("float_power", float_power); Var.float_power = _float_power_impl
    _alias("signbit", signbit); Var.signbit = _signbit_impl
    # reductions: logsumexp (attention/MoE/loss/beam), nansum/nanmean, std_mean/var_mean,
    # aminmax, quantile. NaN handling uses nan_to_num plus an explicit isnan mask.
    g.logsumexp = logsumexp; Var.logsumexp = logsumexp
    _alias("nansum", nansum); Var.nansum = _nansum_impl
    _alias("nanmean", nanmean); Var.nanmean = _nanmean_impl
    _alias("std_mean", std_mean)
    _alias("var_mean", var_mean)
    _alias("aminmax", aminmax); Var.aminmax = _aminmax_impl
    _alias("quantile", quantile)
    _alias("nanquantile", nanquantile)
    # Keep the Tensor methods on the same numerical owners as the top-level
    # functions.  The owners intentionally use the documented CPU NumPy
    # fallback, so method and function forms share the same fidelity limits.
    Var.quantile = lambda self, q, dim=None, keepdim=False, interpolation="linear", **kwargs: quantile(
        self, q, dim=dim, keepdim=keepdim, interpolation=interpolation, **kwargs)
    Var.nanquantile = lambda self, q, dim=None, keepdim=False, interpolation="linear", **kwargs: nanquantile(
        self, q, dim=dim, keepdim=keepdim, interpolation=interpolation, **kwargs)
    _alias("square", square)
    _alias("addmm", addmm)

    # ---- torch.* ops used by mmdetection (additive aliases) ----
    _alias("mm", mm)
    _alias("mv", mv)
    _alias("masked_select", masked_select)
    _alias("split_with_sizes", split_with_sizes)
    _alias("_shape_as_tensor", _shape_as_tensor)
    _alias("nan_to_num_", nan_to_num_)
    # torch.randint_like(input, low, high=None, *, dtype=...): jittor's native lacks
    # the dtype kwarg (DINO's denoising uses it). Force-override with torch semantics.
    g.randint_like = randint_like

    _alias("sparse_coo_tensor", sparse_coo_tensor)
    import jittor.sparse as _jt_sparse
    if not hasattr(_jt_sparse, "sum"):
        def _sparse_sum(x, dim=None):
            d = x._dense if isinstance(x, _SparseCOO) else x
            return _SparseCOO(d.sum(dim) if dim is not None else d.sum())
        _jt_sparse.sum = _sparse_sum

    # det/inverse on (batched) square matrices (mmrotate GWD/KLD/KFIoU Gaussian losses)
    def _vdet(self):
        import jittor.linalg as _la; return _la.det(self)
    def _vinv(self):
        import jittor.linalg as _la; return _la.inv(self)
    if not hasattr(Var, "det"):       Var.det = _vdet
    if not hasattr(Var, "inverse"):   Var.inverse = _vinv
    g.det = det
    g.inverse = inverse

    # ---- linalg (peft / lora init need svd_lowrank, svd) ----
    _alias("svd", svd)
    _alias("svd_lowrank", svd_lowrank)
    _alias("pca_lowrank", pca_lowrank)


def install_parity(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    def module(name):
        return registry.ensure(name)
    import jittor.linalg as linalg
    registry.publish("torch.linalg", linalg)
    g.linalg = linalg

    import jittor.sparse as sparse
    registry.publish("torch.sparse", sparse)
    g.sparse = sparse

    special = module("torch.special")
    for name in ("erf", "erfc", "exp", "expm1", "log1p", "sinc"):
        value = getattr(g, name, None)
        if value is not None:
            setattr(special, name, value)
    special.expit = getattr(g, "sigmoid")
    g.special = special


def install_signal(ctx):
    """Window and short-time Fourier transform.

    Whisper-style mel feature extraction (Qwen2-Audio/Omni and any other audio
    front-end) calls these two directly. NumPy carries the arithmetic: the inputs
    are single waveforms, so the transform is not on a throughput path, and an
    exact match with torch's definition matters more than speed here.
    """
    g = ctx.jittor_module
    if not hasattr(g, "hann_window"):
        g.hann_window = hann_window
    if not hasattr(g, "stft"):
        g.stft = stft
