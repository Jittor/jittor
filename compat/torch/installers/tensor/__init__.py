"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""
from ...fidelity import Fidelity, register_api_bindings
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import builtins as _builtins

import jittor as jt

from jittor import nn

import numbers

import numpy as np

from ...functional import _diff, _torch_norm_impl, _torch_where_select, _trapz

from ...grad import _GradDecoratorCtx

from ...nested import _NestedTensor, _rebuild_var_from_numpy, _TorchSize, _torch_prune_leaf_registry, _torch_register_leaf

from ...tensor_state import get_tensor_state

from ..factories import _install_random_and_linspace, _set_use_cuda, _wrap_constructors

from ..numerical import log_softmax as _numerical_log_softmax, masked_select as _numerical_masked_select, softmax as _numerical_softmax

from ...types import _DEVICE_CTX_STACK, _device_is_cpu, _device_is_cuda, _dtype_to_str, _make_cpu_resident, _make_cuda_resident, _mark_cpu_like, _var_has_cpu_residency_hint, _var_is_cpu_resident, device, dtype, _cuda_index_of, _move_to_cuda_index

from ...fidelity import Fidelity, register_fidelity

import collections as _collections

from ....diagnostics import EXPECTED, swallowed

from .... import fsdp_hooks as _fsdp_hooks

from .... import collectives as _collectives

from .reductions import corrcoef

register_fidelity(
    "torch.corrcoef", corrcoef, Fidelity.APPROXIMATE,
    "matches Torch correlation values for CPU tensors through NumPy; device, "
    "dtype, gradient, and keyword semantics are not implemented",
)

from .reductions import broadcast_shapes

register_fidelity(
    "torch.broadcast_shapes", broadcast_shapes, Fidelity.APPROXIMATE,
    "matches Torch broadcasted shape tuples through NumPy; symbolic, named "
    "dimensions, and device semantics are not implemented",
)

_NATIVE_AMAX = jt.amax

_NATIVE_AMIN = jt.amin

_NATIVE_COUNT_NONZERO = jt.count_nonzero

_REDUCTION_EXTRAS_FIDELITY_DETAIL = (
    "re-exports Jittor's native values-only reduction owner, which matches "
    "Torch values and keepdim shape for int/tuple dims on supported real "
    "tensors; device, layout, dtype, and out keyword semantics are not "
    "implemented"
)

from .reductions import amax

from .reductions import amin

from .reductions import count_nonzero

for _reduction_extra in (amax, amin, count_nonzero):
    # ``install_methods`` retrofits an axis->dim adapter onto the Var reduction
    # methods; the stable objects take ``axis`` themselves, so the adapter has to
    # leave them alone or the module-level object and the method stop being one.
    _reduction_extra._torch_accepts_axis = True
    register_fidelity(
        "torch." + _reduction_extra.__name__,
        _reduction_extra,
        Fidelity.APPROXIMATE,
        _REDUCTION_EXTRAS_FIDELITY_DETAIL,
    )

del _reduction_extra

_FLOAT32_MAX = 3.4028234663852886e38

_NAN_TO_NUM_FIDELITY_DETAIL = (
    "matches Torch NaN and +-inf replacement exactly for the default "
    "float32-max bounds; it is a clamp rather than an isinf ternary because "
    "the latter segfaults in JIT codegen over a tensor holding inf/nan, so a "
    "narrow custom posinf/neginf also clamps finite values past that bound, "
    "and device, layout, dtype, and out keyword semantics are not implemented"
)

_LOGADDEXP_FIDELITY_DETAIL = (
    "matches Torch log(exp(a) + exp(b)) through the max-shifted stable form, "
    "so inputs that would overflow exp() individually still resolve; device, "
    "layout, dtype, and out keyword semantics are not implemented"
)

from .arithmetic import nan_to_num

from .arithmetic import logaddexp

register_fidelity(
    "torch.nan_to_num", nan_to_num, Fidelity.APPROXIMATE,
    _NAN_TO_NUM_FIDELITY_DETAIL)

register_fidelity(
    "torch.logaddexp", logaddexp, Fidelity.APPROXIMATE,
    _LOGADDEXP_FIDELITY_DETAIL)

_NATIVE_ARGSORT = jt.argsort

_NATIVE_GATHER = jt.gather

_NATIVE_MEDIAN = jt.median

_TopK = _collections.namedtuple("topk", ["values", "indices"])

_Sort = _collections.namedtuple("sort", ["values", "indices"])

_Median = _collections.namedtuple("median", ["values", "indices"])

_ORDERING_FIDELITY_DETAIL = (
    "matches Torch sorted values, the lower-median position, and the int64 "
    "index dtype for supported real tensors, and CPU and CUDA agree exactly on "
    "the values; the underlying sort is not stable, so with duplicate keys the "
    "returned indices are whichever the backend picked and are measurably "
    "different on CPU and CUDA, while stable, out, device, layout, and "
    "named-dimension semantics are not implemented"
)

from .ordering import sort

from .ordering import argsort

from .ordering import topk

from .ordering import median

for _ordering_impl in (sort, argsort, topk, median):
    register_fidelity(
        "torch." + _ordering_impl.__name__,
        _ordering_impl,
        Fidelity.APPROXIMATE,
        _ORDERING_FIDELITY_DETAIL,
    )

del _ordering_impl

_NATIVE_CUMSUM = jt.cumsum

_NATIVE_CUMPROD = getattr(jt, "cumprod", None)

_CUMULATIVE_FIDELITY_DETAIL = (
    "matches Torch cumulative values, the bool/uint8 -> int64 promotion, the "
    "dtype cast, and out identity for supported tensors on CPU and CUDA; the "
    "summation order is the backend's, so a float32 scan is reproducible per "
    "device but CPU and CUDA agree only to float32 rounding (~1e-6 relative "
    "over a few thousand elements, with the parallel scan the more accurate "
    "of the two), while device, layout, and named-dimension semantics are not "
    "implemented"
)

from .scans import _assign_out

from .scans import _cumulative

from .scans import cumsum

from .scans import cumprod

for _cumulative_name in (cumsum, cumprod):
    _cumulative_name._torch_accepts_axis = True
    register_fidelity(
        "torch." + _cumulative_name.__name__,
        _cumulative_name,
        Fidelity.APPROXIMATE,
        _CUMULATIVE_FIDELITY_DETAIL,
    )

del _cumulative_name

from .distributed import _ddp_all_reduce_grads

_MinMax = _collections.namedtuple("torch_return_types", ["values", "indices"])

_NATIVE_ARGMAX = jt.argmax

_NATIVE_ARGMIN = jt.argmin

_NATIVE_MAXIMUM = jt.maximum

_NATIVE_MINIMUM = jt.minimum

_NATIVE_MAX = jt.max

_NATIVE_MIN = jt.min

_NATIVE_MAX_METHOD = jt.Var.max

_NATIVE_MIN_METHOD = jt.Var.min

_NATIVE_VAR_METHOD = jt.Var.var

_ARG_REDUCTION_FIDELITY_DETAIL = (
    "matches Torch index values, the int64 index dtype, the axis alias, and "
    "the keepdim shape for supported real tensors, and CPU and CUDA agree bit "
    "for bit even on rows with duplicate keys (measured on 8x512 float32 with "
    "about five duplicates per key); Jittor's native argmax returns an "
    "(index, value) pair whose value half is dropped here, while out, device, "
    "layout, and named-dimension semantics are not implemented"
)

from .reductions import _reduce_index

from .reductions import argmax

from .reductions import argmin

_MINMAX_FIDELITY_DETAIL = (
    "matches Torch's values-only full reduction, the (values, indices) pair "
    "for a dim, the elementwise two-tensor form, the int64 index dtype, and "
    "the axis alias for supported real tensors, and CPU and CUDA agree bit for "
    "bit on both halves; the keepdims spelling is reserved for Jittor's own "
    "values-only callers, while out, device, layout, and named-dimension "
    "semantics are not implemented"
)

from .reductions import _maxmin

from .reductions import max

from .reductions import min

_VARIANCE_FIDELITY_DETAIL = (
    "matches Torch's unbiased default (correction=1), the correction, "
    "unbiased, keepdim and axis keywords, and a tuple of dims for supported "
    "real tensors, with std derived from var so it carries none of Jittor's "
    "1e-6 floor; the summation order is the backend's, so CPU and CUDA agree "
    "only to float32 rounding (about 1.3e-07 relative over 512 elements, with "
    "the parallel reduction the more accurate of the two), while out, device, "
    "layout, and named-dimension semantics are not implemented"
)

from .reductions import _correction_to_unbiased

from .reductions import _multidim_var

from .reductions import var

from .reductions import std

_MASKED_SCATTER_FIDELITY_DETAIL = (
    "matches Torch's row-major consumption of the source, mask broadcasting, "
    "the destination's dtype, and the in-place spelling's return identity for "
    "supported real tensors, stays differentiable through both operands, and "
    "CPU and CUDA agree bit for bit; out, device, and layout semantics are not "
    "implemented"
)

_UNFOLD_FIDELITY_DETAIL = (
    "matches Torch's sliding-window shape and values for a positive size and "
    "step on supported real tensors, and CPU and CUDA agree bit for bit; the "
    "result is a materialized reindex rather than a stride view, so writes to "
    "it do not reach the source, and device, layout, and dtype keyword "
    "semantics are not implemented"
)

_ADDC_FIDELITY_DETAIL = (
    "matches Torch's input + value * (tensor1 op tensor2) values for supported "
    "broadcastable real tensors but omits out, device, layout, and dtype "
    "keyword semantics"
)

_BROADCAST_TO_FIDELITY_DETAIL = (
    "matches Torch's expansion shape and values through Jittor's native "
    "broadcast for supported tensors but omits out, device, layout, and dtype "
    "keyword semantics"
)

from .indexing import masked_scatter

from .indexing import masked_scatter_

from .indexing import unfold

from .indexing import addcmul

from .indexing import addcdiv

from .indexing import broadcast_to

diagonal = jt.diagonal

_NATIVE_DIAGONAL_FIDELITY_DETAIL = (
    "re-exports Jittor's native diagonal owner, whose signature and values "
    "already match Torch for supported real tensors, negative offset and "
    "negative dim1/dim2 included, and CPU and CUDA agree bit for bit; the "
    "result is materialized rather than the stride view Torch returns, and "
    "out, device, layout, and dtype keyword semantics are not implemented"
)

for _reduction_impl, _reduction_api, _reduction_detail in (
        (argmax, "torch.argmax", _ARG_REDUCTION_FIDELITY_DETAIL),
        (argmin, "torch.argmin", _ARG_REDUCTION_FIDELITY_DETAIL),
        (max, "torch.max", _MINMAX_FIDELITY_DETAIL),
        (min, "torch.min", _MINMAX_FIDELITY_DETAIL),
        (var, "torch.var", _VARIANCE_FIDELITY_DETAIL),
        (std, "torch.std", _VARIANCE_FIDELITY_DETAIL),
        (masked_scatter, "torch.Tensor.masked_scatter",
         _MASKED_SCATTER_FIDELITY_DETAIL),
        (masked_scatter_, "torch.Tensor.masked_scatter_",
         _MASKED_SCATTER_FIDELITY_DETAIL),
        (unfold, "torch.Tensor.unfold", _UNFOLD_FIDELITY_DETAIL),
        (addcmul, "torch.Tensor.addcmul", _ADDC_FIDELITY_DETAIL),
        (addcdiv, "torch.Tensor.addcdiv", _ADDC_FIDELITY_DETAIL),
        (broadcast_to, "torch.broadcast_to", _BROADCAST_TO_FIDELITY_DETAIL),
        (diagonal, "torch.diagonal", _NATIVE_DIAGONAL_FIDELITY_DETAIL)):
    register_fidelity(
        _reduction_api, _reduction_impl, Fidelity.APPROXIMATE,
        _reduction_detail)

for _reduction_impl in (argmax, argmin, max, min, var, std):
    _reduction_impl._torch_accepts_axis = True

del _reduction_impl, _reduction_api, _reduction_detail

def _install_reductions(g):
    """Bind the reduction/mask/view family; every object here is module level.

    ``g`` IS the jittor module, so each name below is one stable object bound to
    both the module and the ``Var`` spelling. Binding the *same* object to both
    is the point: the two used to be built from separate closures and had
    measurably different behaviour (``torch.var(x, axis=0)`` reduced over
    everything while ``x.var(axis=0)`` reduced over axis 0).
    """
    Var = g.Var

    # jittor's own argmax returns (index, value) and its max(dim) returns
    # values only, so torch's contract needs the compat objects on both sides.
    g.argmax = Var.argmax = argmax
    g.argmin = Var.argmin = argmin
    g.max = Var.max = max
    g.min = Var.min = min
    g.var = Var.var = var
    g.std = Var.std = std

    # The ordering and cumulative families live at module level too; install
    # only binds them. jittor-core uses none of these as Var methods (only the
    # python list.sort builtin), so torch semantics here are safe; .max/.min ARE
    # used internally, which is why _maxmin keeps the native keepdims path.
    g.topk = Var.topk = topk
    g.sort = Var.sort = sort
    g.argsort = Var.argsort = argsort
    g.median = Var.median = median

    # Owned elsewhere: bind the one existing owner rather than a second copy.
    # ``softmax``/``log_softmax`` accept torch's ``dtype=`` (cast before the op),
    # which jittor's native method rejects -- vLLM's sampler spells it
    # ``logits.softmax(dim=-1, dtype=torch.float32)``.
    g.diagonal = Var.diagonal = diagonal
    g.masked_select = Var.masked_select = _numerical_masked_select
    g.softmax = Var.softmax = _numerical_softmax
    g.log_softmax = Var.log_softmax = _numerical_log_softmax

    # Methods with no module-level torch spelling to widen.
    Var.masked_scatter = masked_scatter
    Var.masked_scatter_ = masked_scatter_
    Var.unfold = unfold
    Var.addcmul = addcmul
    Var.addcdiv = addcdiv
    g.broadcast_to = Var.broadcast_to = broadcast_to

    # --- elementwise / reduction ops missing as torch methods (all additive) ---
    # sign/trunc/frac used to be installed here too, guarded by hasattr, while
    # core.install_misc -- which runs later -- owns torch.sign/torch.trunc and
    # overwrites Var.trunc unconditionally. The two copies did not agree, so the
    # family now has exactly one owner in installers/core.py.
    if not hasattr(Var, "nan_to_num"):
        Var.nan_to_num = nan_to_num
        g.nan_to_num = nan_to_num
    # amax/amin/count_nonzero already have a native owner (jittor.misc.reductions)
    # whose contract is the Torch one; bind the stable compat objects that wrap it
    # rather than carrying a second copy of the reduction here.
    if not hasattr(Var, "amax"):
        Var.amax = amax
        Var.amin = amin
        g.amax = amax
        g.amin = amin
    if not hasattr(Var, "count_nonzero"):
        Var.count_nonzero = count_nonzero
        g.count_nonzero = count_nonzero
    if not hasattr(g, "logaddexp"):
        g.logaddexp = logaddexp
        Var.logaddexp = logaddexp

from .methods import _install_tensor_methods

from ...tensor_state import compatibility_owner
from ...frontend import tensor_frontend

_native_grad = jt.grad
_native_index_select = jt.index_select
_jt_concat = jt.concat
_jt_stack = jt.stack
_orig_no_grad = jt.no_grad
_orig_enable_grad = jt.enable_grad


def no_grad(func=None):
    return _GradDecoratorCtx(_orig_no_grad, func)


def enable_grad(func=None):
    return _GradDecoratorCtx(_orig_enable_grad, func)


def inference_mode(func=None):
    return _GradDecoratorCtx(_orig_no_grad, func)

def _grad_compat(loss, targets, *a, **k):
    # A lone Var target must return a lone grad (native jt.grad unwraps it via
    # core.grad(...)[0]). Wrapping it into [targets] here made jt.grad(loss, var)
    # return a 1-element LIST instead of a Var, breaking single-target callers
    # (e.g. softmax/ctc backward in test_misc_op). Remember the single-Var case and
    # unwrap the result to restore native behavior; list/iterable targets pass through.
    single = isinstance(targets, jt.Var)
    if type(targets) is not list:
        if single:
            targets = [targets]
        else:
            try:
                targets = list(targets)
            except EXPECTED as exc:
                swallowed("torch/installers/tensor.py _grad_compat: targets = list(targets)", exc)
    res = _native_grad(loss, targets, *a, **k)
    if single and isinstance(res, (list, tuple)) and len(res) == 1:
        return res[0]
    return res


def _index_select(input, dim, index, *, out=None):
    result = _native_index_select(input, dim, index)
    if out is not None:
        out[...] = result
        return out
    return result


class _TensorMeta(type):
    def __instancecheck__(cls, inst):
        return isinstance(inst, (compatibility_owner(jt).Var, _NestedTensor))
    def __subclasscheck__(cls, sub):
        return issubclass(sub, (compatibility_owner(jt).Var, _NestedTensor))
    def __call__(cls, *args, **kw):
        if len(args) == 0:
            return jt.empty((0,))
        if all(isinstance(a, int) for a in args):   # torch.Tensor(*sizes)
            return jt.empty(tuple(args))
        # torch.Tensor(size) with a shape object (torch.Size / our Size / a
        # jittor NanoVector, e.g. weight.size()) -> an uninitialized tensor of
        # that shape, NOT data (mmdet SAConv2d: torch.Tensor(self.weight.size())).
        if len(args) == 1 and isinstance(args[0], (jt.NanoVector, _TorchSize)):
            return jt.empty(tuple(int(x) for x in args[0]))
        data = args[0]
        if isinstance(data, compatibility_owner(jt).Var):
            return data.float32()
        return jt.array(data).float32()


class Tensor(metaclass=_TensorMeta):
    pass


class _TypedTensorMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, compatibility_owner(jt).Var) and _jittor_dtype_name(obj.dtype) == cls._jdtype
    def __call__(cls, *args, **kw):
        with tensor_frontend(compatibility_owner(jt).Var):
            tensor_input = len(args) == 1 and isinstance(args[0], compatibility_owner(jt).Var)
            if tensor_input:
                v = args[0]
            elif len(args) == 1 and not isinstance(args[0], int):
                v = jt.array(args[0], dtype=cls._jdtype)
            elif len(args) == 0:
                v = jt.zeros((0,), dtype=cls._jdtype)
            else:
                v = jt.zeros(tuple(int(a) for a in args), dtype=cls._jdtype)
            result = v.cast(cls._jdtype)
            if not tensor_input:
                result.requires_grad_(False)
            return result


def _array_keep_dtype(data):
    # jittor's jt.array downcasts numpy int64 -> int32; torch keeps int64.
    # Preserve the source dtype for (u)int64/float64 so dtypes match torch.
    import numpy as _np
    # jt.array rejects ndarray SUBCLASSES (e.g. the adapter's numpy-backed
    # buffer tensors) -> coerce to a base ndarray (same data, no copy).
    if isinstance(data, _np.ndarray) and type(data) is not _np.ndarray:
        data = _np.asarray(data)
    if isinstance(data, _np.ndarray):
        dn = data.dtype.name
        # jt.array(numpy_int64) silently downcasts to int32, OVERFLOWING values
        # that don't fit in 32 bits (e.g. byte counts ~1e10) BEFORE any later
        # .int64() cast can recover them. Build the wide-dtype Var directly.
        if dn in ("int64", "uint64"):
            return jt.array(data, dtype="int64")
        if dn == "float64":
            return jt.array(data, dtype="float64")
    return jt.array(data)


def tensor(data, dtype=None, device=None, requires_grad=False, **kw):
    g = compatibility_owner(jt)
    Var = g.Var
    with tensor_frontend(Var):
        import numpy as _np
        ds = _dtype_to_str(dtype)
        numpy_dtypes = {"bool", "uint8", "int8", "int16", "int32", "int64",
                        "uint16", "uint32", "uint64", "float16", "float32", "float64",
                        "complex64", "complex128"}
        storage_dtype = ds if ds in numpy_dtypes else "float32" if ds == "bfloat16" else None
        if isinstance(data, jt.Var):
            v = jt.Var.copy(data).detach()
        elif isinstance(data, (_np.ndarray, _np.generic)):
            # NumPy input carries its own dtype; explicit conversion happens
            # before the native array constructor can narrow it.
            data = _np.asarray(data, dtype=_jittor_dtype_name(storage_dtype))
            v = _array_keep_dtype(data)          # explicit numpy: preserve dtype (torch does too)
        else:
            # torch's tensor/as_tensor([t1, t2, ...]) flattens SCALAR tensors into a
            # 1-D tensor; jittor has no 0-d scalars (a "scalar" Var is shape (1,)), so
            # numpy.asarray of a list-of-Vars adds a spurious dim ((1,)->(1,1)). Coerce
            # contained scalar Vars to Python numbers first (e.g. tapas builds shapes
            # via torch.as_tensor([index.num_segments])).
            if isinstance(data, (list, tuple)) and any(isinstance(d, Var) for d in data):
                data = [(d.item() if isinstance(d, Var) and d.numel() == 1 else d)
                        for d in data]
            # Resolve Python defaults before constructing native storage. An
            # explicit float64 value must never pass through float32 first.
            arr = _np.asarray(data, dtype=_jittor_dtype_name(storage_dtype))
            if ds is None and arr.dtype.kind in ("f", "c"):
                getter = getattr(g, "get_default_dtype", None)
                default_dtype = _dtype_to_str(getter()) if getter is not None else "float32"
                if arr.dtype.kind == "c":
                    complex_dtype = {"float32": "complex64", "float64": "complex128"}.get(default_dtype)
                    if complex_dtype is None:
                        raise NotImplementedError("complex construction for default dtype %s" % _jittor_dtype_name(default_dtype))
                    ds = complex_dtype
                else:
                    ds = default_dtype
                arr = arr.astype("float32" if ds == "bfloat16" else ds)
            v = _array_keep_dtype(arr)
        if ds is not None:
            v = v.cast(ds)
        # torch.tensor(..., device='cpu') must land in host memory so native
        # extensions' tensor.is_cpu() checks pass.
        if _device_is_cpu(device):
            v = _make_cpu_resident(v)
        elif _device_is_cuda(device):
            _set_use_cuda()
            v = _make_cuda_resident(v, force=True)
            v = _move_to_cuda_index(v, g.device(device))
        if g is not jt:
            v.requires_grad_(bool(requires_grad))
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        if g is jt:
            v._jt_plain_tensor = True  # Only the explicit legacy Module adapter reads this.
        return v


def as_tensor(data, dtype=None, device=None):
    g = compatibility_owner(jt)
    Var = g.Var
    with tensor_frontend(Var):
        if isinstance(data, jt.Var):
            r = data if isinstance(data, Var) else g.Tensor(data)
            if dtype is not None and _jittor_dtype_name(r.dtype) != _dtype_to_str(dtype):
                r = r.cast(_dtype_to_str(dtype))
            if _device_is_cpu(device):
                return _make_cpu_resident(r)
            if _device_is_cuda(device):
                _set_use_cuda()
                return _move_to_cuda_index(_make_cuda_resident(r, force=True), g.device(device))
            return r
        return tensor(data, dtype=dtype, device=device)


def from_numpy(arr, *, device=None):
    g = compatibility_owner(jt)
    Var = g.Var
    with tensor_frontend(Var):
        v = _array_keep_dtype(arr)
        if g is not jt:
            v.requires_grad_(False)
        if _device_is_cpu(device):
            return _make_cpu_resident(v)
        if _device_is_cuda(device):
            _set_use_cuda()
            return _move_to_cuda_index(_make_cuda_resident(v, force=True), g.device(device))
        return v


def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False):
    g = compatibility_owner(jt)
    Var = g.Var
    with tensor_frontend(Var):
        import numpy as _np
        ds = _dtype_to_str(dtype)
        np_dtype = {
            "bool": _np.bool_, "uint8": _np.uint8, "int8": _np.int8,
            "uint16": _np.uint16, "int16": _np.int16,
            "uint32": _np.uint32, "int32": _np.int32,
            "uint64": _np.uint64, "int64": _np.int64,
            "float16": _np.float16, "float32": _np.float32,
            "float64": _np.float64,
        }.get(ds)
        if ds == "bfloat16":
            raw = _np.frombuffer(buffer, dtype=_np.uint16, count=count, offset=offset)
            arr = (raw.astype(_np.uint32) << 16).view(_np.float32)
            v = from_numpy(_np.ascontiguousarray(arr))
        else:
            if np_dtype is None:
                raise TypeError(f"torch.frombuffer unsupported dtype: {_jittor_dtype_name(dtype)}")
            arr = _np.frombuffer(buffer, dtype=_jittor_dtype_name(np_dtype), count=count, offset=offset)
            v = from_numpy(_np.ascontiguousarray(arr))
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        return v


class Generator:
    def __init__(self, device=None):
        self.device = globals()["device"](device or "cpu")
        self._seed = 0
    def manual_seed(self, s):
        self._seed = int(s)
        return self
    def get_state(self):
        return jt.array([self._seed])
    def set_state(self, s):
        return self
    def seed(self):
        return self._seed
    def initial_seed(self):
        return self._seed


class layout:  # torch.layout placeholder
    pass


class memory_format:
    pass


def _nested_from_tensors(tensors, *a, layout=None, **k):
    return _NestedTensor.from_tensors(tensors, ragged_idx=k.pop("ragged_idx", 1))


def _nested_from_jagged(values, offsets, *a, **k):
    return _NestedTensor.from_jagged(values, offsets, ragged_idx=k.pop("ragged_idx", None))


def _check_to_pybool(cond):
    if hasattr(cond, "all") and not isinstance(cond, (bool, int, float)):
        try:
            return bool(cond.all().item())
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _check_to_pybool: return bool(cond.all().item())", exc)
            return bool(cond)
    return bool(cond)


def cat(tensors, dim=0, out=None, axis=None):
    g = compatibility_owner(jt)
    Var = g.Var
    if axis is not None: dim = axis      # torch accepts axis= (mmrotate PSC head)
    # Honor the __torch_function__ protocol: tensordict (and other tensor-likes)
    # override torch.cat to handle their own structure -- e.g. cat a list of
    # TensorDicts field-by-field. Without this, jittor's concat treats each
    # TensorDict as a Var (dtype None) and aborts. Delegate to the first arg
    # whose type overrides __torch_function__ (Vars are handled normally below).
    try:
        _seq = list(tensors)
    except TypeError:
        _seq = None
    if _seq is not None:
        if any(isinstance(_t, _NestedTensor) for _t in _seq):
            assert all(isinstance(_t, _NestedTensor) for _t in _seq), "cannot cat nested and dense tensors together"
            if dim == 0:
                parts = []
                for _t in _seq:
                    parts.extend(list(_t.unbind(0)))
                return _NestedTensor.from_tensors(
                    parts,
                    ragged_idx=getattr(_seq[0], "_ragged_idx", _seq[0].dim() - 1),
                )
            assert all(len(_t) == len(_seq[0]) for _t in _seq), "nested cat with dim!=0 requires same batch size"
            return _NestedTensor.from_tensors(
                [_jt_concat([_t.unbind(0)[i] for _t in _seq], dim=dim - 1) for i in range(len(_seq[0]))],
                ragged_idx=getattr(_seq[0], "_ragged_idx", _seq[0].dim() - 1),
            )
        for _t in _seq:
            _tf = getattr(type(_t), "__torch_function__", None)
            if _tf is not None and not isinstance(_t, jt.Var):
                _kw = {}
                if dim != 0: _kw["dim"] = dim
                if out is not None: _kw["out"] = out
                return _tf(g.cat, (type(_t),), (_seq,), _kw)
    tensors = [t for t in tensors if t is not None]
    nonempty = [t for t in tensors if t.numel() > 0]
    if len(nonempty) == 0:
        return tensors[0]
    if len(nonempty) == 1:
        return nonempty[0]
    # torch requires all tensors to share ndim. jittor has no 0-d scalars, so
    # a torch-scalar `s` (0-d) becomes a [1] Var and `s.unsqueeze(0)` yields
    # [1,1] instead of torch's [1] -- mixing 2-D and 1-D entries that torch
    # would see as uniformly 1-D (e.g. SOLO's per-image dice losses). Strip
    # the spurious LEADING size-1 dims off any over-ranked entry so the ndims
    # line up the way torch sees them. Only size-1 leading dims are removed;
    # a genuine ndim/shape mismatch is left for jittor's concat to reject.
    # ``min`` is this module's torch reduction, so reach for the builtin.
    min_nd = _builtins.min(t.ndim for t in nonempty)
    fixed = []
    for t in nonempty:
        while t.ndim > min_nd and t.shape[0] == 1:
            t = t.squeeze(0)
        fixed.append(t)
    out_var = _jt_concat(fixed, dim)
    # jittor's concat downcasts a uniform uint8 input to int8 (e.g. mask-rcnn-c4
    # builds a uint8 pos_inds mask via torch.cat of uint8 ones/zeros). torch keeps
    # the common input dtype; restore it so downstream byte-mask indexing works.
    in_dtypes = {_jittor_dtype_name(t.dtype) for t in fixed}
    if len(in_dtypes) == 1:
        d = in_dtypes.pop()
        if _jittor_dtype_name(out_var.dtype) != d:
            out_var = out_var.cast(d)
    return out_var


def stack(tensors, dim=0, *, axis=None, out=None):
    if axis is not None: dim = axis
    res = _jt_stack(list(tensors), dim)
    if out is not None:
        out.assign(res)
        return out
    return res


class FloatTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'float32'


class DoubleTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'float64'


class HalfTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'float16'


class BFloat16Tensor(metaclass=_TypedTensorMeta):
    _jdtype = 'bfloat16'


class LongTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'int64'


class IntTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'int32'


class ShortTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'int16'


class CharTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'int8'


class ByteTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'uint8'


class BoolTensor(metaclass=_TypedTensorMeta):
    _jdtype = 'bool'


_TYPED_TENSOR_CLASSES = {
    'FloatTensor': FloatTensor,
    'DoubleTensor': DoubleTensor,
    'HalfTensor': HalfTensor,
    'BFloat16Tensor': BFloat16Tensor,
    'LongTensor': LongTensor,
    'IntTensor': IntTensor,
    'ShortTensor': ShortTensor,
    'CharTensor': CharTensor,
    'ByteTensor': ByteTensor,
    'BoolTensor': BoolTensor,
}

def _check_condition(condition, message, exception):
    if not _check_to_pybool(condition):
        text = message() if callable(message) else message
        raise exception(text if text is not None else "Expected cond to be True, but got False")

def _check(condition, message=None):
    return _check_condition(condition, message, RuntimeError)


def _check_index(condition, message=None):
    return _check_condition(condition, message, IndexError)


def _check_value(condition, message=None):
    return _check_condition(condition, message, ValueError)


def _check_type(condition, message=None):
    return _check_condition(condition, message, TypeError)


def _check_not_implemented(condition, message=None):
    return _check_condition(condition, message, NotImplementedError)


def _check_tensor_all(condition, message=None):
    return _check_condition(condition, message, RuntimeError)


def _check_is_size(i, message=None, **kwargs):
    return _check(int(i) >= 0, message)


def _assert_async(t, *args, **kwargs):
    return _check(_check_to_pybool(t), "torch._assert_async failed")


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    _DTYPE_OBJS = ctx.state["dtypes"]
    # jt.grad's C-binding only accepts a *plain* list of targets, so passing the
    # torch-style parameters() iterator/_ParamList (a list subclass) or a single
    # Var raises a cryptic "Wrong inputs arguments". Coerce to a plain list (and
    # accept a lone Var, like torch.autograd.grad). Internal jittor callers pass a
    # plain list -> passthrough, so this never changes their behavior.
    g.grad = _grad_compat

    # torch.no_grad / enable_grad work as bare decorator (@torch.no_grad),
    # called decorator (@torch.no_grad()), and context manager.
    # NB: g IS the jittor module, so capture the originals before overwriting.
    g.no_grad = no_grad
    g.enable_grad = enable_grad
    g.inference_mode = inference_mode

    Var = ctx.state["Var"]
    from ...frontend import frontend_factory
    g.index_select = _index_select
    # torch.Tensor is both (a) the isinstance target and (b) a legacy constructor:
    # torch.Tensor(d0, d1, ...) makes an UNINITIALISED tensor of that shape (DETR's
    # _init_layers: torch.Tensor(num_levels, embed_dims)), while torch.Tensor(data)
    # builds from data. A metaclass gives us both without breaking isinstance(x, Var).
    g.Tensor = Var if g is not ctx.native_backend else Tensor
    # torch's typed tensor classes (FloatTensor/LongTensor/...). jittor is dtype-typed
    # at the data level (no tensor subclasses), but we must NOT just alias them all to
    # Var: that makes isinstance(any_var, torch.LongTensor) always True, so libraries
    # that detect integer tensors via isinstance break with silent-wrong results
    # (e.g. diffusers EulerDiscreteScheduler.step rejects every float timestep with
    # "Passing integer indices ... is not supported"). Instead give each a metaclass
    # whose isinstance check matches the Var's actual dtype, and whose construction
    # casts to that dtype (torch.FloatTensor(2,3) / torch.LongTensor([1,2])).
    for name, tensor_class in _TYPED_TENSOR_CLASSES.items():
        setattr(g, name, tensor_class)


    g.tensor = tensor

    g.as_tensor = as_tensor

    g.from_numpy = from_numpy

    g.frombuffer = frombuffer

    Size = _TorchSize
    g.Size = Size

    g.broadcast_shapes = broadcast_shapes

    g.corrcoef = corrcoef

    # torch.Generator (RNG handle) -- jittor uses a global seed; provide a
    # lightweight stand-in that supports manual_seed and is accepted where a
    # generator is passed (it is otherwise ignored).
    g.Generator = Generator

    # numeric / misc top-level constants and small types
    import math as _math
    g.inf = _math.inf
    g.nan = _math.nan
    g.pi = _math.pi
    g.e = _math.e
    g.strided = "strided"
    g.jagged = "jagged"
    g.contiguous_format = "contiguous_format"
    g.preserve_format = "preserve_format"
    g.channels_last = "channels_last"
    g.layout = layout
    g.memory_format = memory_format

    import types as _types_nested
    nested_mod = _types_nested.ModuleType("torch.nested")
    nested_mod.__path__ = []
    nested_mod.as_nested_tensor = _nested_from_tensors
    nested_mod.nested_tensor = _nested_from_tensors
    nested_mod.nested_tensor_from_jagged = _nested_from_jagged
    g.nested = nested_mod
    _modules["torch.nested"] = nested_mod
    nested_internal_mod = _types_nested.ModuleType("torch.nested._internal")
    nested_internal_mod.__path__ = []
    nested_tensor_mod = _types_nested.ModuleType("torch.nested._internal.nested_tensor")
    nested_tensor_mod.NestedTensor = _NestedTensor
    nested_internal_mod.nested_tensor = nested_tensor_mod
    nested_mod._internal = nested_internal_mod
    _modules["torch.nested._internal"] = nested_internal_mod
    _modules["torch.nested._internal.nested_tensor"] = nested_tensor_mod

    # torch._check family: assertion helpers used by dynamo / TorchScript-friendly
    # code (e.g. vLLM's sampler does `torch._check(x.shape[0] >= 1)`). The message
    # may be a zero-arg callable that torch invokes lazily only on failure. The
    # condition is usually a python bool but can be a bool tensor (_check_tensor_all).
    g._check = _check
    g._check_is_size = _check_is_size
    g._check_index = _check_index
    g._check_value = _check_value
    g._check_type = _check_type
    g._check_not_implemented = _check_not_implemented
    g._check_tensor_all = _check_tensor_all
    g._assert_async = _assert_async

    # torch.cat: tolerate empty tensors (skip zero-numel inputs) like torch,
    # accept `dim=`/`out=`. jittor's concat trips on an empty leading tensor.
    g.cat = cat
    g.concat = cat
    g.concatenate = cat

    # torch.stack accepts a numpy-style `axis=` alias for `dim=` (and `out=`); jittor's
    # jt.stack is `stack(x, dim=0)` only, so trl's PPO advantage stacking
    # `torch.stack(advantages_reversed[::-1], axis=1)` dies on the unexpected kwarg.
    g.stack = stack

    # Wrap tensor constructors to tolerate torch's device=/requires_grad=/
    # layout=/pin_memory= kwargs and torch dtype objects. jittor's versions
    # don't accept device=, which torch code passes everywhere.
    _wrap_constructors(g)
    _install_random_and_linspace(g)

    _install_reductions(g)

    register_api_bindings(g, 'torch',
        ('Generator', 'Size', 'Tensor', 'as_tensor', 'broadcast_shapes', 'cat', 'channels_last', 'concat', 'concatenate', 'contiguous_format', 'corrcoef', 'e', 'enable_grad', 'from_numpy', 'frombuffer', 'grad', 'index_select', 'inf', 'inference_mode', 'jagged', 'layout', 'memory_format', 'nan', 'nested', 'no_grad', 'pi', 'preserve_format', 'stack', 'strided', 'tensor') + tuple(_TYPED_TENSOR_CLASSES.keys()),
        Fidelity.APPROXIMATE, 'Native tensor allocation and conversion with Torch dtype, device, and frontend policies; supported argument and backend subsets apply')

from .shape_api import (
    _SHAPE_REDUCTION_APIS,
    _shape_relu,
    _shape_relu_,
    _shape_eq,
    _shape_ne,
    _shape_gt,
    _shape_ge,
    _shape_lt,
    _shape_le,
    _shape_neg,
    _shape_reciprocal,
    _shape_expm1,
    _shape_log1p,
    _shape_square,
    _shape_square_,
    _shape_clamp_min,
    _shape_clamp_max,
    _shape_bmm,
    _shape_mm,
    _shape_mv,
    _shape_fliplr,
    _shape_flipud,
    _shape_diff,
    _shape_trapz,
    _shape_trapezoid,
    _shape_fmod,
    _shape_remainder,
    _shape_softplus,

    _bitcast,
    _dtype_itemsize_name,
    _index_add_inplace,
    _looks_like_dtype,
    _norm_reduce_kw,
    _torch_reshape,
    _torch_size,
    _torch_sum,
    _torch_var_sum,
)
from types import MappingProxyType

def install_methods(ctx):
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    _install_tensor_methods(g, Var, _DTYPE_OBJS)
    # torch's Tensor.size() returns a torch.Size (tuple subclass) when called with
    # no arg, and an int for size(dim); jittor's native size() returns a NanoVector,
    # which breaks torch idioms like `(n,) + data.size()[1:]` (mmdet's unmap()).
    _Size = getattr(g, "Size", tuple)
    Var.size = _torch_size

    # jittor's core reshape/view reject a torch.Size (a tuple SUBCLASS) -> normalize
    # a single Size/tuple-subclass arg to a plain tuple so `x.reshape(other.size())`
    # / `x.view(t.size())` works (mmdet queryinst). Only intervene for that case to
    # keep the (very hot) reshape path otherwise untouched.
    _orig_reshape = Var.reshape
    _np_view_of = None
    Var.reshape = _torch_reshape
    Var.view = _torch_reshape

    # Keep the existing Torch-facing promotion for narrow integer sums
    # (yolox/rtmdet SimOTA assigners do mask.sum() on a uint8 match matrix).
    # Native CUDA reductions now support these dtypes directly, but exposing the
    # narrow native output here would change the compatibility-layer dtype policy.
    # torch reductions accept a *tuple* of dims (e.g. loss.mean(dim=(1, 2)) in
    # yolact_head, x.sum(dim=(2, 3))). jittor splits these into a scalar overload
    # (kwarg `dim`, single int) and a tuple overload (kwarg `dims`); passing a tuple
    # under `dim` raises "Not a valid keyword: dim". Normalize: route a tuple/list of
    # dims to `dims`, a scalar to `dim`, accepting it via `axis`, `dim`, or as the
    # first positional arg (torch also allows axis as a dim alias).


    _orig_var_sum = Var.sum
    _native_reductions = {}
    _orig_module_sum = getattr(g, "sum", None)
    Var.sum = _torch_var_sum
    if _orig_module_sum is not None:
        g.sum = _torch_sum
    # Full dim/dims/keepdim normalization for the plain reductions that map onto
    # jittor's scalar-`dim` / tuple-`dims` overload pair (mean/prod/any/all). mmdet
    # exercises tuple dims here, e.g. yolact_head's loss.mean(dim=(1, 2)).
    for _rn in ("mean", "prod"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            _native_reductions[_rn] = _ro
            setattr(Var, _rn, _SHAPE_REDUCTION_APIS[_rn])
    # any/all: jittor's only accept a scalar `dim` (no `dims` tuple, no keepdims).
    # Support torch's tuple-of-dims and keepdim by reducing one dim at a time
    # (descending so earlier dim indices stay valid), keeping a length-1 axis when
    # keepdim is set. Plain scalar/axis use falls through to the native op.
    for _rn in ("any", "all"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            _native_reductions[_rn] = _ro
            setattr(Var, _rn, _SHAPE_REDUCTION_APIS[_rn])
    # max/min/argmax/argmin/amax/amin/cumsum/norm/std/var are already wrapped above
    # with custom torch-return semantics (value+index tuples, etc.); only translate
    # torch's `axis` alias for them so we don't disturb that handling.
    for _rn in ("max", "min", "argmax", "argmin", "amax", "amin", "cumsum",
                "norm", "std", "var"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            _native_reductions[_rn] = _ro
            if not getattr(_ro, "_torch_accepts_axis", False):
                setattr(Var, _rn, _SHAPE_REDUCTION_APIS[_rn])

    # ---- Tensor methods used by mmdetection + cheap torch-standard completeness ----
    # (.relu 86x, .eq 11x, .gt 12x, .diff, .fliplr are exercised by mmdet; the rest
    #  are one-line torch standards added to reduce downstream surprises.)
    if not hasattr(Var, "relu"):        Var.relu = _shape_relu
    if not hasattr(Var, "relu_"):       Var.relu_ = _shape_relu_
    if not hasattr(Var, "eq"):          Var.eq = _shape_eq
    if not hasattr(Var, "ne"):          Var.ne = _shape_ne
    if not hasattr(Var, "gt"):          Var.gt = _shape_gt
    if not hasattr(Var, "ge"):          Var.ge = _shape_ge
    if not hasattr(Var, "lt"):          Var.lt = _shape_lt
    if not hasattr(Var, "le"):          Var.le = _shape_le
    if not hasattr(Var, "neg"):         Var.neg = _shape_neg
    if not hasattr(Var, "reciprocal"):  Var.reciprocal = _shape_reciprocal
    if not hasattr(Var, "expm1"):       Var.expm1 = _shape_expm1
    if not hasattr(Var, "log1p"):       Var.log1p = _shape_log1p
    if not hasattr(Var, "square"):      Var.square = _shape_square
    if not hasattr(Var, "square_"):     Var.square_ = _shape_square_
    if not hasattr(Var, "clamp_min"):   Var.clamp_min = _shape_clamp_min
    if not hasattr(Var, "clamp_max"):   Var.clamp_max = _shape_clamp_max
    _orig_index_add_inplace = getattr(Var, "index_add_", None)
    if _orig_index_add_inplace is not None and not getattr(_orig_index_add_inplace, "_torch_returns_self", False):
        _index_add_inplace._torch_returns_self = True
        Var.index_add_ = _index_add_inplace
    if not hasattr(Var, "bmm"):         Var.bmm = _shape_bmm
    if not hasattr(Var, "mm"):          Var.mm = _shape_mm
    if not hasattr(Var, "mv"):          Var.mv = _shape_mv
    if not hasattr(Var, "fliplr"):      Var.fliplr = _shape_fliplr
    if not hasattr(Var, "flipud"):      Var.flipud = _shape_flipud
    if not hasattr(Var, "diff"):
        Var.diff = _shape_diff
    if not hasattr(Var, "trapz"):
        Var.trapz = _shape_trapz
    if not hasattr(Var, "trapezoid"):
        Var.trapezoid = _shape_trapezoid
    if not hasattr(Var, "fmod"):        # truncated remainder, sign of dividend
        Var.fmod = _shape_fmod
    if not hasattr(Var, "remainder"):   # floored remainder, sign of divisor
        Var.remainder = _shape_remainder
    if not hasattr(Var, "softplus"):    Var.softplus = _shape_softplus

    ctx.state["tensor_shape_api"] = MappingProxyType({
        "reductions": MappingProxyType(_native_reductions),
        '_Size': locals().get('_Size'),
        '_orig_index_add_inplace': locals().get('_orig_index_add_inplace'),
        '_orig_module_sum': locals().get('_orig_module_sum'),
        '_orig_reshape': locals().get('_orig_reshape'),
        '_orig_var_sum': locals().get('_orig_var_sum'),
    })

    register_api_bindings(Var, 'torch.Tensor',
        ('bmm', 'clamp_max', 'clamp_min', 'diff', 'eq', 'expm1', 'fliplr', 'flipud', 'fmod', 'ge', 'gt', 'index_add_', 'le', 'log1p', 'lt', 'mm', 'mv', 'ne', 'neg', 'reciprocal', 'relu', 'relu_', 'remainder', 'reshape', 'size', 'softplus', 'square', 'square_', 'sum', 'trapezoid', 'trapz', 'view') + tuple(_SHAPE_REDUCTION_APIS.keys()),
        Fidelity.APPROXIMATE, 'Tensor operations share the native Var/Op graph and explicit frontend state; unsupported layouts, device capabilities, and retained compatibility approximations remain restricted')
