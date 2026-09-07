"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

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

def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    _DTYPE_OBJS = ctx.state["dtypes"]
    # jt.grad's C-binding only accepts a *plain* list of targets, so passing the
    # torch-style parameters() iterator/_ParamList (a list subclass) or a single
    # Var raises a cryptic "Wrong inputs arguments". Coerce to a plain list (and
    # accept a lone Var, like torch.autograd.grad). Internal jittor callers pass a
    # plain list -> passthrough, so this never changes their behavior.
    _native_grad = g.grad
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
    g.grad = _grad_compat

    # torch.no_grad / enable_grad work as bare decorator (@torch.no_grad),
    # called decorator (@torch.no_grad()), and context manager.
    # NB: g IS the jittor module, so capture the originals before overwriting.
    _orig_no_grad = jt.no_grad
    _orig_enable_grad = jt.enable_grad
    g.no_grad = lambda func=None: _GradDecoratorCtx(_orig_no_grad, func)
    g.enable_grad = lambda func=None: _GradDecoratorCtx(_orig_enable_grad, func)
    g.inference_mode = lambda func=None: _GradDecoratorCtx(_orig_no_grad, func)

    Var = ctx.state["Var"]
    from ...frontend import frontend_factory
    _native_index_select = g.index_select
    def _index_select(input, dim, index, *, out=None):
        result = _native_index_select(input, dim, index)
        if out is not None:
            out[...] = result
            return out
        return result
    g.index_select = _index_select
    # torch.Tensor is both (a) the isinstance target and (b) a legacy constructor:
    # torch.Tensor(d0, d1, ...) makes an UNINITIALISED tensor of that shape (DETR's
    # _init_layers: torch.Tensor(num_levels, embed_dims)), while torch.Tensor(data)
    # builds from data. A metaclass gives us both without breaking isinstance(x, Var).
    class _TensorMeta(type):
        def __instancecheck__(cls, inst):
            return isinstance(inst, (Var, _NestedTensor))
        def __subclasscheck__(cls, sub):
            return issubclass(sub, (Var, _NestedTensor))
        def __call__(cls, *args, **kw):
            if len(args) == 0:
                return jt.empty((0,))
            if all(isinstance(a, int) for a in args):   # torch.Tensor(*sizes)
                return jt.empty(tuple(args))
            # torch.Tensor(size) with a shape object (torch.Size / our Size / a
            # jittor NanoVector, e.g. weight.size()) -> an uninitialized tensor of
            # that shape, NOT data (mmdet SAConv2d: torch.Tensor(self.weight.size())).
            if len(args) == 1 and isinstance(args[0], (jt.NanoVector, Size)):
                return jt.empty(tuple(int(x) for x in args[0]))
            data = args[0]
            if isinstance(data, Var):
                return data.float32()
            return jt.array(data).float32()
    class Tensor(metaclass=_TensorMeta):
        pass
    g.Tensor = Var if g is not ctx.native_backend else Tensor
    # torch's typed tensor classes (FloatTensor/LongTensor/...). jittor is dtype-typed
    # at the data level (no tensor subclasses), but we must NOT just alias them all to
    # Var: that makes isinstance(any_var, torch.LongTensor) always True, so libraries
    # that detect integer tensors via isinstance break with silent-wrong results
    # (e.g. diffusers EulerDiscreteScheduler.step rejects every float timestep with
    # "Passing integer indices ... is not supported"). Instead give each a metaclass
    # whose isinstance check matches the Var's actual dtype, and whose construction
    # casts to that dtype (torch.FloatTensor(2,3) / torch.LongTensor([1,2])).
    _TYPED_TENSOR_DTYPE = {
        "FloatTensor": "float32", "DoubleTensor": "float64", "HalfTensor": "float16",
        "BFloat16Tensor": "bfloat16", "LongTensor": "int64", "IntTensor": "int32",
        "ShortTensor": "int16", "CharTensor": "int8", "ByteTensor": "uint8",
        "BoolTensor": "bool",
    }
    class _TypedTensorMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, Var) and str(obj.dtype) == cls._jdtype
        def __call__(cls, *args, **kw):
            if len(args) == 1 and isinstance(args[0], Var):
                v = args[0]
            elif len(args) == 1 and not isinstance(args[0], int):
                v = jt.array(args[0])           # from list/ndarray
            elif len(args) == 0:
                v = jt.zeros((0,))
            else:
                v = jt.zeros(tuple(int(a) for a in args))  # from sizes
            return v.cast(cls._jdtype)
    _TypedTensorMeta.__call__ = frontend_factory(_TypedTensorMeta.__call__, Var)
    for _tn, _dt in _TYPED_TENSOR_DTYPE.items():
        setattr(g, _tn, _TypedTensorMeta(_tn, (), {"_jdtype": _dt}))

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
        import numpy as _np
        ds = _dtype_to_str(dtype)
        if isinstance(data, Var):
            v = data.clone()
        elif isinstance(data, _np.ndarray):
            # Respect an explicit complex64 request before constructing the Var.
            # NumPy otherwise keeps complex literals as unsupported complex128,
            # so casting only after jt.array() is too late.
            if ds == "complex64" and data.dtype.name != "complex64":
                data = _np.asarray(data, dtype=_np.complex64)
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
            # Python scalar/list/tuple: numpy infers float64 from Python floats, but
            # torch's default float dtype is float32. Match torch (and avoid float64,
            # which Ascend/ACL does not support) by downcasting inferred float64.
            arr = _np.asarray(data, dtype=_np.complex64 if ds == "complex64" else None)
            if arr.dtype == _np.float64:
                arr = arr.astype(_np.float32)
            elif arr.dtype == _np.complex128 and ds != "complex128":
                # torch's default complex dtype follows its default float dtype,
                # so Python complex literals default to complex64.
                arr = arr.astype(_np.complex64)
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
        if g is not ctx.native_backend:
            v.requires_grad_(bool(requires_grad))
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        v._jt_plain_tensor = True   # see _torch_style_registration (core_api)
        return v
    tensor = frontend_factory(tensor, Var)
    g.tensor = tensor

    def as_tensor(data, dtype=None, device=None):
        if isinstance(data, Var):
            r = data if dtype is None else data.cast(_dtype_to_str(dtype))
            if _device_is_cpu(device):
                return _make_cpu_resident(r)
            if _device_is_cuda(device):
                _set_use_cuda()
                return _make_cuda_resident(r, force=True)
            return r
        return tensor(data, dtype=dtype, device=device)
    g.as_tensor = frontend_factory(as_tensor, Var)

    def from_numpy(arr, *, device=None):
        v = _array_keep_dtype(arr)
        if g is not ctx.native_backend:
            v.requires_grad_(False)
        if _device_is_cpu(device):
            return _make_cpu_resident(v)
        if _device_is_cuda(device):
            _set_use_cuda()
            return _make_cuda_resident(v, force=True)
        return v
    g.from_numpy = frontend_factory(from_numpy, Var)

    def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False):
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
                raise TypeError(f"torch.frombuffer unsupported dtype: {dtype}")
            arr = _np.frombuffer(buffer, dtype=np_dtype, count=count, offset=offset)
            v = from_numpy(_np.ascontiguousarray(arr))
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        return v
    g.frombuffer = frontend_factory(frombuffer, Var)

    Size = _TorchSize
    g.Size = Size

    g.broadcast_shapes = broadcast_shapes

    g.corrcoef = corrcoef

    # torch.Generator (RNG handle) -- jittor uses a global seed; provide a
    # lightweight stand-in that supports manual_seed and is accepted where a
    # generator is passed (it is otherwise ignored).
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
    class layout:  # torch.layout placeholder
        pass
    g.layout = layout
    class memory_format:
        pass
    g.memory_format = memory_format

    import types as _types_nested
    nested_mod = _types_nested.ModuleType("torch.nested")
    nested_mod.__path__ = []
    def _nested_from_tensors(tensors, *a, layout=None, **k):
        return _NestedTensor.from_tensors(tensors, ragged_idx=k.pop("ragged_idx", 1))
    def _nested_from_jagged(values, offsets, *a, **k):
        return _NestedTensor.from_jagged(values, offsets, ragged_idx=k.pop("ragged_idx", None))
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
    def _check_to_pybool(cond):
        if hasattr(cond, "all") and not isinstance(cond, (bool, int, float)):
            try:
                return bool(cond.all().item())
            except EXPECTED as exc:
                swallowed("torch/installers/tensor.py _check_to_pybool: return bool(cond.all().item())", exc)
                return bool(cond)
        return bool(cond)
    def _check_with(_exc):
        def _chk(cond, message=None):
            if not _check_to_pybool(cond):
                msg = message() if callable(message) else message
                raise _exc(msg if msg is not None else "Expected cond to be True, but got False")
        return _chk
    g._check = _check_with(RuntimeError)
    g._check_is_size = lambda i, message=None, **k: g._check(int(i) >= 0, message)
    g._check_index = _check_with(IndexError)
    g._check_value = _check_with(ValueError)
    g._check_type = _check_with(TypeError)
    g._check_not_implemented = _check_with(NotImplementedError)
    g._check_tensor_all = _check_with(RuntimeError)
    g._assert_async = lambda t, *a, **k: g._check(_check_to_pybool(t), "torch._assert_async failed")

    # torch.cat: tolerate empty tensors (skip zero-numel inputs) like torch,
    # accept `dim=`/`out=`. jittor's concat trips on an empty leading tensor.
    _jt_concat = jt.concat
    def cat(tensors, dim=0, out=None, axis=None):
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
        in_dtypes = {str(t.dtype) for t in fixed}
        if len(in_dtypes) == 1:
            d = in_dtypes.pop()
            if str(out_var.dtype) != d:
                out_var = out_var.cast(d)
        return out_var
    g.cat = cat
    g.concat = cat
    g.concatenate = cat

    # torch.stack accepts a numpy-style `axis=` alias for `dim=` (and `out=`); jittor's
    # jt.stack is `stack(x, dim=0)` only, so trl's PPO advantage stacking
    # `torch.stack(advantages_reversed[::-1], axis=1)` dies on the unexpected kwarg.
    _jt_stack = jt.stack
    def stack(tensors, dim=0, *, axis=None, out=None):
        if axis is not None: dim = axis
        res = _jt_stack(list(tensors), dim)
        if out is not None:
            out.assign(res)
            return out
        return res
    g.stack = stack

    # Wrap tensor constructors to tolerate torch's device=/requires_grad=/
    # layout=/pin_memory= kwargs and torch dtype objects. jittor's versions
    # don't accept device=, which torch code passes everywhere.
    _wrap_constructors(g)
    _install_random_and_linspace(g)

    _install_reductions(g)

def install_methods(ctx):
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    _install_tensor_methods(g, Var, _DTYPE_OBJS)
    # torch's Tensor.size() returns a torch.Size (tuple subclass) when called with
    # no arg, and an int for size(dim); jittor's native size() returns a NanoVector,
    # which breaks torch idioms like `(n,) + data.size()[1:]` (mmdet's unmap()).
    _Size = getattr(g, "Size", tuple)
    def _torch_size(self, dim=None):
        return self.shape[dim] if dim is not None else _Size(self.shape)
    Var.size = _torch_size

    # jittor's core reshape/view reject a torch.Size (a tuple SUBCLASS) -> normalize
    # a single Size/tuple-subclass arg to a plain tuple so `x.reshape(other.size())`
    # / `x.view(t.size())` works (mmdet queryinst). Only intervene for that case to
    # keep the (very hot) reshape path otherwise untouched.
    _orig_reshape = Var.reshape
    _np_view_of = None
    def _dtype_itemsize_name(ds):
        d = dtype._registry.get(ds)
        if d is not None:
            return d.itemsize
        return dtype(ds).itemsize
    def _bitcast(self, dt):
        import numpy as _np
        nonlocal _np_view_of
        if _np_view_of is None:
            _np_view_of = {"bool": _np.bool_, "uint8": _np.uint8, "int8": _np.int8, "uint16": _np.uint16,
                           "int16": _np.int16, "int32": _np.int32, "int64": _np.int64,
                           "float16": _np.float16, "bfloat16": _np.uint16,
                           "float32": _np.float32, "float64": _np.float64}
        ds = getattr(dt, "name", str(dt)).replace("torch.", "")
        itemsize = getattr(dt, "itemsize", None)
        itemsize = itemsize if isinstance(itemsize, int) else _dtype_itemsize_name(ds)
        old_itemsize = getattr(getattr(self, "dtype", None), "itemsize", None)
        if old_itemsize is None:
            old_itemsize = _dtype_itemsize_name(str(self.dtype))
        shape = list(self.shape)
        if len(shape) == 0:
            if old_itemsize != itemsize:
                raise RuntimeError("view(dtype) cannot change itemsize on a scalar tensor")
        else:
            last_bytes = int(shape[-1]) * int(old_itemsize)
            if itemsize <= 0 or last_bytes % int(itemsize) != 0:
                raise RuntimeError("view(dtype) requires the last dimension to be byte-compatible")
            shape[-1] = last_bytes // int(itemsize)
        reinterpret_view = getattr(jt, "reinterpret_view", None)
        npd = _np_view_of.get(ds, _np.uint8)
        if reinterpret_view is not None and ds in _np_view_of:
            return reinterpret_view(self, shape, ds)
        return jt.array(_np.ascontiguousarray(self.numpy()).view(npd))
    def _torch_reshape(self, *shape, **_kw):
        # torch's `.view(dtype)` / `.view(dtype=...)` REINTERPRETS the bytes as
        # another dtype (bitcast), e.g. weight.view(torch.uint8) for byte-packing
        # in vLLM weight transfer. jittor has no dtype-view; bitcast via numpy.
        # (NB: 'dtype' the kwarg must not shadow the `dtype` class used below.)
        _dt = _kw.get("dtype", None)
        if _dt is not None:
            return _bitcast(self, _dt)
        if not shape:
            # torch spells the target shape as a keyword too: `reshape(shape=...)`
            # (diffusers' DiT unpatchify) and `view(size=...)`. Dropping it left
            # an empty positional tuple and a "shape can't be empty" core error.
            _named = _kw.get("shape", _kw.get("size", None))
            if _named is not None:
                shape = _named if isinstance(_named, (tuple, list)) else (_named,)
                shape = (tuple(int(s) for s in shape),)
        if len(shape) == 1 and isinstance(shape[0], dtype):
            return _bitcast(self, shape[0])
        if len(shape) == 1 and isinstance(shape[0], tuple) and type(shape[0]) is not tuple:
            shape = (tuple(int(s) for s in shape[0]),)
        return _orig_reshape(self, *shape)
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
    def _norm_reduce_kw(a, k):
        d = None
        if "axis" in k:
            d = k.pop("axis")
        if "dim" in k:
            d = k.pop("dim")
        if "dims" in k:
            d = k.pop("dims")
        if d is None and len(a) >= 1:
            if isinstance(a[0], (tuple, list)):
                d = a[0]; a = a[1:]            # consume positional tuple-of-dims
            elif isinstance(a[0], (int, np.integer)) and not isinstance(a[0], bool):
                d = a[0]; a = a[1:]            # consume positional scalar dim
        # torch spells it keepdim; jittor's tuple overload spells it keepdims.
        keep = k.pop("keepdim", k.pop("keepdims", None))
        if keep is None and d is not None and len(a) >= 1 and isinstance(a[0], bool):
            keep = a[0]; a = a[1:]             # consume positional keepdim
        if d is not None:
            # jittor's scalar `dim` overload rejects keepdims, while its tuple
            # `dims` overload supports it -> always route through `dims` when a
            # keepdim was requested (wrap a scalar dim into a 1-tuple).
            if isinstance(d, (tuple, list)):
                k["dims"] = tuple(int(x) for x in d)
            elif keep is not None:
                k["dims"] = (int(d),)
            else:
                k["dim"] = int(d)
        if keep is not None:
            k["keepdims"] = bool(keep)
        return a, k

    def _looks_like_dtype(x):
        return isinstance(x, dtype) or (isinstance(x, str) and x.replace("torch.", "") in dtype._registry)

    _orig_var_sum = Var.sum
    _orig_module_sum = getattr(g, "sum", None)
    def _torch_var_sum(self, *a, **k):
        out = k.pop("out", None)
        dt = k.pop("dtype", None)
        a, k = _norm_reduce_kw(a, k)
        if dt is None and len(a) >= 1 and _looks_like_dtype(a[0]):
            dt = a[0]
            a = a[1:]
        if dt is not None:
            self = self.cast(_dtype_to_str(dt))
        elif str(self.dtype) in ("uint8", "int8", "uint16"):
            self = self.int32()
        result = _orig_var_sum(self, *a, **k)
        if out is not None:
            out.assign(result)
            return out
        return result
    Var.sum = _torch_var_sum
    if _orig_module_sum is not None:
        def _torch_sum(input, *a, **k):
            if isinstance(input, Var):
                return _torch_var_sum(input, *a, **k)
            return _orig_module_sum(input, *a, **k)
        g.sum = _torch_sum
    # Full dim/dims/keepdim normalization for the plain reductions that map onto
    # jittor's scalar-`dim` / tuple-`dims` overload pair (mean/prod/any/all). mmdet
    # exercises tuple dims here, e.g. yolact_head's loss.mean(dim=(1, 2)).
    def _reduce_wrap(orig):
        def _w(self, *a, **k):
            a, k = _norm_reduce_kw(a, k)
            return orig(self, *a, **k)
        return _w
    for _rn in ("mean", "prod"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            setattr(Var, _rn, _reduce_wrap(_ro))
    # any/all: jittor's only accept a scalar `dim` (no `dims` tuple, no keepdims).
    # Support torch's tuple-of-dims and keepdim by reducing one dim at a time
    # (descending so earlier dim indices stay valid), keeping a length-1 axis when
    # keepdim is set. Plain scalar/axis use falls through to the native op.
    def _anyall_wrap(orig, name):
        def _w(self, *a, **k):
            d = None
            if "axis" in k: d = k.pop("axis")
            if "dim" in k:  d = k.pop("dim")
            if "dims" in k: d = k.pop("dims")
            if d is None and len(a) >= 1 and isinstance(a[0], (tuple, list)):
                d = a[0]; a = a[1:]
            keep = k.pop("keepdim", k.pop("keepdims", None))
            if d is None:
                return orig(self, *a, **k)
            dims = [int(x) for x in d] if isinstance(d, (tuple, list)) else [int(d)]
            ndim = self.ndim
            dims = sorted((x % ndim for x in dims), reverse=True)
            out = self
            for ax in dims:
                out = orig(out, dim=ax)
                if keep:
                    out = out.unsqueeze(ax)
            return out
        return _w
    for _rn in ("any", "all"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            setattr(Var, _rn, _anyall_wrap(_ro, _rn))
    # max/min/argmax/argmin/amax/amin/cumsum/norm/std/var are already wrapped above
    # with custom torch-return semantics (value+index tuples, etc.); only translate
    # torch's `axis` alias for them so we don't disturb that handling.
    def _axis_to_dim(orig):
        if getattr(orig, "_torch_accepts_axis", False):
            return orig
        def _w(self, *a, **k):
            if "axis" in k:
                k["dim"] = k.pop("axis")
            return orig(self, *a, **k)
        return _w
    for _rn in ("max", "min", "argmax", "argmin", "amax", "amin", "cumsum",
                "norm", "std", "var"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            setattr(Var, _rn, _axis_to_dim(_ro))

    # ---- Tensor methods used by mmdetection + cheap torch-standard completeness ----
    # (.relu 86x, .eq 11x, .gt 12x, .diff, .fliplr are exercised by mmdet; the rest
    #  are one-line torch standards added to reduce downstream surprises.)
    if not hasattr(Var, "relu"):        Var.relu = lambda self: nn.relu(self)
    if not hasattr(Var, "relu_"):       Var.relu_ = lambda self: nn.relu(self)
    if not hasattr(Var, "eq"):          Var.eq = lambda self, other: self == other
    if not hasattr(Var, "ne"):          Var.ne = lambda self, other: self != other
    if not hasattr(Var, "gt"):          Var.gt = lambda self, other: self > other
    if not hasattr(Var, "ge"):          Var.ge = lambda self, other: self >= other
    if not hasattr(Var, "lt"):          Var.lt = lambda self, other: self < other
    if not hasattr(Var, "le"):          Var.le = lambda self, other: self <= other
    if not hasattr(Var, "neg"):         Var.neg = lambda self: -self
    if not hasattr(Var, "reciprocal"):  Var.reciprocal = lambda self: 1.0 / self
    if not hasattr(Var, "expm1"):       Var.expm1 = lambda self: jt.exp(self) - 1
    if not hasattr(Var, "log1p"):       Var.log1p = lambda self: jt.log(self + 1)
    if not hasattr(Var, "square"):      Var.square = lambda self: self * self
    if not hasattr(Var, "square_"):     Var.square_ = lambda self: self.assign(self * self)
    if not hasattr(Var, "clamp_min"):   Var.clamp_min = lambda self, v: jt.maximum(self, v)
    if not hasattr(Var, "clamp_max"):   Var.clamp_max = lambda self, v: jt.minimum(self, v)
    _orig_index_add_inplace = getattr(Var, "index_add_", None)
    if _orig_index_add_inplace is not None and not getattr(_orig_index_add_inplace, "_torch_returns_self", False):
        def _index_add_inplace(self, dim, index, source, *, alpha=1):
            if alpha != 1:
                source = source * alpha
            _orig_index_add_inplace(self, dim, index, source)
            return self
        _index_add_inplace._torch_returns_self = True
        Var.index_add_ = _index_add_inplace
    if not hasattr(Var, "bmm"):         Var.bmm = lambda self, other: jt.matmul(self, other)
    if not hasattr(Var, "mm"):          Var.mm = lambda self, other: jt.matmul(self, other)
    if not hasattr(Var, "mv"):          Var.mv = lambda self, vec: g.mv(self, vec)
    if not hasattr(Var, "fliplr"):      Var.fliplr = lambda self: jt.flip(self, 1)
    if not hasattr(Var, "flipud"):      Var.flipud = lambda self: jt.flip(self, 0)
    if not hasattr(Var, "diff"):
        Var.diff = lambda self, n=1, dim=-1, prepend=None, append=None: _diff(self, n, dim, prepend, append)
    if not hasattr(Var, "trapz"):
        Var.trapz = lambda self, x=None, dx=1, dim=-1: _trapz(self, x=x, dx=dx, dim=dim)
    if not hasattr(Var, "trapezoid"):
        Var.trapezoid = lambda self, x=None, dx=1, dim=-1: _trapz(self, x=x, dx=dx, dim=dim)
    if not hasattr(Var, "fmod"):        # truncated remainder, sign of dividend
        Var.fmod = lambda self, other: self - jt.trunc(self / other) * other
    if not hasattr(Var, "remainder"):   # floored remainder, sign of divisor
        Var.remainder = lambda self, other: self - jt.floor(self / other) * other
    if not hasattr(Var, "softplus"):    Var.softplus = lambda self, beta=1, threshold=20: nn.softplus(self)
