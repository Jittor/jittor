""" torch_compat: make `import jittor as torch` behave like PyTorch.

Augments the top-level ``jittor`` namespace with the names and small semantic
shims that PyTorch code (and libraries such as transformers) expect, so that
``import jittor as torch`` can run unmodified torch code.

Imported at the end of ``jittor/__init__.py``. Everything here is additive: it
never removes or changes existing jittor behaviour, only fills gaps and adds
aliases.
"""
import jittor as jt
from jittor import nn
from collections.abc import Mapping
import numbers
import numpy as np


from ._torch_compat.functional import (
    _diff,
    _isin,
    _repeat_interleave,
    _torch_norm_impl,
    _torch_where_select,
    _trapz,
)
from ._torch_compat.grad import (
    _amp_passthrough_decorator,
    _AutocastContext,
    _clip_grad_norm_device,
    _GradDecoratorCtx,
    _GradScaler,
)
from ._torch_compat.lr_scheduler import _install_lr_scheduler
from ._torch_compat.nested import (
    _NestedTensor,
    _rebuild_nested_tensor,
    _rebuild_var_from_numpy,
    _TorchSize,
    _torch_make_parameter,
    _torch_prune_leaf_registry,
    _torch_register_leaf,
)
from ._torch_compat.optimizers import _install_optimizers
from ._torch_compat.runtime import (
    bind_runtime as _bind_compat_runtime,
    preserve_facade_origins as _preserve_facade_origins,
)
from ._torch_compat.serialization import _install_safetensors_shim
from ._torch_compat.types import (
    _DEVICE_CTX_STACK,
    _device_is_cpu,
    _device_is_cuda,
    _dtype_to_str,
    _make_cpu_resident,
    _make_cuda_resident,
    _make_dtypes,
    _mark_cpu_like,
    _var_has_cpu_residency_hint,
    _var_is_cpu_resident,
    device,
    dtype,
)

_bind_compat_runtime(jt)
del _bind_compat_runtime

# Keep class/function identities and existing pickle paths stable while the
# implementation lives in focused private modules.
_COMPAT_FACADE_SYMBOLS = (
    _diff, _isin, _repeat_interleave, _torch_norm_impl,
    _torch_where_select, _trapz,
    _amp_passthrough_decorator, _AutocastContext,
    _clip_grad_norm_device, _GradDecoratorCtx, _GradScaler,
    _install_lr_scheduler,
    _NestedTensor, _rebuild_nested_tensor, _rebuild_var_from_numpy,
    _TorchSize, _torch_make_parameter, _torch_prune_leaf_registry,
    _torch_register_leaf,
    _install_optimizers, _install_safetensors_shim,
    _device_is_cpu, _device_is_cuda, _dtype_to_str,
    _make_cpu_resident, _make_cuda_resident, _make_dtypes,
    _mark_cpu_like, _var_has_cpu_residency_hint, _var_is_cpu_resident,
    device, dtype,
)
_preserve_facade_origins(_COMPAT_FACADE_SYMBOLS)
del _preserve_facade_origins


def install(torch):
    g = torch
    import sys as _sys_install
    if _sys_install.modules.get("torch") is None:
        _sys_install.modules["torch"] = g
    g.torch = g
    _sys_install.modules.setdefault("torch.torch", g)
    g._torch_make_parameter = _torch_make_parameter
    g._torch_prune_leaf_registry = _torch_prune_leaf_registry
    if not hasattr(g, "_vj_native_load"):
        g._vj_native_load = getattr(g, "load", None)
    if not hasattr(g, "_vj_native_save"):
        g._vj_native_save = getattr(g, "save", None)
    if not hasattr(g, "_vj_native_where"):
        g._vj_native_where = getattr(g, "where", None)
    if not hasattr(g, "_vj_native_nonzero"):
        g._vj_native_nonzero = getattr(g, "nonzero", None)

    # Pillow 11 rejects int8 RGB arrays. Some legacy torch projects, including
    # graphdeco gaussian-splatting, use np.byte as a uint8 alias before
    # Image.fromarray(..., "RGB"). PyTorch/torchvision environments historically
    # tolerated that path, so reinterpret int8 image buffers as uint8.
    try:
        from PIL import Image as _PILImage
        _pil_fromarray = _PILImage.fromarray
        if not getattr(_pil_fromarray, "_jittor_torch_compat", False):
            def _fromarray_compat(obj, mode=None, *args, **kwargs):
                if mode in ("RGB", "RGBA", "L") and getattr(obj, "dtype", None) is not None:
                    try:
                        import numpy as _np
                        if obj.dtype == _np.int8:
                            obj = obj.view(_np.uint8)
                    except Exception:
                        pass
                return _pil_fromarray(obj, mode=mode, *args, **kwargs)
            _fromarray_compat._jittor_torch_compat = True
            _PILImage.fromarray = _fromarray_compat
    except Exception:
        pass

    # Critical: jittor dispatches every op to CPU unless flags.use_cuda is set.
    # The accelerator (Ascend NPU via jt.compiler.has_acl, or NVIDIA GPU via
    # jt.has_cuda) is present, but use_cuda defaults to 0 -- so `import torch` +
    # model.to("cuda") (a no-op here) would silently run the ENTIRE model on CPU,
    # ~10000x slower (a 2048^3 matmul: 20s CPU vs 2ms NPU). Enable device dispatch
    # globally whenever an accelerator exists, so tensors/ops land on it by default,
    # matching what torch users expect from .cuda()/.to(device).
    try:
        # Don't force CUDA when NO device is visible at runtime: an explicit
        # empty CUDA_VISIBLE_DEVICES (e.g. a CPU-only Ray orchestrator actor,
        # num_gpus=0) means no GPU -- forcing use_cuda=1 then crashes on the
        # first CUDA op (cudaErrorNoDevice). Unset/non-empty => devices present.
        import os as _os
        _cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
        _no_gpu = _cvd is not None and _cvd.strip() == ""
        if (getattr(jt.compiler, "has_acl", 0) or getattr(jt, "has_cuda", 0)) and not _no_gpu:
            jt.flags.use_cuda = 1
    except Exception:
        pass
    _DTYPE_OBJS = _make_dtypes(g)
    g.dtype = dtype
    g.device = device
    g.GradScaler = _GradScaler        # picked up by torch.amp/torch.cuda.amp in the shim
    try:
        import jittor.nn as _jt_nn_top
        for _conv_name in ("conv1d", "conv2d", "conv3d",
                           "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
            if not hasattr(g, _conv_name) and hasattr(_jt_nn_top, _conv_name):
                setattr(g, _conv_name, getattr(_jt_nn_top, _conv_name))
    except Exception:
        pass

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
                except Exception:
                    pass
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

    Var = jt.Var
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
    g.Tensor = Tensor
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
            jt.flags.use_cuda = 1
            v = _make_cuda_resident(v, force=True)
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        return v
    g.tensor = tensor

    def as_tensor(data, dtype=None, device=None):
        if isinstance(data, Var):
            r = data if dtype is None else data.cast(_dtype_to_str(dtype))
            if _device_is_cpu(device):
                return _make_cpu_resident(r)
            if _device_is_cuda(device):
                jt.flags.use_cuda = 1
                return _make_cuda_resident(r, force=True)
            return r
        return tensor(data, dtype=dtype, device=device)
    g.as_tensor = as_tensor

    def from_numpy(arr, *, device=None):
        v = _array_keep_dtype(arr)
        if _device_is_cpu(device):
            return _make_cpu_resident(v)
        if _device_is_cuda(device):
            jt.flags.use_cuda = 1
            return _make_cuda_resident(v, force=True)
        return v
    g.from_numpy = from_numpy

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
    g.frombuffer = frombuffer

    Size = _TorchSize
    g.Size = Size

    # torch.broadcast_shapes(*shapes) -> Size : broadcasted shape of the inputs
    # (used by verl's advantage/reward broadcasting). numpy implements the same rule.
    def broadcast_shapes(*shapes):
        import numpy as _npb
        norm = [(int(s),) if isinstance(s, (int, np.integer)) else tuple(int(d) for d in s) for s in shapes]
        return Size(_npb.broadcast_shapes(*norm)) if norm else Size(())
    g.broadcast_shapes = broadcast_shapes

    # torch.corrcoef(input) -> correlation-coefficient matrix (verl logs the
    # rollout-vs-recompute logprob correlation as a diagnostic). numpy matches.
    def corrcoef(x, *a, **k):
        import numpy as _npc
        r = _npc.corrcoef(x.float32().numpy())
        return jt.array(_npc.ascontiguousarray(r))
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
    import sys as _sys_nested
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
    _sys_nested.modules["torch.nested"] = nested_mod
    nested_internal_mod = _types_nested.ModuleType("torch.nested._internal")
    nested_internal_mod.__path__ = []
    nested_tensor_mod = _types_nested.ModuleType("torch.nested._internal.nested_tensor")
    nested_tensor_mod.NestedTensor = _NestedTensor
    nested_internal_mod.nested_tensor = nested_tensor_mod
    nested_mod._internal = nested_internal_mod
    _sys_nested.modules["torch.nested._internal"] = nested_internal_mod
    _sys_nested.modules["torch.nested._internal.nested_tensor"] = nested_tensor_mod

    # torch._check family: assertion helpers used by dynamo / TorchScript-friendly
    # code (e.g. vLLM's sampler does `torch._check(x.shape[0] >= 1)`). The message
    # may be a zero-arg callable that torch invokes lazily only on failure. The
    # condition is usually a python bool but can be a bool tensor (_check_tensor_all).
    def _check_to_pybool(cond):
        if hasattr(cond, "all") and not isinstance(cond, (bool, int, float)):
            try:
                return bool(cond.all().item())
            except Exception:
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
        min_nd = min(t.ndim for t in nonempty)
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

    if not hasattr(nn, "functional"):
        import types as _types
        F = _types.ModuleType("jittor.nn.functional")
        for fname in dir(nn):
            fobj = getattr(nn, fname)
            if callable(fobj) and not isinstance(fobj, type):
                setattr(F, fname, fobj)
        if hasattr(nn, "relu"): F.relu = nn.relu
        if hasattr(nn, "gelu"): F.gelu = nn.gelu
        if hasattr(nn, "softmax"):
            # torch: F.softmax(input, dim=None, _stacklevel=3, dtype=None).
            # When dtype is given, input is cast to it before softmax (used by
            # transformers' eager attention: F.softmax(scores, dim=-1, dtype=fp32)).
            _jt_softmax = nn.softmax
            def _softmax(input, dim=-1, _stacklevel=3, dtype=None):
                if dtype is not None:
                    input = input.cast(_dtype_to_str(dtype))
                return _jt_softmax(input, dim=dim)
            F.softmax = _softmax
        if hasattr(nn, "linear"): F.linear = nn.linear
        if hasattr(nn, "interpolate"):
            # torch.nn.functional.interpolate defaults to mode='nearest', but
            # jittor.nn.interpolate defaults to 'bilinear'. Code that omits the
            # mode (e.g. YOLOV3Neck: F.interpolate(x, scale_factor=2)) silently
            # gets the wrong upsampling. Wrap so the torch-shim functional matches
            # torch's default and accepts torch's arg name / extra kwargs. Only
            # this shim copy is affected, not jittor's native nn.interpolate.
            _jt_interpolate = nn.interpolate
            def _interpolate(input=None, size=None, scale_factor=None,
                             mode="nearest", align_corners=None,
                             recompute_scale_factor=None, antialias=False,
                             **_kw):
                if input is None:
                    input = _kw.pop("X")
                ac = False if align_corners is None else align_corners
                return _jt_interpolate(input, size=size,
                                       scale_factor=scale_factor, mode=mode,
                                       align_corners=ac)
            F.interpolate = _interpolate
        if hasattr(nn, "cross_entropy_loss"):
            _jt_ce = nn.cross_entropy_loss
            # torch.nn.functional.cross_entropy(..., label_smoothing=): jittor's
            # cross_entropy_loss has no label_smoothing (used by many training recipes:
            # ImageNet, translation, some SFT). Delegate to jittor for ls=0 (verified
            # correct incl. weight/ignore_index); implement smoothing to match torch:
            #   loss_i = (1-ls)*nll_i + (ls/C)*smooth_i,  nll_i = -w[t]*logp[i,t],
            #   smooth_i = -sum_c(w_c*logp[i,c]);  mean divides by sum(w[t]) (or count).
            def _cross_entropy(input, target, weight=None, size_average=None,
                               ignore_index=-100, reduce=None, reduction="mean",
                               label_smoothing=0.0):
                # torch: a floating-point target with the SAME shape as input is a
                # class-probability ("soft label") target (mixup / distillation / soft
                # label-smoothing). jittor's cross_entropy_loss only understands integer
                # class-index targets, so handle the soft case here.
                if (isinstance(target, jt.Var) and target.ndim == input.ndim
                        and "int" not in str(target.dtype)):
                    Cc = int(input.shape[1]) if input.ndim >= 2 else int(input.shape[-1])
                    cdim = 1 if input.ndim >= 2 else -1
                    logp = nn.log_softmax(input, dim=cdim)
                    tgt = target
                    if label_smoothing:
                        tgt = (1.0 - label_smoothing) * tgt + label_smoothing / Cc
                    if weight is not None:
                        wsh = [1] * input.ndim; wsh[cdim] = Cc
                        wloss = -(tgt * logp * weight.reshape(wsh)).sum(dim=cdim)
                    else:
                        wloss = -(tgt * logp).sum(dim=cdim)
                    if reduction == "sum":
                        return wloss.sum()
                    if reduction == "none":
                        return wloss
                    return wloss.mean()        # torch divides the soft-target loss by N
                if not label_smoothing:
                    ii = -100 if ignore_index is None else ignore_index
                    return _jt_ce(input, target, weight=weight, ignore_index=ii,
                                  reduction=reduction)
                C = int(input.shape[1]) if input.ndim >= 2 else int(input.shape[-1])
                if input.ndim > 2:                  # (N,C,d...) -> (M,C)
                    perm = [0] + list(range(2, input.ndim)) + [1]
                    x = input.transpose(perm).reshape((-1, C))
                else:
                    x = input
                t = target.reshape((-1,))
                logp = nn.log_softmax(x, dim=-1)
                ig = None if ignore_index is None else ignore_index
                t_safe = t if ig is None else jt.ternary(t == ig, jt.zeros_like(t), t)
                nll = -logp.gather(1, t_safe.reshape((-1, 1))).reshape((-1,))
                if weight is not None:
                    wt = weight[t_safe]
                    nll = nll * wt
                    smooth = -(logp * weight.reshape((1, -1))).sum(dim=-1)
                else:
                    wt = None
                    smooth = -logp.sum(dim=-1)
                loss = (1.0 - label_smoothing) * nll + (label_smoothing / C) * smooth
                if ig is not None:
                    keep = (t != ig).float32()
                    loss = loss * keep
                    norm = (wt * keep).sum() if wt is not None else keep.sum()
                else:
                    norm = wt.sum() if wt is not None else jt.array(float(t.shape[0]))
                if reduction == "sum":
                    return loss.sum()
                if reduction == "none":
                    return loss.reshape(target.shape) if input.ndim > 2 else loss
                return loss.sum() / norm
            F.cross_entropy = _cross_entropy
        # Loss functions jittor's functional lacks but real workloads use: kl_div
        # (knowledge distillation), binary_cross_entropy (non-logits), huber_loss,
        # cosine_embedding_loss, margin_ranking_loss, gaussian_nll_loss. Verified
        # bit-equal to real torch 2.12. Use maximum/minimum/ternary (not clamp) to avoid
        # the torch_compat clamp kwarg overloading.
        import math as _math_loss
        def _reduce(loss, reduction):
            if reduction == "none":
                return loss
            if reduction == "sum":
                return loss.sum()
            return loss.mean()
        if not hasattr(F, "kl_div"):
            def _kl_div(input, target, size_average=None, reduce=None,
                        reduction="mean", log_target=False):
                if log_target:
                    loss = jt.exp(target) * (target - input)
                else:
                    # target*(log target - input); target==0 contributes 0 (avoid 0*-inf)
                    safe = jt.maximum(target, 1e-12)
                    loss = target * (jt.log(safe) - input)
                if reduction == "batchmean":
                    return loss.sum() / input.shape[0]
                return _reduce(loss, reduction)
            F.kl_div = _kl_div
        if not hasattr(F, "binary_cross_entropy"):
            def _bce(input, target, weight=None, size_average=None, reduce=None,
                     reduction="mean"):
                # input are probabilities in [0,1]; torch clamps the logs to >= -100.
                li = jt.maximum(jt.log(jt.maximum(input, 1e-44)), -100.0)
                l1 = jt.maximum(jt.log(jt.maximum(1.0 - input, 1e-44)), -100.0)
                loss = -(target * li + (1.0 - target) * l1)
                if weight is not None:
                    loss = loss * weight
                return _reduce(loss, reduction)
            F.binary_cross_entropy = _bce
        if not hasattr(F, "huber_loss"):
            def _huber(input, target, reduction="mean", delta=1.0):
                d = (input - target).abs()
                loss = jt.ternary(d < delta, 0.5 * d * d, delta * (d - 0.5 * delta))
                return _reduce(loss, reduction)
            F.huber_loss = _huber
        if not hasattr(F, "margin_ranking_loss"):
            def _margin_ranking(input1, input2, target, margin=0.0,
                                size_average=None, reduce=None, reduction="mean"):
                loss = jt.maximum(-target * (input1 - input2) + margin, 0.0)
                return _reduce(loss, reduction)
            F.margin_ranking_loss = _margin_ranking
        if not hasattr(F, "cosine_embedding_loss"):
            def _cosine_embedding(input1, input2, target, margin=0.0,
                                  size_average=None, reduce=None, reduction="mean"):
                cos = F.cosine_similarity(input1, input2)
                loss = jt.ternary(target == 1, 1.0 - cos, jt.maximum(cos - margin, 0.0))
                return _reduce(loss, reduction)
            F.cosine_embedding_loss = _cosine_embedding
        if not hasattr(F, "gaussian_nll_loss"):
            def _gaussian_nll(input, target, var, full=False, eps=1e-6, reduction="mean"):
                v = jt.maximum(var, eps)
                loss = 0.5 * (jt.log(v) + (input - target) ** 2 / v)
                if full:
                    loss = loss + 0.5 * _math_loss.log(2 * _math_loss.pi)
                return _reduce(loss, reduction)
            F.gaussian_nll_loss = _gaussian_nll
        # nn.*Loss class versions (criterion = nn.HuberLoss()): thin wrappers over the
        # functional. KLDivLoss/BCELoss/BCEWithLogitsLoss/CrossEntropyLoss/MSELoss/L1Loss
        # already exist on jittor.nn (verified correct); add the rest.
        _Mod = nn.Module
        def _add_loss_class(cname, fn, defaults, arg_order):
            if hasattr(nn, cname):
                return
            class _L(_Mod):
                def __init__(self, *a, **k):
                    super().__init__()
                    self._kw = dict(defaults); self._kw.update(k)
                    for nm, val in zip(arg_order, a):
                        self._kw[nm] = val
                def execute(self, *inputs):
                    return fn(*inputs, **self._kw)
            _L.__name__ = cname
            setattr(nn, cname, _L)
        _add_loss_class("HuberLoss", F.huber_loss, dict(reduction="mean", delta=1.0), ("reduction", "delta"))
        _add_loss_class("SmoothL1Loss", F.smooth_l1_loss, dict(reduction="mean"), ("reduction",))
        _add_loss_class("MarginRankingLoss", F.margin_ranking_loss, dict(margin=0.0, reduction="mean"), ("margin", "reduction"))
        _add_loss_class("CosineEmbeddingLoss", F.cosine_embedding_loss, dict(margin=0.0, reduction="mean"), ("margin", "reduction"))
        _add_loss_class("GaussianNLLLoss", F.gaussian_nll_loss, dict(full=False, eps=1e-6, reduction="mean"), ("full", "eps", "reduction"))
        _add_loss_class("NLLLoss", F.nll_loss, dict(reduction="mean"), ("weight", "size_average", "ignore_index"))
        # pixel_shuffle / pixel_unshuffle (super-resolution, some VAE decoders): jittor's
        # functional lacks them. (N, C*r^2, H, W) <-> (N, C, H*r, W*r). Verified vs torch.
        if not hasattr(F, "pixel_shuffle"):
            def _pixel_shuffle(input, upscale_factor):
                r = upscale_factor
                N, Cr2, H, W = input.shape
                C = Cr2 // (r * r)
                return input.reshape((N, C, r, r, H, W)).permute(0, 1, 4, 2, 5, 3).reshape((N, C, H * r, W * r))
            F.pixel_shuffle = _pixel_shuffle
            g.pixel_shuffle = _pixel_shuffle
        if not hasattr(F, "pixel_unshuffle"):
            def _pixel_unshuffle(input, downscale_factor):
                r = downscale_factor
                N, C, H, W = input.shape
                return input.reshape((N, C, H // r, r, W // r, r)).permute(0, 1, 3, 5, 2, 4).reshape((N, C * r * r, H // r, W // r))
            F.pixel_unshuffle = _pixel_unshuffle
            g.pixel_unshuffle = _pixel_unshuffle
        for _pscn, _psfn in (("PixelShuffle", "pixel_shuffle"), ("PixelUnshuffle", "pixel_unshuffle")):
            if not hasattr(nn, _pscn):
                def _mk(fn):
                    class _PS(nn.Module):
                        def __init__(self, factor): super().__init__(); self._f = factor
                        def execute(self, x): return getattr(F, fn)(x, self._f)
                    return _PS
                _cls = _mk(_psfn); _cls.__name__ = _pscn; setattr(nn, _pscn, _cls)
        # F.logsigmoid (DPO/preference losses), F.gumbel_softmax (discrete/MoE sampling).
        if not hasattr(F, "logsigmoid"):
            # stable: log(sigmoid(x)) = min(x,0) - log(1+exp(-|x|))
            F.logsigmoid = lambda input: jt.minimum(input, 0.0) - jt.log(1.0 + jt.exp(-jt.abs(input)))
        if not hasattr(F, "gumbel_softmax"):
            def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
                u = jt.rand(logits.shape)
                g = -jt.log(-jt.log(u + eps) + eps)             # Gumbel(0,1) noise
                y = nn.softmax((logits + g) / tau, dim=dim)
                if hard:
                    m = y.max(dim, keepdims=True)
                    y_hard = (y >= m).float32()
                    y = (y_hard - y).stop_grad() + y            # straight-through estimator
                return y
            F.gumbel_softmax = _gumbel_softmax
        if not hasattr(F, "rms_norm"):
            # F.rms_norm (torch 2.4+): x / sqrt(mean(x^2, over last len(normalized_shape)
            # dims) + eps) * weight. The norm modern LLMs (Llama/Qwen/Gemma) use.
            def _rms_norm(input, normalized_shape, weight=None, eps=None):
                if eps is None:
                    eps = 1.1920929e-07                          # finfo(float32).eps, torch default
                ndn = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
                dims = list(range(input.ndim - ndn, input.ndim))
                out = input * (1.0 / jt.sqrt((input * input).mean(dims, keepdims=True) + eps))
                return out * weight if weight is not None else out
            F.rms_norm = _rms_norm
        # Activations / losses jittor's functional lacked (verified vs real torch 2.12).
        if not hasattr(F, "softmin"):
            F.softmin = lambda input, dim=-1, _stacklevel=3, dtype=None: nn.softmax(-input, dim=dim)
        if not hasattr(F, "tanhshrink"):
            F.tanhshrink = lambda input: input - jt.tanh(input)
        if not hasattr(F, "celu"):
            F.celu = lambda input, alpha=1.0, inplace=False: \
                jt.maximum(input, 0.0) + jt.minimum(0.0, alpha * (jt.exp(input / alpha) - 1))
        if not hasattr(F, "selu"):
            def _selu(input, inplace=False):
                a = 1.6732632423543772848170429916717
                s = 1.0507009873554804934193349852946
                return s * (jt.maximum(input, 0.0) + jt.minimum(0.0, a * (jt.exp(input) - 1)))
            F.selu = _selu
        if not hasattr(F, "threshold"):
            def _threshold(input, threshold, value, inplace=False):
                m = (input > threshold).float32()
                return m * input + (1 - m) * value
            F.threshold = _threshold
        if not hasattr(F, "triplet_margin_loss"):
            def _triplet(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6,
                         swap=False, size_average=None, reduce=None, reduction="mean"):
                def _d(a, b):
                    return ((jt.abs(a - b) ** p).sum(-1) + eps) ** (1.0 / p)
                dp, dn = _d(anchor, positive), _d(anchor, negative)
                if swap:
                    dn = jt.minimum(dn, _d(positive, negative))
                loss = jt.maximum(dp - dn + margin, 0.0)
                return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
            F.triplet_margin_loss = _triplet
        if not hasattr(F, "poisson_nll_loss"):
            def _poisson_nll(input, target, log_input=True, full=False, size_average=None,
                             eps=1e-8, reduce=None, reduction="mean"):
                loss = (jt.exp(input) - target * input) if log_input else (input - target * jt.log(input + eps))
                if full:
                    import math as _mp
                    stir = target * jt.log(jt.maximum(target, eps)) - target + 0.5 * jt.log(2 * _mp.pi * jt.maximum(target, eps))
                    loss = loss + jt.ternary(target > 1, stir, jt.zeros_like(target))
                return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
            F.poisson_nll_loss = _poisson_nll
        if not hasattr(F, "ctc_loss"):
            # F.ctc_loss (wav2vec2 / speech ASR): the CTC forward (alpha) DP in log space.
            # log_probs (T,N,C) log-softmax; targets (N,S) padded or 1-D concatenated.
            # Differentiable (grad flows to log_probs). Verified bit-equal to real torch.
            import numpy as _np_ctc
            _CNEG = -1e30
            def _ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
                          reduction="mean", zero_infinity=False):
                def _ints(v):
                    return [int(x) for x in (v.numpy().reshape(-1) if isinstance(v, jt.Var) else _np_ctc.asarray(v).reshape(-1))]
                in_lens, tgt_lens = _ints(input_lengths), _ints(target_lengths)
                tnp = targets.numpy() if isinstance(targets, jt.Var) else _np_ctc.asarray(targets)
                flat = (tnp.ndim == 1)
                def _shift(v, k):
                    return jt.concat([jt.full((k,), _CNEG), v[:int(v.shape[0]) - k]]) if k > 0 else v
                def _lse(mats):
                    m = mats[0]
                    for x in mats[1:]:
                        m = jt.maximum(m, x)
                    return m + jt.safe_log(sum(jt.exp(x - m) for x in mats))
                N = log_probs.shape[1]
                losses, offset = [], 0
                for n in range(N):
                    Tn, Sn = in_lens[n], tgt_lens[n]
                    if flat:
                        seq = [int(x) for x in tnp[offset:offset + Sn]]; offset += Sn
                    else:
                        seq = [int(x) for x in tnp[n, :Sn]]
                    ext = [blank]
                    for lab in seq:
                        ext += [lab, blank]
                    L = len(ext)
                    ext_idx = jt.array(_np_ctc.array(ext, dtype="int64"))
                    skip = _np_ctc.zeros(L, dtype="float32")
                    for s in range(2, L):
                        if ext[s] != blank and ext[s] != ext[s - 2]:
                            skip[s] = 1.0
                    skip_v = jt.array(skip)
                    start = _np_ctc.full(L, _CNEG, dtype="float32"); start[0] = 0.0
                    if L > 1:
                        start[1] = 0.0
                    lp_n = log_probs[:Tn, n, :]
                    alpha = lp_n[0][ext_idx] + jt.array(start)
                    for t in range(1, Tn):
                        a2 = _shift(alpha, 2) * skip_v + (1 - skip_v) * _CNEG
                        alpha = lp_n[t][ext_idx] + _lse([alpha, _shift(alpha, 1), a2])
                    losses.append(-(_lse([alpha[L - 1], alpha[L - 2]]) if L > 1 else alpha[L - 1]))
                out = jt.stack(losses).reshape((N,))   # (N,1)->(N,): jittor has no 0-d scalar
                if zero_infinity:
                    out = jt.ternary(jt.isfinite(out), out, jt.zeros_like(out))
                if reduction == "none":
                    return out
                if reduction == "sum":
                    return out.sum()
                tl = jt.array(_np_ctc.array([max(s, 1) for s in tgt_lens], dtype="float32"))
                return (out / tl).mean()
            F.ctc_loss = _ctc_loss
        if hasattr(nn, "layer_norm"): F.layer_norm = nn.layer_norm
        if hasattr(nn, "embedding"): F.embedding = nn.embedding
        nn.functional = F
    g.nn.functional = nn.functional
    if not hasattr(nn.functional, "cosine_similarity") and hasattr(nn, "cosine_similarity"):
        nn.functional.cosine_similarity = nn.cosine_similarity
    if not hasattr(nn.functional, "pairwise_distance") and hasattr(nn, "pairwise_distance"):
        nn.functional.pairwise_distance = nn.pairwise_distance

    import os as _os

    _sdpa_flash_backend_cache = {}

    def _sdpa_static_backend_cache_enabled():
        return (_os.environ.get("JITTOR_TORCH_INFERENCE") or "").strip().lower() \
            in ("1", "true", "yes", "on")

    def _sdpa_flash_stats():
        stats = getattr(jt, "_torch_sdpa_flash_stats", None)
        if stats is None:
            stats = {"hits": 0, "misses": {}, "casts": {}, "backend": None}
            jt._torch_sdpa_flash_stats = stats
        return stats

    def _sdpa_flash_miss(reason):
        misses = _sdpa_flash_stats()["misses"]
        misses[reason] = misses.get(reason, 0) + 1

    def _sdpa_flash_cast(reason):
        casts = _sdpa_flash_stats()["casts"]
        casts[reason] = casts.get(reason, 0) + 1

    def _sdpa_flash_hit(backend_name):
        stats = _sdpa_flash_stats()
        stats["hits"] += 1
        stats["backend"] = backend_name

    def _sdpa_flash_template_dim(dim):
        dim = int(dim)
        if dim <= 0 or dim > 256 or dim % 8 != 0:
            return None
        if dim <= 32:
            return 32
        if dim <= 64:
            return 64
        if dim <= 96:
            return 96
        if dim <= 128:
            return 128
        if dim <= 192:
            return 192
        return 256

    def _sdpa_flash_float32_cast_target():
        raw = (_os.environ.get("JITTOR_FLASH_ATTN_CAST_FLOAT32") or "").strip().lower()
        if raw in ("1", "true", "yes", "on", "fp16", "float16", "half"):
            return "float16"
        if raw in ("bf16", "bfloat16"):
            return "bfloat16"
        return None

    def _try_flash_scaled_dot_product_attention(query, key, value, attn_mask,
                                                dropout_p, is_causal, sf,
                                                enable_gqa=False):
        if attn_mask is not None:
            _sdpa_flash_miss("mask")
            return None
        if float(dropout_p or 0.0) != 0.0:
            _sdpa_flash_miss("dropout")
            return None
        if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
            _sdpa_flash_miss("not_cuda_no_grad")
            return None
        q_shape, k_shape, v_shape = tuple(query.shape), tuple(key.shape), tuple(value.shape)
        if len(q_shape) < 3 or len(q_shape) != len(k_shape) or len(q_shape) != len(v_shape):
            _sdpa_flash_miss("rank")
            return None
        if q_shape[:-3] != k_shape[:-3] or q_shape[:-3] != v_shape[:-3]:
            _sdpa_flash_miss("batch")
            return None
        query_heads = int(q_shape[-3])
        key_heads = int(k_shape[-3])
        value_heads = int(v_shape[-3])
        gqa_heads_ok = (key_heads > 0 and enable_gqa
                        and query_heads % key_heads == 0)
        if key_heads != value_heads or not (
                query_heads == key_heads or gqa_heads_ok):
            _sdpa_flash_miss("heads")
            return None
        if q_shape[-1] != k_shape[-1] or q_shape[-1] != v_shape[-1]:
            _sdpa_flash_miss("head_dim_mismatch")
            return None
        # For CLIP-style short self-attention, the two cuBLAS matmuls plus the
        # fused softmax are faster than materializing the three layout copies
        # required by the separate-QKV FlashAttention wrapper. Keep this
        # inference-only and narrowly shaped so decoding, GQA and training keep
        # their existing backend choice.
        short_square_math = (
            _sdpa_static_backend_cache_enabled()
            and not enable_gqa and not is_causal
            and len(q_shape) == 4 and 0 < int(q_shape[0]) <= 8
            and query_heads == key_heads == value_heads == 12
            and int(q_shape[-1]) == 64
            and int(q_shape[-2]) == int(k_shape[-2]) == int(v_shape[-2])
            and int(q_shape[-2]) <= 64
            and str(query.dtype) == str(key.dtype) == str(value.dtype)
            and str(query.dtype) == "float16")
        if short_square_math:
            _sdpa_flash_miss("short_square_math")
            return None
        template_dim = _sdpa_flash_template_dim(q_shape[-1])
        if template_dim is None:
            _sdpa_flash_miss("head_dim")
            return None
        q_dtype, k_dtype, v_dtype = str(query.dtype), str(key.dtype), str(value.dtype)
        original_dtype = q_dtype
        cast_back = False
        if not (q_dtype == k_dtype == v_dtype and q_dtype in ("float16", "bfloat16")):
            cast_target = _sdpa_flash_float32_cast_target()
            if cast_target is None or not (q_dtype == k_dtype == v_dtype == "float32"):
                _sdpa_flash_miss("dtype")
                return None
            query = query.to(cast_target)
            key = key.to(cast_target)
            value = value.to(cast_target)
            q_dtype = k_dtype = v_dtype = cast_target
            cast_back = True
            _sdpa_flash_cast("float32_to_%s" % cast_target)
        try:
            from jittor.torch_shim import flashattn_jittor as _fa_jittor
        except Exception:
            _sdpa_flash_miss("no_loader")
            return None
        cache_key = (template_dim, q_dtype)
        static_cache = _sdpa_static_backend_cache_enabled()
        token_fn = getattr(_fa_jittor, "backend_cache_token", None)
        backend_token = (token_fn() if static_cache and callable(token_fn)
                         else None)
        cached = (_sdpa_flash_backend_cache.get(cache_key)
                  if static_cache and backend_token is not None else None)
        if cached is not None and cached[0] == backend_token:
            backend, capability_miss = cached[1], None
        else:
            backend, capability_miss = _fa_jittor.load_backend_for(
                template_dim, q_dtype)
            publication_fn = getattr(
                _fa_jittor, "backend_publication_token", None)
            publication_token = (
                publication_fn(backend) if callable(publication_fn) else None)
            backend_token = (token_fn() if static_cache and callable(token_fn)
                             else None)
            if (static_cache and backend_token is not None
                    and publication_token == backend_token
                    and backend is not None and capability_miss is None):
                _sdpa_flash_backend_cache[cache_key] = (
                    backend_token,
                    backend,
                )
        if backend is None:
            if _fa_jittor.required():
                raise RuntimeError(
                    "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set, but native "
                    "flash-attn backend is unavailable: %s"
                    % (_fa_jittor.last_error() or "unknown error")
                )
            _sdpa_flash_miss("no_backend")
            return None
        if capability_miss is not None:
            if _fa_jittor.required():
                raise RuntimeError(
                    "native flash-attn backend could not expand for %s: %s"
                    % (capability_miss, _fa_jittor.last_error() or "unsupported capability")
                )
            _sdpa_flash_miss(capability_miss)
            return None
        # load_backend_for() already returned the capability-checked backend.
        # Calling through the public flash_attn stub would invoke the loader a
        # second time for every layer and rescan all backend environment keys.
        fn = getattr(backend, "flash_attn_func", None)
        if not callable(fn):
            if _fa_jittor.required():
                raise RuntimeError("flash_attn shim has no flash_attn_func")
            _sdpa_flash_miss("no_func")
            return None
        prefix = q_shape[:-3]
        p = len(prefix)
        batch = 1
        for size in prefix:
            batch *= int(size)
        heads, lq, head_dim = query_heads, int(q_shape[-2]), int(q_shape[-1])
        lk = int(k_shape[-2])
        q_axes = tuple(list(range(p)) + [p + 1, p, p + 2])
        # Native flash-attn is an external C++/CUDA extension. Crossing that
        # boundary with a lazy permute/reshape expression can leave the bridge
        # holding transient metadata; clone materializes a stable row-major
        # tensor while keeping the kernel path fused.
        q_dense = query.permute(*q_axes).reshape((batch, lq, heads, head_dim)).clone()
        k_dense = key.permute(*q_axes).reshape((batch, lk, key_heads, head_dim)).clone()
        v_dense = value.permute(*q_axes).reshape((batch, lk, value_heads, head_dim)).clone()
        try:
            out = fn(q_dense, k_dense, v_dense, 0.0, float(sf), bool(is_causal))
        except Exception:
            if _fa_jittor.required():
                raise
            _sdpa_flash_miss("call_failed")
            return None
        if out is None:
            if _fa_jittor.required():
                raise RuntimeError(
                    "native flash-attn backend returned no output while "
                    "JITTOR_FLASH_ATTN_JITTOR_REQUIRED is set"
                )
            _sdpa_flash_miss("returned_none")
            return None
        out = out.reshape(tuple(prefix) + (lq, heads, head_dim))
        out_axes = tuple(list(range(p)) + [p + 1, p, p + 2])
        _sdpa_flash_hit(_fa_jittor.backend_name())
        out = out.permute(*out_axes)
        if cast_back and str(out.dtype) != original_dtype:
            out = out.to(original_dtype)
        return out

    # scaled_dot_product_attention (torch F.sdpa) -- native flash-attn
    # inference fast path when available, otherwise standard math impl with
    # causal masking + attn_mask + GQA support (jittor has no native sdpa).
    if not hasattr(nn.functional, "scaled_dot_product_attention"):
        import math as _math
        def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=0.0, is_causal=False,
                                         scale=None, enable_gqa=False, **kw):
            # query: (..., Lq, E), key/value: (..., Lk, E)
            d = query.shape[-1]
            sf = (1.0 / _math.sqrt(d)) if scale is None else scale
            flash = _try_flash_scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, sf,
                enable_gqa=enable_gqa)
            if flash is not None:
                return flash
            # The native FlashAttention backend accepts grouped-query attention
            # directly. Expand K/V only for the math fallback, where matmul
            # requires the head counts to match.
            if enable_gqa:
                query_heads = int(query.shape[-3])
                key_heads = int(key.shape[-3])
                value_heads = int(value.shape[-3])
                if key_heads != query_heads:
                    if key_heads <= 0 or query_heads % key_heads != 0:
                        raise RuntimeError(
                            "key heads must divide query heads for GQA")
                    key = key.repeat_interleave(
                        query_heads // key_heads, dim=-3)
                if value_heads != query_heads:
                    if value_heads <= 0 or query_heads % value_heads != 0:
                        raise RuntimeError(
                            "value heads must divide query heads for GQA")
                    value = value.repeat_interleave(
                        query_heads // value_heads, dim=-3)
            q_dtype, k_dtype = str(query.dtype), str(key.dtype)
            if (jt.flags.use_cuda and jt.compile_extern.cublas_ops
                    and len(query.shape) >= 3 and len(query.shape) == len(key.shape)
                    and query.shape[:-2] == key.shape[:-2]
                    and q_dtype == k_dtype and "float" in q_dtype
                    and "complex" not in q_dtype and "complex" not in k_dtype):
                scores = jt.compile_extern.cublas_ops.cublas_batched_matmul(query, key, 0, 1) * sf
            else:
                scores = jt.matmul(query, key.transpose(-1, -2)) * sf
            mask_row_valid = None
            mask_softmax = None
            if is_causal:
                Lq, Lk = query.shape[-2], key.shape[-2]
                mask = jt.triu(jt.ones((Lq, Lk)), 1) * (-1e30)
                scores = scores + mask
            if attn_mask is not None:
                try:
                    import jittor.other.code_softmax as _code_softmax
                    if _code_softmax.can_softmax_v1(scores, -1):
                        mask_softmax = _code_softmax
                except Exception:
                    mask_softmax = None
                if str(attn_mask.dtype) == "bool":
                    if mask_softmax is not None:
                        zero_bias = jt.zeros_like(attn_mask, dtype=scores.dtype)
                        mask_bias = jt.ternary(
                            attn_mask, zero_bias,
                            zero_bias + float("-inf"))
                        scores = scores + mask_bias
                    else:
                        mask_row_valid = attn_mask.sum(-1, keepdims=True) > 0
                        scores = scores + (1 - attn_mask.float32()) * (-1e30)
                else:
                    scores = scores + attn_mask
                    if mask_softmax is None:
                        neg_inf = jt.isinf(attn_mask) & (attn_mask < 0)
                        mask_row_valid = neg_inf.sum(-1, keepdims=True) < attn_mask.shape[-1]
            if mask_row_valid is not None:
                # Keep the softmax graph finite as well as its final output.
                # Masking only after softmax would leave NaNs in its backward
                # for an additive row containing only -inf values.
                scores = jt.ternary(
                    mask_row_valid, scores, jt.zeros_like(scores))
            if mask_softmax is not None:
                attn = mask_softmax.softmax_v1(
                    scores, zero_all_neg_inf=True)
            else:
                attn = nn.softmax(scores, dim=-1)
            if mask_row_valid is not None:
                # PyTorch defines a fully masked query row as all zeros. A
                # finite sentinel would otherwise produce a uniform row, while
                # an additive -inf mask produces NaNs in ordinary softmax.
                attn = jt.ternary(mask_row_valid, attn, jt.zeros_like(attn))
            if float(dropout_p or 0.0) > 0.0:
                # torch SDPA always applies the supplied probability. Callers
                # pass dropout_p=0 in evaluation, so this path is training-only.
                attn = nn.dropout(attn, p=float(dropout_p), is_train=True)
            # Masks and causal bias are intentionally accumulated in fp32 for
            # low-precision inputs. Cast the probabilities back before the
            # value matmul, matching torch SDPA's output dtype and avoiding a
            # cublas fp32-by-fp16 dtype mismatch in training fallbacks.
            value_attn = attn
            if str(value_attn.dtype) != str(value.dtype):
                value_attn = value_attn.cast(str(value.dtype))
            out = jt.matmul(value_attn, value)
            if str(out.dtype) != str(query.dtype):
                out = out.cast(str(query.dtype))
            return out
        nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    g.scaled_dot_product_attention = nn.functional.scaled_dot_product_attention
    g._torch_sdpa_flash_backend_cache = _sdpa_flash_backend_cache

    _install_nn_extras(nn)
    import sys as _sys_nn
    _sys_nn.modules["torch.nn"] = nn
    if hasattr(nn, "functional"):
        _sys_nn.modules["torch.nn.functional"] = nn.functional
    _install_cuda(g)
    _install_version(g)
    _install_distributed(g)
    _install_tensor_methods(g, Var, _DTYPE_OBJS)
    _install_misc(g, Var, _DTYPE_OBJS)
    _install_torchdata_stateful_dataloader(g)
    _install_torchmetrics_fastpaths(g)
    _install_optimizers(g)
    _install_lr_scheduler(g)
    _install_autograd_function(g)
    _install_autograd(g)
    _install_tensordict_compat()
    _install_safetensors_shim()
    try:
        _install_flash_attn_shim()
    except Exception:
        pass


def _install_tensordict_compat():
    """Patch tensordict indexing for jittor Vars used as torch-style indices."""
    try:
        from tensordict.base import TensorDictBase
        from tensordict._lazy import LazyStackedTensorDict
    except Exception:
        return

    if getattr(TensorDictBase, "_jittor_index_compat", False):
        return

    def _normalize_index(idx):
        if isinstance(idx, jt.Var):
            arr = np.asarray(idx.detach().cpu().numpy())
            if arr.ndim == 0:
                return bool(arr.item()) if arr.dtype == np.bool_ else int(arr.item())
            if arr.dtype == np.bool_:
                return [int(i) for i in np.flatnonzero(arr)]
            return [int(x) for x in arr.reshape(-1)]
        if isinstance(idx, tuple):
            return tuple(_normalize_index(i) for i in idx)
        if isinstance(idx, list):
            return [_normalize_index(i) for i in idx]
        return idx

    _td_getitem = TensorDictBase.__getitem__
    def _getitem(self, index):
        return _td_getitem(self, _normalize_index(index))
    TensorDictBase.__getitem__ = _getitem
    TensorDictBase.__getitems__ = _getitem

    _lazy_getitem = LazyStackedTensorDict.__getitem__
    def _lazy_index(self, index):
        return _lazy_getitem(self, _normalize_index(index))
    LazyStackedTensorDict.__getitem__ = _lazy_index

    TensorDictBase._jittor_index_compat = True


def _install_autograd_function(g):
    """torch.autograd.Function exposes ctx.save_for_backward(*tensors) in
    forward() and a ctx.saved_tensors tuple in backward(). jittor's Function
    stores backward state via plain `self.<attr> = ...`, so it lacks both
    (bloom's GeLUFunction calls them). Add them to the Function class.
    """
    Fn = getattr(g, "Function", None)
    if Fn is None:
        return
    if not hasattr(Fn, "save_for_backward"):
        def save_for_backward(self, *tensors):
            # torch stores a tuple; a single un-tupled call still yields a tuple
            self._saved_tensors = tuple(tensors)
        Fn.save_for_backward = save_for_backward
    if "saved_tensors" not in getattr(Fn, "__dict__", {}):
        def _saved_tensors(self):
            return getattr(self, "_saved_tensors", ())
        Fn.saved_tensors = property(_saved_tensors)
    # torch's autograd engine reduces (sums) each grad a Function.backward returns
    # down to the shape of the corresponding *input* whenever forward broadcast that
    # input (e.g. TOOD's SigmoidGeometricMean multiplies cls_logits [N,80,H,W] by
    # cls_prob [N,1,H,W]; backward returns grad_y at [N,80,H,W]). jittor performs no
    # such reduction and raises "dvar->num != var->num". Record the forward input
    # shapes on __call__, then sum-to-shape each returned grad in the grad bridge.
    def _sum_grad_to(grad, shape):
        if grad is None or shape is None or not isinstance(grad, jt.Var):
            return grad
        gshape = grad.shape
        if list(gshape) == list(shape):
            return grad
        # Incompatible element counts: the returned grad does not correspond to
        # this input's true gradient. A custom Function may return a fully-shaped
        # grad for an input it actually ignores (3DGS's rasterizer returns a
        # [P,C] grad for the EMPTY placeholder inputs colors_precomp / cov3Ds_precomp
        # that aren't requires_grad in torch). torch discards grads for such
        # inputs; emulate by returning a correctly-shaped zero (jittor still tapes
        # the placeholder Var, so it needs a shape-matching grad, not None).
        tgt_items = 1
        for s in shape: tgt_items *= int(s)
        g_items = 1
        for s in gshape: g_items *= int(s)
        if tgt_items == 0 or (tgt_items != g_items and g_items % max(tgt_items, 1) != 0):
            return jt.zeros([int(s) for s in shape], dtype=grad.dtype)
        # drop leading dims that the input doesn't have (broadcast prepended them)
        extra = len(gshape) - len(shape)
        if extra > 0:
            grad = grad.sum(dims=tuple(range(extra)))
            gshape = grad.shape
        # sum over dims where the input was size-1 but grad is larger (keepdim)
        reduce_dims = [i for i in range(len(shape))
                       if int(shape[i]) == 1 and int(gshape[i]) != 1]
        if reduce_dims:
            grad = grad.sum(dims=tuple(reduce_dims), keepdims=True)
        if list(grad.shape) != list(shape):
            grad = grad.reshape(tuple(int(s) for s in shape))
        return grad

    _orig_fn_call = Fn.__call__
    def _call_record_inputs(self, *args, **kw):
        # capture forward input shapes (positional only -- jittor only tapes those)
        try:
            self._fwd_input_shapes = [
                (tuple(v.shape) if isinstance(v, jt.Var) else None) for v in args]
        except Exception:
            self._fwd_input_shapes = None
        # torch.autograd.Function exposes `ctx.needs_input_grad`: a tuple with one
        # bool per positional forward arg, True iff that arg is a tensor that
        # requires grad. Custom Functions branch on it (e.g. flex_gemm spconv:
        # `need_grad = any(ctx.needs_input_grad)`). A non-Var arg, or a Var with
        # stop-grad, contributes False -- matching torch.
        try:
            self.needs_input_grad = tuple(
                bool(isinstance(v, jt.Var) and not v.is_stop_grad()) for v in args)
        except Exception:
            self.needs_input_grad = tuple(isinstance(v, jt.Var) for v in args)
        out = _orig_fn_call(self, *args, **kw)
        # Capture each forward OUTPUT's (shape, dtype) so the grad bridge can
        # materialize a zeros grad for outputs that don't reach the backward'd
        # scalar (torch's materialize_grads=True; see grad() below).
        try:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            self._fwd_outputs = [
                (tuple(o.shape), str(o.dtype)) if isinstance(o, jt.Var) else None
                for o in outs]
        except Exception:
            self._fwd_outputs = None
        return out
    if getattr(Fn.__call__, "_torch_records_inputs", False) is not True:
        _call_record_inputs._torch_records_inputs = True
        Fn.__call__ = _call_record_inputs

    # torch.autograd.Function defines `@staticmethod backward(ctx, *grad_outputs)`;
    # jittor's Function.__call__ tapes self._grad, which calls `self.grad(*grads)`.
    # The shim maps execute->forward and save_for_backward/saved_tensors, but never
    # bridged backward->grad, so a torch-style custom Function (e.g. bloom's
    # GeLUFunction) raised "'GeLUFunction' object has no attribute 'grad'" in the
    # backward pass. Add a base `grad` that routes to a torch-style `backward` with
    # the instance as ctx. Gated on the base lacking its own grad; every native
    # jittor Function subclass (ACL ops, EMD, ...) defines grad(), which MRO-shadows
    # this, so they're untouched.
    # torch.autograd.Function defaults to materialize_grads=True; a Function may
    # opt out via ctx.set_materialize_grads(False). Store the flag on the ctx.
    if not hasattr(Fn, "set_materialize_grads"):
        def set_materialize_grads(self, value):
            self._materialize_grads = bool(value)
        Fn.set_materialize_grads = set_materialize_grads
    if "grad" not in getattr(Fn, "__dict__", {}):
        def grad(self, *grad_outputs):
            bw = getattr(type(self), "backward", None)
            if bw is None:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute 'grad'")
            # materialize_grads (torch default True): jittor hands None for a taped
            # output that doesn't reach the backward'd scalar, but torch passes
            # zeros_like(output) for FLOATING-point outputs (int/bool ones stay
            # None — non-differentiable). 3DGS's rasterizer returns (color, radii,
            # depth); a colour-only loss leaves depth's grad None, yet the C++
            # backward requires a real zero tensor for it.
            if getattr(self, "_materialize_grads", True) and any(
                    g is None for g in grad_outputs):
                outs = getattr(self, "_fwd_outputs", None)
                if outs is not None:
                    go = list(grad_outputs)
                    for i in range(min(len(go), len(outs))):
                        if go[i] is None and outs[i] is not None:
                            shp, dt = outs[i]
                            if not any(t in dt for t in ("int", "bool", "uint")):
                                go[i] = jt.zeros(shp, dtype=dt)
                    grad_outputs = tuple(go)
            ret = bw(self, *grad_outputs)
            shapes = getattr(self, "_fwd_input_shapes", None)
            if shapes is None:
                return ret
            single = not isinstance(ret, (tuple, list))
            grads = [ret] if single else list(ret)
            # reduce each input-grad to its forward input shape (torch broadcast bwd)
            for i in range(min(len(grads), len(shapes))):
                grads[i] = _sum_grad_to(grads[i], shapes[i])
            return grads[0] if single else tuple(grads)
        Fn.grad = grad


def _install_autograd(g):
    """Expose torch.autograd.grad / torch.autograd.backward (jittor lacks the
    `torch.autograd` namespace functions; it only has jt.grad). These wrap
    jt.grad so `import jittor as torch; torch.autograd.grad(out, inputs)` works.
    """
    import types as _types
    import jittor as _jt
    autograd = getattr(g, "autograd", None)
    if autograd is None or not isinstance(autograd, _types.ModuleType):
        autograd = _types.ModuleType("torch.autograd")
    # carry over the symbols other layers expect on torch.autograd
    if not hasattr(autograd, "Function"):
        autograd.Function = getattr(_jt, "Function", object)
    if not hasattr(autograd, "no_grad"):
        autograd.no_grad = getattr(g, "no_grad", _jt.no_grad)
    if not hasattr(autograd, "enable_grad"):
        autograd.enable_grad = getattr(g, "enable_grad", _jt.enable_grad)

    def _as_list(x):
        if isinstance(x, _jt.Var):
            return [x]
        return list(x)

    def grad(outputs, inputs, grad_outputs=None, retain_graph=None,
             create_graph=False, only_inputs=True, allow_unused=None,
             is_grads_batched=False, materialize_grads=False, **kw):
        # torch.autograd.grad(outputs, inputs, ...) -> tuple of grads, one per
        # input. jittor's jt.grad takes a single scalar loss; when several
        # outputs (or grad_outputs weights) are given, reduce them to one scalar
        # via sum(grad_outputs * output), matching torch's vector-Jacobian product.
        outs = _as_list(outputs)
        ins = _as_list(inputs)
        if grad_outputs is None:
            loss = outs[0].sum() if len(outs) == 1 else sum(o.sum() for o in outs)
        else:
            gos = _as_list(grad_outputs)
            loss = sum((o * w).sum() for o, w in zip(outs, gos))
        rg = bool(create_graph) if retain_graph is None else bool(retain_graph)
        if materialize_grads and allow_unused is False:
            raise ValueError(
                "Expected allow_unused to be True or not passed when "
                "materialize_grads=True, but got: allow_unused=False.")
        allow_unused = bool(materialize_grads) if allow_unused is None \
            else bool(allow_unused)
        gs = list(_jt.core.grad_optional(loss, ins, rg))
        missing = [i for i, value in enumerate(gs) if value is None]
        if missing and materialize_grads:
            for i in missing:
                gs[i] = _jt.zeros_like(ins[i])
                if create_graph:
                    gs[i].start_grad()
                else:
                    gs[i].stop_grad()
        elif missing and not allow_unused:
            raise RuntimeError(
                "One of the differentiated Tensors appears to not have been "
                "used in the graph. Set allow_unused=True if this is desired.")
        return tuple(gs)
    autograd.grad = grad

    def backward(tensors, grad_tensors=None, retain_graph=None,
                 create_graph=False, inputs=None, **kw):
        # torch.autograd.backward(tensors, ...) accumulates grads into leaf
        # .grad. Route each tensor through Var.backward (the optimizer bridge /
        # no-optimizer leaf path installed on Var).
        ts = _as_list(tensors)
        gts = None if grad_tensors is None else _as_list(grad_tensors)
        for i, t in enumerate(ts):
            gt = None if gts is None else gts[i]
            t.backward(gradient=gt, retain_graph=retain_graph)
        return None
    autograd.backward = backward

    if not hasattr(autograd, "Variable"):
        autograd.Variable = g.Tensor
    # torch.autograd.set_detect_anomaly / detect_anomaly — debug hooks jittor
    # lacks; 3DGS train.py calls set_detect_anomaly(args.detect_anomaly) at start.
    import contextlib as _ctxlib
    autograd.set_detect_anomaly = lambda *a, **k: _ctxlib.nullcontext()
    autograd.detect_anomaly = lambda *a, **k: _ctxlib.nullcontext()
    g.autograd = autograd
    import sys as _sys_autograd
    _sys_autograd.modules["torch.autograd"] = autograd
    autograd.__path__ = getattr(autograd, "__path__", [])
    functional = _sys_autograd.modules.get("torch.autograd.functional")
    if functional is None:
        functional = _types.ModuleType("torch.autograd.functional")
        from jittor.gradfunctional import jvp as _jvp, vjp as _vjp
        functional.jvp = _jvp
        functional.vjp = _vjp
        functional.__all__ = ["jvp", "vjp"]
        _sys_autograd.modules["torch.autograd.functional"] = functional
    autograd.functional = functional
    if "torch.autograd.profiler" not in _sys_autograd.modules:
        _prof = _types.ModuleType("torch.autograd.profiler")
        class EventList(list):
            def table(self, *args, **kwargs):
                return ""
            def export_chrome_trace(self, *args, **kwargs):
                return None
        class _RecordFunction:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        class profile(_RecordFunction):
            def function_events(self):
                return EventList()
            @property
            def key_averages(self):
                return lambda *args, **kwargs: EventList()
            def export_chrome_trace(self, *args, **kwargs):
                return None
        _prof.EventList = EventList
        _prof.record_function = lambda *args, **kwargs: _RecordFunction()
        _prof.profile = profile
        _prof.emit_nvtx = lambda *args, **kwargs: _RecordFunction()
        _prof.kineto_available = lambda: False
        _sys_autograd.modules["torch.autograd.profiler"] = _prof
    autograd.profiler = _sys_autograd.modules["torch.autograd.profiler"]






import collections as _collections
_MinMax = _collections.namedtuple("torch_return_types", ["values", "indices"])
_TopK = _collections.namedtuple("topk", ["values", "indices"])
_Sort = _collections.namedtuple("sort", ["values", "indices"])


def _install_reductions(g):
    """torch-correct argmax/argmin/max/min/sort/topk (jittor's differ:
    jittor argmax->(idx,val), jittor max(dim)->values only).
    NB: g IS the jittor module, so capture the ORIGINAL jittor ops before
    overwriting (else infinite recursion)."""
    import jittor as _jt
    _argmax = _jt.argmax
    _argmin = _jt.argmin
    _argsort = _jt.argsort
    _maximum = _jt.maximum
    _minimum = _jt.minimum
    _jt_max = _jt.max          # jittor-native reductions (values only)
    _jt_min = _jt.min
    _jt_var_max = _jt.Var.max  # native METHODS (0-dim scalar for full reduction)
    _jt_var_min = _jt.Var.min
    _topk = getattr(_jt, "topk", None)
    _gather = _jt.gather

    def _reduce_index(result):
        if isinstance(result, (tuple, list)):
            result = result[0]
        return result.int64()

    def argmax(x, dim=None, keepdim=False, keepdims=None):
        if keepdims is not None:
            keepdim = keepdims
        if dim is None:
            return _reduce_index(_argmax(x.reshape(-1), 0))
        try:
            res = _argmax(x, dim, keepdims=keepdim)
        except TypeError:
            res = _argmax(x, dim, keepdim=keepdim)
        return _reduce_index(res)
    def argmin(x, dim=None, keepdim=False, keepdims=None):
        if keepdims is not None:
            keepdim = keepdims
        if dim is None:
            return _reduce_index(_argmin(x.reshape(-1), 0))
        try:
            res = _argmin(x, dim, keepdims=keepdim)
        except TypeError:
            res = _argmin(x, dim, keepdim=keepdim)
        return _reduce_index(res)
    g.argmax = argmax
    g.argmin = argmin

    def _maxmin(which, x, *args, **kwargs):
        # jittor-internal callers use the `keepdims` kwarg (with an 's') and
        # expect values-only semantics; delegate straight to the native op so
        # we don't break jittor's own softmax/layernorm/etc.
        if "keepdims" in kwargs:
            native = _jt_max if which == "max" else _jt_min
            return native(x, *args, **kwargs)
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        other = kwargs.get("other", None)
        pos = list(args)
        if pos:
            if isinstance(pos[0], _jt.Var):
                other = pos[0]
            else:
                dim = pos[0]
                if len(pos) > 1:
                    keepdim = pos[1]
        if other is not None:
            return _maximum(x, other) if which == "max" else _minimum(x, other)
        if dim is None:
            # native scalar reduction via the captured METHOD (0-dim scalar);
            # NOT x.max(), which now routes back into this wrapper (recursion).
            return _jt_var_max(x) if which == "max" else _jt_var_min(x)
        af = argmax if which == "max" else argmin
        idx = af(x, dim=dim, keepdim=keepdim)
        if keepdim:
            val = _jt.gather(x, dim, idx)
        elif x.ndim == 1:
            val = x[idx]
        else:
            val = _jt.gather(x, dim, idx.unsqueeze(dim)).squeeze(dim)
        return _MinMax(val, idx.int64())
    g.max = lambda x, *a, **k: _maxmin("max", x, *a, **k)
    g.min = lambda x, *a, **k: _maxmin("min", x, *a, **k)

    def topk(x, k, dim=-1, largest=True, sorted=True):
        # jittor's native topk is unreliable on the ACL backend (internal
        # getitem "too many slices"); use an argsort-based gather instead.
        idx, _ = _argsort(x, dim=dim, descending=largest)
        nd = x.ndim
        d = dim if dim >= 0 else dim + nd
        sl = [slice(None)] * nd
        sl[d] = slice(0, k)
        idx = idx[tuple(sl)]
        val = _gather(x, d, idx)
        return _TopK(val, idx.int64())
    g.topk = topk

    def sort(x, dim=-1, descending=False, **kw):
        idx, val = _argsort(x, dim=dim, descending=descending)
        return _Sort(val, idx.int64())
    g.sort = sort
    g.argsort = lambda x, dim=-1, descending=False, **kw: _argsort(x, dim=dim, descending=descending)[0].int64()

    # --- Tensor METHOD forms. jittor-core uses none of these as Var methods (only
    # the python list.sort builtin), so installing torch semantics here is safe;
    # it was verified that .max/.min methods ARE used internally, so those stay
    # native (values-only) and are intentionally NOT overridden. ---
    Var = _jt.Var
    Var.sort = lambda self, dim=-1, descending=False, **kw: sort(self, dim=dim, descending=descending)
    Var.argsort = lambda self, dim=-1, descending=False, **kw: g.argsort(self, dim=dim, descending=descending)
    Var.topk = lambda self, k, dim=-1, largest=True, sorted=True: topk(self, k, dim=dim, largest=largest, sorted=sorted)
    # Tensor.softmax/log_softmax accept a `dtype=` (cast before the op) which
    # jittor's native method rejects (vLLM's sampler: logits.softmax(dim=-1,
    # dtype=torch.float32)).
    def _var_softmax(self, dim=-1, dtype=None, **kw):
        x = self.cast(_dtype_to_str(dtype)) if dtype is not None else self
        return _jt.nn.softmax(x, dim=dim)
    Var.softmax = _var_softmax
    def _var_log_softmax(self, dim=-1, dtype=None, **kw):
        x = self.cast(_dtype_to_str(dtype)) if dtype is not None else self
        return _jt.nn.log_softmax(x, dim=dim)
    Var.log_softmax = _var_log_softmax
    # torch's Tensor.max(dim)/min(dim) returns the (values, indices) namedtuple --
    # mmdetection relies on this pervasively (`v, i = overlaps.max(dim=0)`). jittor's
    # native method returns values-only and is used by core/linalg/einops with the
    # `keepdims=` spelling (handled natively inside _maxmin) or a bare dim. Route
    # everything through _maxmin: keepdims= -> native values; a bare/torch dim ->
    # namedtuple; no dim -> native scalar. The few jittor-internal callers that pass
    # a BARE dim and want values-only extract `.values` at their call site.
    Var.max = lambda self, *a, **k: _maxmin("max", self, *a, **k)
    Var.min = lambda self, *a, **k: _maxmin("min", self, *a, **k)

    # torch's var/std default to UNBIASED (Bessel, correction=1); jittor's native var
    # defaults to biased (numpy-aligned) -- a silent-wrong divergence for torch code.
    # Fix in the torch layer only (native jt.var stays numpy-aligned). Support both
    # the legacy `unbiased=` and modern `correction=` kwargs.
    _jt_var = Var.var
    def _correction_to_unbiased(unbiased, correction):
        if correction is not None:
            return correction != 0
        if unbiased is not None:
            return bool(unbiased)
        return True                       # torch default
    def _multidim_var(self, dims, unbiased, keepdim):
        # torch-compat: var over a LIST/TUPLE of axes. jittor's native var() `dim=`
        # slot is scalar-only (a list crashes with `is_type<int64>(oi)`), and its
        # separate `dims=` path returns a WRONG-shaped/value result for partial
        # multi-axis reductions. Compute directly from mean/sum (which DO accept a
        # tuple) so every axis subset matches torch exactly, preserving unbiased
        # (Bessel) + keepdim semantics.
        dims = [int(d) % self.ndim for d in dims]
        mean = _jt.mean(self, dims, keepdims=True)
        sqr = (self - mean) ** 2
        out = _jt.sum(sqr, dims=dims, keepdims=keepdim)
        n = 1
        for d in dims:
            n *= self.shape[d]
        if unbiased:
            n = n - 1
        return out / n
    def _torch_var(self, dim=None, unbiased=None, keepdim=False, keepdims=None,
                   correction=None, **kw):
        ub = _correction_to_unbiased(unbiased, correction)
        kd = bool(keepdim) or bool(keepdims)
        if isinstance(dim, (list, tuple)):
            return _multidim_var(self, dim, ub, kd)
        return _jt_var(self, dim=dim, unbiased=ub, keepdims=kd)
    def _torch_std(self, dim=None, unbiased=None, keepdim=False, keepdims=None,
                   correction=None, **kw):
        # std == sqrt(var) with the correct bias. jittor's native std is hardcoded
        # unbiased AND floors at maximum(1e-6) (torch doesn't), so derive from var.
        return _torch_var(self, dim=dim, unbiased=unbiased, keepdim=keepdim,
                          keepdims=keepdims, correction=correction).sqrt()
    Var.var = _torch_var
    Var.std = _torch_std
    g.var = lambda x, *a, **k: _torch_var(x, *a, **k)
    g.std = lambda x, *a, **k: _torch_std(x, *a, **k)

    # missing methods (truly absent on Var -> pure additive)
    Var.masked_select = lambda self, mask: self[mask]      # torch: 1-D of selected

    def _masked_scatter(self, mask, source):
        # torch.Tensor.masked_scatter(mask, source): copy elements of `source`
        # (consumed in row-major order) into the positions of `self` where `mask`
        # is True; `mask` broadcasts to self.shape. Out-of-place, and DIFFERENTIABLE
        # w.r.t. both self and source -- the Qwen-VL path scatters vision-tower
        # image_embeds into the text inputs_embeds, and grads must reach the ViT.
        # Implemented as gather(source, running-count-of-True) then where(mask),
        # avoiding any sliced in-place write (a jittor no-view no-op).
        m = mask
        if tuple(m.shape) != tuple(self.shape):
            m = m.broadcast(self.shape)
        mb = m.bool()
        flat_mask = mb.reshape(-1)
        # index into source.flatten() for each position = (#True strictly before it)
        sel_idx = flat_mask.int32().cumsum(0) - 1
        sel_idx = sel_idx.maximum(0).minimum(source.numel() - 1)  # clamp (unused where mask False)
        src_flat = source.reshape(-1)
        gathered = src_flat[sel_idx].reshape(self.shape)
        if str(gathered.dtype) != str(self.dtype):
            gathered = gathered.cast(str(self.dtype))
        return jt.ternary(mb, gathered, self)
    Var.masked_scatter = _masked_scatter

    def _masked_scatter_(self, mask, source):
        # in-place variant: write the result back through assign() so the same Var
        # (and any module attribute holding it) reflects the update.
        out = _masked_scatter(self, mask, source)
        self.assign(out)
        return self
    Var.masked_scatter_ = _masked_scatter_

    def _unfold(self, dimension, size, step):
        # torch's Tensor.unfold(dim, size, step): sliding windows along `dim`,
        # appending a new last dim of length `size`. out[...,i,...,j]=x[...,i*step+j,...]
        nd = self.ndim
        d = dimension if dimension >= 0 else dimension + nd
        n = (self.shape[d] - size) // step + 1
        out_shape = list(self.shape); out_shape[d] = n; out_shape.append(size)
        src = [f"i{k}" for k in range(nd)]
        src[d] = f"i{d}*{step}+i{nd}"                       # window pos + within-window
        return self.reindex(out_shape, src)
    Var.unfold = _unfold

    def _diagonal(self, offset=0, dim1=0, dim2=1):
        # torch's Tensor.diagonal: drop dim1,dim2 and append a diagonal dim.
        nd = self.ndim
        d1 = dim1 if dim1 >= 0 else dim1 + nd
        d2 = dim2 if dim2 >= 0 else dim2 + nd
        s1, s2 = self.shape[d1], self.shape[d2]
        dl = max(0, min(s1, s2 - offset)) if offset >= 0 else max(0, min(s1 + offset, s2))
        keep = [k for k in range(nd) if k != d1 and k != d2]
        out_shape = [self.shape[k] for k in keep] + [dl]
        last = len(keep)
        src = [None] * nd
        for outpos, k in enumerate(keep):
            src[k] = f"i{outpos}"
        src[d1] = f"i{last}+{max(0, -offset)}"
        src[d2] = f"i{last}+{max(0, offset)}"
        return self.reindex(out_shape, src)
    Var.diagonal = _diagonal

    # --- elementwise / reduction ops missing as torch methods (all additive) ---
    if not hasattr(Var, "sign"):
        # torch sign: -1/0/+1 (nan->nan in torch; this gives 0 for nan, an accepted edge)
        Var.sign = lambda self: (self > 0).cast(self.dtype) - (self < 0).cast(self.dtype)
    if not hasattr(Var, "trunc"):
        Var.trunc = lambda self: _jt.ternary(self >= 0, _jt.floor(self), _jt.ceil(self))
    if not hasattr(Var, "frac"):
        Var.frac = lambda self: self - _jt.ternary(self >= 0, _jt.floor(self), _jt.ceil(self))
    if not hasattr(Var, "nan_to_num"):
        def _nan_to_num(self, nan=0.0, posinf=None, neginf=None):
            # Replace nan with one ternary, then clamp to the ±inf replacement bounds.
            # NB: a jittor JIT codegen bug SEGFAULTS on chained isinf+ternary over a
            # tensor holding inf/nan (tracked, #11), so we deliberately avoid that and
            # use a clamp. This is EXACT for the default (float32-max) bounds -- finite
            # values are untouched and ±inf map to ±max. For *narrow custom* posinf/
            # neginf it also clamps finite values past them (a rare, documented
            # deviation accepted to avoid the core segfault).
            pi = 3.4028234663852886e38 if posinf is None else posinf   # exact float32 max
            ni = -3.4028234663852886e38 if neginf is None else neginf
            out = _jt.ternary(_jt.isnan(self), _jt.full_like(self, nan), self)
            return out.minimum(pi).maximum(ni)
        Var.nan_to_num = _nan_to_num
        g.nan_to_num = lambda x, nan=0.0, posinf=None, neginf=None: _nan_to_num(x, nan, posinf, neginf)
    if not hasattr(Var, "amax"):
        def _amax(self, dim=None, keepdim=False):
            d = list(dim) if isinstance(dim, (tuple, list)) else dim
            return _jt_max(self, d, keepdims=keepdim) if d is not None else self.max()
        def _amin(self, dim=None, keepdim=False):
            d = list(dim) if isinstance(dim, (tuple, list)) else dim
            return _jt_min(self, d, keepdims=keepdim) if d is not None else self.min()
        Var.amax = _amax
        Var.amin = _amin
        g.amax = lambda x, dim=None, keepdim=False: _amax(x, dim, keepdim)
        g.amin = lambda x, dim=None, keepdim=False: _amin(x, dim, keepdim)
    if not hasattr(Var, "count_nonzero"):
        def _count_nonzero(self, dim=None):
            nz = (self != 0).int32()
            return nz.sum(dim) if dim is not None else nz.sum()
        Var.count_nonzero = _count_nonzero
        g.count_nonzero = lambda x, dim=None: _count_nonzero(x, dim)
    if not hasattr(g, "logaddexp"):
        def _logaddexp(a, b):
            m = _jt.maximum(a, b)                       # numerically stable
            return m + _jt.log(_jt.exp(a - m) + _jt.exp(b - m))
        g.logaddexp = _logaddexp
        Var.logaddexp = _logaddexp

    # argmax/argmin METHOD forms: torch returns just the indices; jittor's native
    # Var.argmax returns (idx, val). Core uses these only in docstrings, so override.
    Var.argmax = lambda self, dim=None, keepdim=False: argmax(self, dim, keepdim)
    Var.argmin = lambda self, dim=None, keepdim=False: argmin(self, dim, keepdim)
    # addcmul/addcdiv: self + value * (t1 (*|/) t2)
    Var.addcmul = lambda self, t1, t2, value=1: self + value * (t1 * t2)
    Var.addcdiv = lambda self, t1, t2, value=1: self + value * (t1 / t2)
    if not hasattr(Var, "broadcast_to"):
        Var.broadcast_to = lambda self, shape: self.broadcast(shape)
    # torch-compat: module-level torch.broadcast_to(input, shape) (some code calls
    # the functional form, not the method). Expands `input` to `shape` without copy.
    if not hasattr(g, "broadcast_to"):
        g.broadcast_to = lambda input, shape: input.broadcast(shape)


def _wrap_constructors(g):
    """Wrap jittor tensor constructors to accept torch kwargs (device=,
    requires_grad=, layout=, pin_memory=, out=) and torch dtype objects."""
    import functools, inspect
    _DROP = ("device", "requires_grad", "layout", "pin_memory", "memory_format",
             "out", "non_blocking")

    def wrap(name):
        orig = getattr(g, name, None)
        if orig is None:
            return
        # A few jittor factories (ones_like, tril, triu) have no `dtype` param at
        # all, unlike torch where every one of these accepts dtype=. For those we
        # can't forward dtype= (jittor raises "unexpected keyword argument
        # 'dtype'"); instead pop it and cast the result. Detect support once here.
        try:
            _sig = inspect.signature(orig)
            _accepts_dtype = ("dtype" in _sig.parameters or
                              any(p.kind == p.VAR_KEYWORD
                                  for p in _sig.parameters.values()))
        except (ValueError, TypeError):
            _accepts_dtype = True  # builtins w/o introspectable sig: assume ok
        def _shape_dim(v):
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, jt.Var):
                try:
                    if int(np.prod(tuple(v.shape))) == 1:
                        return int(v.item())
                except Exception:
                    pass
            return v
        def _shape_arg(v):
            if isinstance(v, jt.NanoVector):
                return tuple(int(x) for x in v)
            if isinstance(v, tuple):
                return tuple(_shape_dim(x) for x in v)
            if isinstance(v, list):
                return tuple(_shape_dim(x) for x in v)
            return _shape_dim(v)
        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            # torch device='cpu' must produce a host-resident Var (native exts
            # check tensor.is_cpu()). Capture the device before dropping it and,
            # when CPU is requested, build the Var under use_cuda=0 so its
            # allocator is the host allocator (Var.location()=='cpu').
            _requested_device = kwargs.get("device")
            _want_cpu = _device_is_cpu(_requested_device)
            _want_cuda = _device_is_cuda(_requested_device)
            if _want_cuda:
                jt.flags.use_cuda = 1
            _requires_grad = bool(kwargs.get("requires_grad", False))
            for k in _DROP:
                kwargs.pop(k, None)
            # torch accepts numpy integer scalars as shape dims, e.g.
            # torch.zeros(1, np.int64(49), 512) (mmdet PVT builds pos_shape via
            # `pretrain_img_size // patch_size`, which yields numpy ints). jittor's
            # C++ shape converter strictly wants Python int (is_type<int64>), so a
            # numpy scalar raises. Coerce numpy integer/float scalars (and 0-dim
            # numpy arrays) in the positional args to plain Python scalars. Only
            # numpy scalars are touched, so normal int/tuple shape args are untouched.
            _is_like_factory = name.endswith("_like")
            if args and not _is_like_factory:
                args = tuple(_shape_arg(a) for a in args)
            # normalize a Size/NanoVector shape arg (e.g. torch.zeros(x.size()))
            # to a plain tuple — jittor's factories reject tuple subclasses /
            # NanoVector. transformers BertEmbeddings does torch.zeros(position_ids.size()).
            if (not _is_like_factory) and args and (isinstance(args[0], jt.NanoVector) or
                         (isinstance(args[0], tuple) and type(args[0]) is not tuple)):
                args = (tuple(int(x) for x in args[0]),) + tuple(args[1:])
            # torch allows the shape via the size= keyword: torch.ones(size=(2,3))
            # (canine's _create_3d_attention_mask_from_input_mask). Only the shape
            # factories get a size= kwarg, and only with no positional shape, so
            # this is safe across all wrapped constructors.
            if "size" in kwargs and not args:
                sz = kwargs.pop("size")
                args = (tuple(sz),) if hasattr(sz, "__len__") else (sz,)
            # torch.full(size, fill_value=...) / full_like(input, fill_value=...):
            # jittor's full(shape, val) / full_like(x, val) take the value as the 2nd
            # positional. transformers' beam scorer passes fill_value= as a keyword, so
            # map it onto the next positional slot (only full/full_like ever get it).
            if "fill_value" in kwargs:
                args = tuple(args) + (kwargs.pop("fill_value"),)
            _cast_to = None  # cast after construction when needed for torch dtype semantics
            if "dtype" in kwargs:
                if kwargs["dtype"] is None:
                    # torch.empty/zeros(..., dtype=None) -> the default dtype.
                    # jittor's factories reject dtype=None, so resolve it.
                    if _accepts_dtype:
                        try:
                            kwargs["dtype"] = _dtype_to_str(g.get_default_dtype())
                        except Exception:
                            kwargs.pop("dtype")
                    else:
                        kwargs.pop("dtype")
                elif _accepts_dtype:
                    kwargs["dtype"] = _dtype_to_str(kwargs["dtype"])
                    _cast_to = kwargs["dtype"]
                else:
                    # ones_like / tril / triu have no dtype param in jittor; torch
                    # accepts one. Pop it and cast the result instead.
                    _cast_to = _dtype_to_str(kwargs.pop("dtype"))
            if _want_cpu and jt.flags.use_cuda:
                # Build on the host so the result is genuinely CPU-resident.
                with jt.flag_scope(use_cuda=0):
                    out = orig(*args, **kwargs)
                    if _cast_to is not None:
                        out = out.cast(_cast_to)
                    out.sync()
                try:
                    out._jittor_torch_ext_mutable = True
                    out._jittor_torch_force_cpu = True
                except Exception:
                    pass
                if _requires_grad:
                    out.requires_grad_(True)
                    _torch_register_leaf(out)
                return out
            out = orig(*args, **kwargs)
            if _cast_to is not None:
                out = out.cast(_cast_to)
            if _want_cuda:
                out = _make_cuda_resident(out, force=True)
            try:
                out._jittor_torch_ext_mutable = True
            except Exception:
                pass
            if _requires_grad:
                out.requires_grad_(True)
                _torch_register_leaf(out)
            return out
        wrapped._torch_wrapped = True
        setattr(g, name, wrapped)

    for name in ("zeros", "ones", "empty", "full", "arange", "rand", "randn",
                 "randint", "eye", "linspace", "zeros_like", "ones_like",
                 "empty_like", "full_like", "randn_like", "rand_like", "tril",
                 "triu", "normal"):
        wrap(name)


def _install_random_and_linspace(g):
    """torch-compat for linspace(dtype=) and the random samplers' generator= arg.

    Runs AFTER _wrap_constructors, so it wraps the already-kwarg-tolerant
    versions. jittor's linspace has no `dtype` and its random ops have no
    `generator`, so torch code passing either currently raises TypeError.
    """
    import functools

    # torch.linspace(..., dtype=) -- jittor's linspace has no dtype param. Pop
    # it and cast the result, matching torch (default float32 stays unchanged).
    _lin = getattr(g, "linspace", None)
    if _lin is not None:
        @functools.wraps(_lin)
        def linspace(*args, dtype=None, **kwargs):
            # torch.linspace(start, end, steps): mmdet (DETR reference points) passes
            # tensor/float scalars for start/end and a tensor for steps; jittor needs
            # python float start/end and an int steps.
            args = list(args)
            if len(args) >= 1 and hasattr(args[0], "item"): args[0] = float(args[0])
            if len(args) >= 2 and hasattr(args[1], "item"): args[1] = float(args[1])
            if len(args) >= 3 and not isinstance(args[2], int): args[2] = int(args[2])
            if "steps" in kwargs and not isinstance(kwargs["steps"], int):
                kwargs["steps"] = int(kwargs["steps"])
            r = _lin(*args, **kwargs)
            if dtype is not None:
                r = r.cast(_dtype_to_str(dtype))
            return r
        g.linspace = linspace

    # torch.randn/rand/randint(..., generator=) -- jittor samplers seed off the
    # global RNG and reject `generator`. When a Generator is given, seed the
    # global RNG from it (initial_seed()/seed) so the draw is reproducible,
    # then restore nothing (matches torch users who pass a seeded generator for
    # determinism). Without a generator, behavior is unchanged.
    def _seed_from(gen):
        if gen is None:
            return
        s = None
        for attr in ("initial_seed", "seed"):
            fn = getattr(gen, attr, None)
            if callable(fn):
                try:
                    s = fn()
                    break
                except Exception:
                    s = None
        if s is None:
            s = getattr(gen, "_seed", None)
        if s is not None and hasattr(jt, "set_global_seed"):
            jt.set_global_seed(int(s))

    def wrap_gen(name):
        orig = getattr(g, name, None)
        if orig is None:
            return
        @functools.wraps(orig)
        def wrapped(*args, generator=None, **kwargs):
            _seed_from(generator)
            return orig(*args, **kwargs)
        setattr(g, name, wrapped)

    for name in ("randn", "rand", "randint", "randperm", "normal",
                 "randn_like", "rand_like", "multinomial", "bernoulli"):
        wrap_gen(name)


def _install_nn_extras(nn):
    # Activation modules torch has that jittor.nn may lack.
    import jittor as _jt
    _install_init_aliases()
    import sys as _sys_nn_private
    import types as _types_nn_private

    if not isinstance(getattr(nn, "Parameter", None), type):
        _native_parameter = getattr(nn, "Parameter", None)
        class _ParameterMeta(type):
            def __instancecheck__(cls, obj):
                return isinstance(obj, _jt.Var)
            def __call__(cls, data=None, requires_grad=True):
                return _torch_make_parameter(data, requires_grad=requires_grad)
        class Parameter(metaclass=_ParameterMeta):
            pass
        class UninitializedTensorMixin:
            pass
        class UninitializedParameter:
            pass
        class UninitializedBuffer:
            pass
        nn.Parameter = Parameter
        param_mod = _types_nn_private.ModuleType("torch.nn.parameter")
        param_mod.Parameter = Parameter
        param_mod.UninitializedTensorMixin = UninitializedTensorMixin
        param_mod.UninitializedParameter = UninitializedParameter
        param_mod.UninitializedBuffer = UninitializedBuffer
        _sys_nn_private.modules["torch.nn.parameter"] = param_mod
        nn.parameter = param_mod

    modules_pkg = _sys_nn_private.modules.get("torch.nn.modules")
    if modules_pkg is None:
        modules_pkg = _types_nn_private.ModuleType("torch.nn.modules")
        _sys_nn_private.modules["torch.nn.modules"] = modules_pkg
    modules_pkg.__path__ = getattr(modules_pkg, "__path__", [])
    module_mod = _sys_nn_private.modules.get("torch.nn.modules.module")
    if module_mod is None:
        module_mod = _types_nn_private.ModuleType("torch.nn.modules.module")
        _sys_nn_private.modules["torch.nn.modules.module"] = module_mod
    module_mod.Module = nn.Module
    module_mod._EXTRA_STATE_KEY_SUFFIX = "_extra_state"
    module_mod._global_backward_hooks = getattr(module_mod, "_global_backward_hooks", {})
    module_mod._global_forward_hooks = getattr(module_mod, "_global_forward_hooks", {})
    module_mod._global_forward_pre_hooks = getattr(module_mod, "_global_forward_pre_hooks", {})
    module_mod._IncompatibleKeys = getattr(module_mod, "_IncompatibleKeys", type(
        "_IncompatibleKeys", (tuple,), {
            "__new__": lambda cls, missing_keys, unexpected_keys: tuple.__new__(cls, (missing_keys, unexpected_keys)),
            "missing_keys": property(lambda self: self[0]),
            "unexpected_keys": property(lambda self: self[1]),
        }))
    modules_pkg.Module = nn.Module
    modules_pkg.module = module_mod
    for _cn in dir(nn):
        if _cn and _cn[0].isupper() and not hasattr(modules_pkg, _cn):
            try:
                setattr(modules_pkg, _cn, getattr(nn, _cn))
            except Exception:
                pass
    container_mod = _sys_nn_private.modules.get("torch.nn.modules.container")
    if container_mod is None:
        container_mod = _types_nn_private.ModuleType("torch.nn.modules.container")
        _sys_nn_private.modules["torch.nn.modules.container"] = container_mod
    for _cn in ("Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict"):
        if hasattr(nn, _cn):
            setattr(container_mod, _cn, getattr(nn, _cn))
    modules_pkg.container = container_mod
    try:
        from jittor.misc import _single, _pair, _triple, _ntuple
    except Exception:
        _single = lambda x: x if isinstance(x, tuple) else (x,)
        _pair = lambda x: x if isinstance(x, tuple) else (x, x)
        _triple = lambda x: x if isinstance(x, tuple) else (x, x, x)
        def _ntuple(n):
            return lambda x: x if isinstance(x, tuple) else tuple([x] * n)

    def _mk_nn_submod(_name, **_attrs):
        _full = "torch.nn.modules." + _name
        _mod = _sys_nn_private.modules.get(_full)
        if _mod is None:
            _mod = _types_nn_private.ModuleType(_full)
            _sys_nn_private.modules[_full] = _mod
        for _ak, _av in _attrs.items():
            if _av is not None:
                setattr(_mod, _ak, _av)
        setattr(modules_pkg, _name, _mod)
        return _mod

    _mk_nn_submod("utils", _single=_single, _pair=_pair, _triple=_triple,
                  _ntuple=_ntuple, _quadruple=_ntuple(4))
    _mk_nn_submod("batchnorm",
                  _BatchNorm=getattr(nn, "BatchNorm", None),
                  BatchNorm=getattr(nn, "BatchNorm", None),
                  BatchNorm1d=getattr(nn, "BatchNorm1d", getattr(nn, "BatchNorm", None)),
                  BatchNorm2d=getattr(nn, "BatchNorm2d", getattr(nn, "BatchNorm", None)),
                  BatchNorm3d=getattr(nn, "BatchNorm3d", getattr(nn, "BatchNorm", None)),
                  SyncBatchNorm=getattr(nn, "SyncBatchNorm", getattr(nn, "BatchNorm", None)))
    _mk_nn_submod("normalization",
                  GroupNorm=getattr(nn, "GroupNorm", None),
                  LayerNorm=getattr(nn, "LayerNorm", None),
                  LocalResponseNorm=getattr(nn, "LocalResponseNorm", None))
    _mk_nn_submod("activation",
                  ReLU=getattr(nn, "ReLU", None), SiLU=getattr(nn, "SiLU", None),
                  Sigmoid=getattr(nn, "Sigmoid", None), Tanh=getattr(nn, "Tanh", None),
                  GELU=getattr(nn, "GELU", None), LeakyReLU=getattr(nn, "LeakyReLU", None))
    nn.modules = modules_pkg

    parallel_mod = _sys_nn_private.modules.get("torch.nn.parallel")
    if parallel_mod is None:
        parallel_mod = _types_nn_private.ModuleType("torch.nn.parallel")
        _sys_nn_private.modules["torch.nn.parallel"] = parallel_mod

    class _DataParallel(nn.Module):
        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module

        def execute(self, *args, **kwargs):
            return self.module(*args, **kwargs)

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class _DistributedDataParallel(_DataParallel):
        require_backward_grad_sync = True

        def no_sync(self):
            import contextlib as _ctxlib
            return _ctxlib.nullcontext()

    parallel_mod.DataParallel = getattr(parallel_mod, "DataParallel", _DataParallel)
    parallel_mod.DistributedDataParallel = getattr(
        parallel_mod, "DistributedDataParallel", _DistributedDataParallel)
    parallel_distributed_mod = _sys_nn_private.modules.get("torch.nn.parallel.distributed")
    if parallel_distributed_mod is None:
        parallel_distributed_mod = _types_nn_private.ModuleType("torch.nn.parallel.distributed")
        _sys_nn_private.modules["torch.nn.parallel.distributed"] = parallel_distributed_mod
    parallel_distributed_mod.DistributedDataParallel = parallel_mod.DistributedDataParallel
    parallel_mod.distributed = parallel_distributed_mod
    nn.DataParallel = parallel_mod.DataParallel
    nn.parallel = parallel_mod

    # transformers 4.56.x imports torch.nn.attention.flex_attention from
    # masking_utils when torch is reported available. TRELLIS does not execute
    # PyTorch flex attention through this API, but the namespace must exist for
    # lazy model imports such as DINOv3ViTModel.
    attn_mod = _sys_nn_private.modules.get("torch.nn.attention")
    if attn_mod is None:
        attn_mod = _types_nn_private.ModuleType("torch.nn.attention")
        _sys_nn_private.modules["torch.nn.attention"] = attn_mod
    flex_mod = _sys_nn_private.modules.get("torch.nn.attention.flex_attention")
    if flex_mod is None:
        flex_mod = _types_nn_private.ModuleType("torch.nn.attention.flex_attention")
        def _flex_attention(*args, **kwargs):
            raise NotImplementedError("flex_attention is not supported on jittor backend")
        flex_mod.flex_attention = _flex_attention
        flex_mod.create_block_mask = lambda *args, **kwargs: None
        flex_mod.BlockMask = type("BlockMask", (), {})
        flex_mod._DEFAULT_SPARSE_BLOCK_SIZE = 128
        flex_mod.and_masks = lambda *args, **kwargs: None
        flex_mod.or_masks = lambda *args, **kwargs: None
        flex_mod.AuxRequest = type("AuxRequest", (), {})
        flex_mod.AuxOutput = type("AuxOutput", (), {})
        flex_mod.flex_attention_hop = None
        flex_mod.noop_mask = lambda *args, **kwargs: None
        _sys_nn_private.modules["torch.nn.attention.flex_attention"] = flex_mod
    attn_mod.flex_attention = flex_mod
    nn.attention = attn_mod

    # nn.utils.clip_grad_norm_/clip_grad_value_ (also provided by torch_shim,
    # but needed for the bare `import jittor as torch` path too).
    if not hasattr(nn, "utils") or not hasattr(getattr(nn, "utils", None), "clip_grad_norm_"):
        import types as _t
        _u = getattr(nn, "utils", None) or _t.ModuleType("torch.nn.utils")
        def _grads_of(params):
            params = list(params)
            opt = getattr(_jt, "_current_optimizer", None)
            out = []
            for p in params:
                gg = None
                if opt is not None:
                    try: gg = opt.find_grad(p)
                    except Exception: gg = None
                if gg is None:
                    gg = getattr(p, "grad", None)
                if gg is not None:
                    out.append(gg)
            return out
        def clip_grad_norm_(parameters, max_norm, norm_type=2.0,
                            error_if_nonfinite=False, **k):
            if isinstance(parameters, _jt.Var):
                parameters = [parameters]
            grads = _grads_of(parameters)
            return _clip_grad_norm_device(
                grads, max_norm, norm_type, error_if_nonfinite)
        def clip_grad_value_(parameters, clip_value, **k):
            if isinstance(parameters, _jt.Var):
                parameters = [parameters]
            for g in _grads_of(parameters):
                g.update(g.clamp(-clip_value, clip_value))
        _u.clip_grad_norm_ = clip_grad_norm_
        _u.clip_grad_value_ = clip_grad_value_

        # --- weight_norm / spectral_norm (reparametrizations) ---
        # torch reparametrizes a module's `weight` param into other params/buffers and
        # recomputes `weight` before each forward via a pre-forward hook. jittor has a
        # single-slot pre-forward hook, so route every reparametrization through one
        # dispatcher that calls each registered recompute fn (supports weight_norm +
        # spectral_norm on the same module, and preserves any pre-existing hook).
        def _ensure_reparam_hook(module):
            fns = getattr(module, "_reparam_fns", None)
            if fns is None:
                fns = []
                module._reparam_fns = fns
                prev = getattr(module, "__fhook2__", None)
                def _dispatch(mod, *a):
                    if prev is not None:
                        prev(mod, *a)
                    for fn in mod._reparam_fns:
                        fn(mod)
                module.register_pre_forward_hook(_dispatch)
            return fns

        def _norm_except_dim(v, dim):
            # L2 norm over all dims except `dim`, keepdim (torch._norm_except_dim, pow=2).
            if dim is None or dim == -1:
                return _jt.sqrt((v * v).sum())
            dims = [d for d in range(v.ndim) if d != dim]
            if not dims:
                return v.abs()
            return _jt.sqrt((v * v).sum(dims, keepdims=True))

        def weight_norm(module, name="weight", dim=0):
            w = getattr(module, name)
            try: delattr(module, name)
            except Exception: pass
            setattr(module, name + "_g", _norm_except_dim(w, dim).clone())
            setattr(module, name + "_v", w.clone())
            def _recompute(mod):
                gg = getattr(mod, name + "_g"); vv = getattr(mod, name + "_v")
                neww = vv * (gg / _norm_except_dim(vv, dim))
                neww.persistent = False          # exclude from parameters()/state_dict()
                setattr(mod, name, neww)
            _ensure_reparam_hook(module).append(_recompute)
            _recompute(module)                   # materialize weight before first forward
            return module

        def remove_weight_norm(module, name="weight"):
            gg = getattr(module, name + "_g"); vv = getattr(module, name + "_v")
            # final weight = v * g/||v||; restore it as a plain trainable param
            dimspec = 0
            final = vv * (gg / _norm_except_dim(vv, dimspec))
            for k in (name + "_g", name + "_v"):
                try: delattr(module, k)
                except Exception: pass
            final.persistent = True
            setattr(module, name, final.clone())
            module._reparam_fns = []             # drop recompute fns (torch removes the hook)
            return module

        def _l2_normalize(x, eps):
            return x / (_jt.sqrt((x * x).sum()) + eps)

        def spectral_norm(module, name="weight", n_power_iterations=1, eps=1e-12, dim=None):
            w = getattr(module, name)
            sdim = 0 if dim is None else dim
            def _to_mat(W):
                if sdim == 0:
                    return W.reshape(W.shape[0], -1)
                perm = [sdim] + [d for d in range(W.ndim) if d != sdim]
                return W.permute(*perm).reshape(W.shape[sdim], -1)
            wmat = _to_mat(w)
            h, wd = int(wmat.shape[0]), int(wmat.shape[1])
            try: delattr(module, name)
            except Exception: pass
            setattr(module, name + "_orig", w.clone())
            module.register_buffer(name + "_u", _l2_normalize(_jt.randn(h), eps))
            module.register_buffer(name + "_v", _l2_normalize(_jt.randn(wd), eps))
            def _recompute(mod):
                W = getattr(mod, name + "_orig"); Wm = _to_mat(W)
                uu = getattr(mod, name + "_u"); vv = getattr(mod, name + "_v")
                for _ in range(max(1, n_power_iterations)):
                    vv = _l2_normalize(_jt.matmul(Wm.transpose(0, 1), uu), eps)
                    uu = _l2_normalize(_jt.matmul(Wm, vv), eps)
                getattr(mod, name + "_u").update(uu)     # warm-start next forward
                getattr(mod, name + "_v").update(vv)
                sigma = _jt.matmul(uu.reshape(1, -1), _jt.matmul(Wm, vv.reshape(-1, 1)))
                neww = W / sigma                          # sigma is 1-element -> scalar divide
                neww.persistent = False
                setattr(mod, name, neww)
            _ensure_reparam_hook(module).append(_recompute)
            _recompute(module)
            return module

        _u.weight_norm = weight_norm
        _u.remove_weight_norm = remove_weight_norm
        _u.spectral_norm = spectral_norm

        # --- nn.utils.rnn.pad_sequence ---
        import types as _trnn
        _rnn = _trnn.ModuleType("torch.nn.utils.rnn")
        def pad_sequence(sequences, batch_first=False, padding_value=0.0):
            seqs = list(sequences)
            max_len = max(int(s.shape[0]) for s in seqs)
            trailing = tuple(seqs[0].shape[1:])
            out = []
            for s in seqs:
                pl = max_len - int(s.shape[0])
                if pl > 0:
                    pad = _jt.ones((pl,) + trailing, dtype=s.dtype) * padding_value
                    s = _jt.concat([s, pad], dim=0)
                out.append(s)
            stacked = _jt.stack(out, dim=0)               # (B, T, *)
            return stacked if batch_first else stacked.transpose(0, 1)
        _rnn.pad_sequence = pad_sequence
        _u.rnn = _rnn
        import sys as _sysrnn
        _sysrnn.modules.setdefault("torch.nn.utils.rnn", _rnn)

        nn.utils = _u

    # Newer PyTorch exposes torch.nn.utils.parametrize and
    # torch.nn.utils.parametrizations. transformers 4.56 probes
    # nn.utils.parametrizations.weight_norm while remapping checkpoint keys.
    import sys as _sys_nn_utils
    import types as _types_nn_utils
    _u = getattr(nn, "utils", None) or _types_nn_utils.ModuleType("torch.nn.utils")
    _u.__path__ = getattr(_u, "__path__", [])
    _sys_nn_utils.modules.setdefault("torch.nn.utils", _u)
    nn.utils = _u
    if not hasattr(_u, "parametrize"):
        _parametrize = _types_nn_utils.ModuleType("torch.nn.utils.parametrize")
        _parametrize.register_parametrization = lambda module, *a, **k: module
        _parametrize.remove_parametrizations = lambda module, *a, **k: module
        _parametrize.is_parametrized = lambda module, *a, **k: False
        _parametrize.type_before_parametrizations = lambda module: type(module)
        _u.parametrize = _parametrize
        _sys_nn_utils.modules["torch.nn.utils.parametrize"] = _parametrize
    else:
        _sys_nn_utils.modules.setdefault("torch.nn.utils.parametrize", _u.parametrize)
    if not hasattr(_u, "parametrizations"):
        _parametrizations = _types_nn_utils.ModuleType("torch.nn.utils.parametrizations")
        _parametrizations.weight_norm = getattr(_u, "weight_norm", lambda module, name="weight", dim=0: module)
        _parametrizations.spectral_norm = getattr(_u, "spectral_norm", lambda module, *a, **k: module)
        _parametrizations.orthogonal = lambda module, *a, **k: module
        _u.parametrizations = _parametrizations
        _sys_nn_utils.modules["torch.nn.utils.parametrizations"] = _parametrizations
    else:
        _sys_nn_utils.modules.setdefault("torch.nn.utils.parametrizations", _u.parametrizations)

    # torchmetrics imports torch.nn.utils.rnn at module import time. Install the
    # module unconditionally because some bootstrap paths create nn.utils before
    # the clip/weight-norm block above runs.
    import builtins as _builtins_rnn
    import collections as _collections_rnn
    _rnn = getattr(_u, "rnn", None)
    if _rnn is None:
        _rnn = _types_nn_utils.ModuleType("torch.nn.utils.rnn")

    def _rnn_lengths_to_list(lengths):
        if isinstance(lengths, _jt.Var):
            lengths = lengths.numpy()
        if hasattr(lengths, "tolist"):
            lengths = lengths.tolist()
        if isinstance(lengths, (_builtins_rnn.int, _builtins_rnn.float)):
            lengths = [lengths]
        return [_builtins_rnn.int(x) for x in list(lengths)]

    def _rnn_index_tensor(x, order, batch_first):
        order = _rnn_lengths_to_list(order)
        if not order:
            return x
        if batch_first:
            return _jt.stack([x[i] for i in order], dim=0)
        return _jt.stack([x[:, i] for i in order], dim=1)

    def _rnn_pad_sequence(sequences, batch_first=False, padding_value=0.0):
        seqs = list(sequences)
        if not seqs:
            raise ValueError("pad_sequence expects a non-empty sequence list")
        max_len = _builtins_rnn.max(_builtins_rnn.int(s.shape[0]) for s in seqs)
        trailing = tuple(seqs[0].shape[1:])
        padded = []
        for s in seqs:
            pad_len = max_len - _builtins_rnn.int(s.shape[0])
            if pad_len > 0:
                pad = _jt.ones((pad_len,) + trailing, dtype=s.dtype) * padding_value
                s = _jt.concat([s, pad], dim=0)
            padded.append(s)
        out = _jt.stack(padded, dim=0)
        return out if batch_first else out.transpose(0, 1)

    _PackedSequenceBase = _collections_rnn.namedtuple(
        "PackedSequence", ("data", "batch_sizes", "sorted_indices", "unsorted_indices"))

    class PackedSequence(_PackedSequenceBase):
        __slots__ = ()

        def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
            return _PackedSequenceBase.__new__(cls, data, batch_sizes, sorted_indices, unsorted_indices)

        def to(self, *args, **kwargs):
            data = self.data.to(*args, **kwargs) if hasattr(self.data, "to") else self.data
            return type(self)(data, self.batch_sizes, self.sorted_indices, self.unsorted_indices)

        cuda = to
        cpu = to

    def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
        lengths_list = _rnn_lengths_to_list(lengths)
        if not enforce_sorted:
            order = sorted(range(len(lengths_list)), key=lambda i: lengths_list[i], reverse=True)
            unsorted = [0] * len(order)
            for sorted_pos, original_pos in enumerate(order):
                unsorted[original_pos] = sorted_pos
            input = _rnn_index_tensor(input, order, batch_first)
            lengths_list = [lengths_list[i] for i in order]
            sorted_indices = _jt.array(order).int64()
            unsorted_indices = _jt.array(unsorted).int64()
        else:
            sorted_indices = None
            unsorted_indices = None

        max_len = lengths_list[0] if lengths_list else 0
        pieces = []
        batch_sizes = []
        for t in range(max_len):
            active = _builtins_rnn.sum(1 for n in lengths_list if n > t)
            if active <= 0:
                break
            batch_sizes.append(active)
            if batch_first:
                pieces.append(input[:active, t])
            else:
                pieces.append(input[t, :active])
        if pieces:
            data = _jt.concat(pieces, dim=0)
        else:
            trailing = tuple(input.shape[2:])
            data = _jt.ones((0,) + trailing, dtype=input.dtype)
        return PackedSequence(data, _jt.array(batch_sizes).int64(), sorted_indices, unsorted_indices)

    def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
        if not isinstance(sequence, PackedSequence):
            return sequence, None
        batch_sizes = _rnn_lengths_to_list(sequence.batch_sizes)
        max_len = len(batch_sizes)
        batch_size = _builtins_rnn.max(batch_sizes) if batch_sizes else 0
        data = sequence.data
        trailing = tuple(data.shape[1:])
        steps = []
        offset = 0
        for active in batch_sizes:
            step = data[offset:offset + active]
            offset += active
            if active < batch_size:
                pad = _jt.ones((batch_size - active,) + trailing, dtype=data.dtype) * padding_value
                step = _jt.concat([step, pad], dim=0)
            steps.append(step)
        if steps:
            out = _jt.stack(steps, dim=0)
        else:
            out = _jt.ones((0, batch_size) + trailing, dtype=data.dtype) * padding_value
        if total_length is not None:
            total_length = _builtins_rnn.int(total_length)
            if total_length < max_len:
                raise ValueError("total_length must be at least the packed sequence length")
            if total_length > max_len:
                pad = _jt.ones((total_length - max_len, batch_size) + trailing, dtype=data.dtype) * padding_value
                out = _jt.concat([out, pad], dim=0)
        lengths_list = [_builtins_rnn.sum(1 for n in batch_sizes if n > i) for i in range(batch_size)]
        if sequence.unsorted_indices is not None:
            out = _rnn_index_tensor(out, sequence.unsorted_indices, batch_first=False)
            order = _rnn_lengths_to_list(sequence.unsorted_indices)
            lengths_list = [lengths_list[i] for i in order]
        if batch_first:
            out = out.transpose(0, 1)
        return out, _jt.array(lengths_list).int64()

    _rnn.pad_sequence = _rnn_pad_sequence
    _rnn.pack_padded_sequence = pack_padded_sequence
    _rnn.pad_packed_sequence = pad_packed_sequence
    _rnn.PackedSequence = PackedSequence
    _u.rnn = _rnn
    _sys_nn_utils.modules["torch.nn.utils.rnn"] = _rnn

    if "torch.nn.utils.prune" not in _sys_nn_utils.modules:
        _prune = _types_nn_utils.ModuleType("torch.nn.utils.prune")

        def _unsupported_prune(*args, **kwargs):
            raise NotImplementedError("torch.nn.utils.prune is not supported on jittor backend")

        class BasePruningMethod:
            PRUNING_TYPE = "unstructured"

            def __call__(self, module, inputs):
                return inputs

            @classmethod
            def apply(cls, module, name, *args, **kwargs):
                return _unsupported_prune(module, name, *args, **kwargs)

            def remove(self, module):
                return module

        class L1Unstructured(BasePruningMethod):
            PRUNING_TYPE = "unstructured"

        class RandomUnstructured(BasePruningMethod):
            PRUNING_TYPE = "unstructured"

        class LnStructured(BasePruningMethod):
            PRUNING_TYPE = "structured"

        class RandomStructured(BasePruningMethod):
            PRUNING_TYPE = "structured"

        _prune.BasePruningMethod = BasePruningMethod
        _prune.L1Unstructured = L1Unstructured
        _prune.RandomUnstructured = RandomUnstructured
        _prune.LnStructured = LnStructured
        _prune.RandomStructured = RandomStructured
        _prune.l1_unstructured = _unsupported_prune
        _prune.random_unstructured = _unsupported_prune
        _prune.ln_structured = _unsupported_prune
        _prune.random_structured = _unsupported_prune
        _prune.global_unstructured = _unsupported_prune
        _prune.remove = _unsupported_prune
        _prune.is_pruned = lambda module: False
        _sys_nn_utils.modules["torch.nn.utils.prune"] = _prune
    _u.prune = _sys_nn_utils.modules["torch.nn.utils.prune"]
    if "torch.nn.utils._named_member_accessor" not in _sys_nn_utils.modules:
        _named_accessor = _types_nn_utils.ModuleType("torch.nn.utils._named_member_accessor")
        def _resolve_parent(module, name):
            parts = str(name).split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            return parent, parts[-1]
        def swap_tensor(module, name, tensor):
            parent, leaf = _resolve_parent(module, name)
            old = getattr(parent, leaf, None)
            setattr(parent, leaf, tensor)
            return old
        _named_accessor.swap_tensor = swap_tensor
        _sys_nn_utils.modules["torch.nn.utils._named_member_accessor"] = _named_accessor
    _u._named_member_accessor = _sys_nn_utils.modules["torch.nn.utils._named_member_accessor"]

    if not hasattr(nn, "Hardswish"):
        class Hardswish(nn.Module):
            def execute(self, x):
                return x * _jt.clamp(x + 3, 0, 6) / 6
        nn.Hardswish = Hardswish
    if not hasattr(nn, "CELU"):           # timm uses nn.CELU
        class CELU(nn.Module):
            def __init__(self, alpha=1.0, inplace=False):
                super().__init__(); self.alpha = alpha
            def execute(self, x):
                a = self.alpha
                return _jt.maximum(x, 0.0) + _jt.minimum(0.0, a * (_jt.exp(x / a) - 1))
        nn.CELU = CELU
    # A batch of standard torch activations jittor.nn may lack (timm's act-layer
    # registry references all of them at import). All are pure elementwise.
    if not hasattr(nn, "SELU"):
        _SELU_S, _SELU_A = 1.0507009873554805, 1.6732632423543772
        class SELU(nn.Module):
            def __init__(self, inplace=False): super().__init__()
            def execute(self, x):
                return _SELU_S * (_jt.maximum(x, 0.0) + _jt.minimum(0.0, _SELU_A * (_jt.exp(x) - 1)))
        nn.SELU = SELU
    if not hasattr(nn, "Softsign"):
        class Softsign(nn.Module):
            def execute(self, x): return x / (1 + _jt.abs(x))
        nn.Softsign = Softsign
    if not hasattr(nn, "Tanhshrink"):
        class Tanhshrink(nn.Module):
            def execute(self, x): return x - _jt.tanh(x)
        nn.Tanhshrink = Tanhshrink
    if not hasattr(nn, "Softplus"):
        class Softplus(nn.Module):
            def __init__(self, beta=1, threshold=20): super().__init__(); self.beta=beta; self.threshold=threshold
            def execute(self, x):
                bx = self.beta * x
                return _jt.ternary(bx > self.threshold, x, _jt.log1p(_jt.exp(bx)) / self.beta)
        nn.Softplus = Softplus
    if not hasattr(nn, "Hardshrink"):
        class Hardshrink(nn.Module):
            def __init__(self, lambd=0.5): super().__init__(); self.lambd=lambd
            def execute(self, x): return x * ((x > self.lambd) | (x < -self.lambd)).float()
        nn.Hardshrink = Hardshrink
    if not hasattr(nn, "Softshrink"):
        class Softshrink(nn.Module):
            def __init__(self, lambd=0.5): super().__init__(); self.lambd=lambd
            def execute(self, x):
                l = self.lambd
                return _jt.maximum(x - l, 0.0) - _jt.maximum(-x - l, 0.0)
        nn.Softshrink = Softshrink
    if not hasattr(nn, "Hardsigmoid"):
        class Hardsigmoid(nn.Module):
            def execute(self, x):
                return _jt.clamp(x + 3, 0, 6) / 6
        nn.Hardsigmoid = Hardsigmoid
    if not hasattr(nn, "Identity"):
        class Identity(nn.Module):
            def __init__(self, *a, **k): super().__init__()
            def execute(self, x): return x
        nn.Identity = Identity
    # ModuleList/Sequential/ModuleDict usually exist; alias ParameterList if not
    if not hasattr(nn, "ParameterList"):
        nn.ParameterList = nn.ModuleList if hasattr(nn, "ModuleList") else list
    # ModuleDict (peft LoRA layers need it); jittor lacks it.
    if not hasattr(nn, "ModuleDict"):
        class ModuleDict(nn.Module):
            def __init__(self, modules=None):
                super().__init__()
                self._keys = []
                if modules:
                    self.update(modules)
            def update(self, modules):
                items = modules.items() if hasattr(modules, "items") else modules
                for k, v in items:
                    self[k] = v
            def __setitem__(self, key, module):
                setattr(self, key, module)
                if key not in self._keys:
                    self._keys.append(key)
            def __getitem__(self, key):
                return getattr(self, key)
            def __delitem__(self, key):
                delattr(self, key)
                if key in self._keys:
                    self._keys.remove(key)
            def __contains__(self, key):
                return key in self._keys
            def __len__(self):
                return len(self._keys)
            def __iter__(self):
                return iter(self._keys)
            def keys(self):
                return list(self._keys)
            def values(self):
                return [getattr(self, k) for k in self._keys]
            def items(self):
                return [(k, getattr(self, k)) for k in self._keys]
            def pop(self, key):
                v = getattr(self, key); self.__delitem__(key); return v
        nn.ModuleDict = ModuleDict

    # Layer classes torch has that jittor.nn may lack -- needed at least for
    # isinstance() checks in model init. Provide a distinct empty subclass so
    # isinstance discrimination still works.
    if not hasattr(nn, "ConvTranspose1d"):
        class ConvTranspose1d(nn.Module):
            # Real 1D transpose-conv (SABL's side_aware_feature_extractor uses it),
            # implemented via conv_transpose2d with a unit height dim so it also
            # rides the cuDNN memory-efficient path.
            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, output_padding=0, groups=1, bias=True,
                         dilation=1, **k):
                super().__init__()
                import jittor as _jt2, math as _math
                g1 = lambda v: v[0] if isinstance(v, (tuple, list)) else v
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = g1(kernel_size)
                self.stride = g1(stride)
                self.padding = g1(padding)
                self.output_padding = g1(output_padding)
                self.dilation = g1(dilation)
                self.groups = groups
                self.weight = _jt2.init.invariant_uniform(
                    [in_channels, out_channels // groups, self.kernel_size], dtype="float")
                if bias:
                    fan = (in_channels // groups) * self.kernel_size
                    bound = 1.0 / _math.sqrt(fan) if fan > 0 else 0.0
                    self.bias = _jt2.init.uniform([out_channels], "float", -bound, bound)
                else:
                    self.bias = None
            def execute(self, x):
                import jittor as _jt2
                x2 = x.unsqueeze(2)                       # (N,Cin,1,L)
                w2 = self.weight.unsqueeze(2)             # (Cin,Cout/g,1,K)
                y = _jt2.nn.conv_transpose2d(
                    x2, w2, None, (1, self.stride), (0, self.padding),
                    (0, self.output_padding), self.groups, (1, self.dilation))
                y = y.squeeze(2)                          # (N,Cout,Lout)
                if self.bias is not None:
                    y = y + self.bias.broadcast(y.shape, [0, 2])
                return y
        nn.ConvTranspose1d = ConvTranspose1d
    if not hasattr(nn, "RMSNorm"):
        class RMSNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **k):
                super().__init__()
                import jittor as _jt2
                if isinstance(normalized_shape, int):
                    normalized_shape = (normalized_shape,)
                self.normalized_shape = tuple(normalized_shape)
                self.eps = eps
                self.weight = _jt2.ones(normalized_shape) if elementwise_affine else None
            def execute(self, x):
                import jittor as _jt2
                v = (x.float32() ** 2).mean(-1, keepdims=True)
                x = x * _jt2.rsqrt(v + self.eps)
                return x * self.weight if self.weight is not None else x
        nn.RMSNorm = RMSNorm
    # nn.MultiheadAttention was an empty stub (no params, no execute -> raised
    # NotImplementedError). Implement it over the existing functional
    # multi_head_attention_forward. Plus nn.TransformerEncoderLayer/Encoder/
    # DecoderLayer/Decoder/Transformer which build on it (used by some models and by
    # users building transformers directly).
    import jittor as _jtm
    if (not hasattr(nn, "MultiheadAttention")) or not hasattr(nn.MultiheadAttention, "execute") \
            or getattr(nn.MultiheadAttention.execute, "__qualname__", "").endswith("Module.execute"):
        class MultiheadAttention(nn.Module):
            def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                         add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                         batch_first=False, device=None, dtype=None):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.dropout = dropout
                self.batch_first = batch_first
                self.head_dim = embed_dim // num_heads
                self.add_zero_attn = add_zero_attn
                self.in_proj_weight = _jtm.init.invariant_uniform((3 * embed_dim, embed_dim), "float32")
                self.in_proj_bias = _jtm.zeros((3 * embed_dim,)) if bias else None
                self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
                self.bias_k = _jtm.init.invariant_uniform((1, 1, embed_dim), "float32") if add_bias_kv else None
                self.bias_v = _jtm.init.invariant_uniform((1, 1, embed_dim), "float32") if add_bias_kv else None

            def execute(self, query, key, value, key_padding_mask=None, need_weights=True,
                        attn_mask=None, average_attn_weights=True, is_causal=False):
                if self.batch_first:
                    query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
                out, w = nn.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
                    self.add_zero_attn, self.dropout if self.is_training() else 0.0,
                    self.out_proj.weight, self.out_proj.bias, training=self.is_training(),
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
                if self.batch_first:
                    out = out.transpose(0, 1)
                return out, w
        nn.MultiheadAttention = MultiheadAttention

    def _act_fn(activation):
        if callable(activation):
            return activation
        return {"relu": nn.relu, "gelu": nn.gelu}.get(activation, nn.relu)

    if not hasattr(nn, "TransformerEncoderLayer"):
        class TransformerEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                         activation="relu", layer_norm_eps=1e-5, batch_first=False,
                         norm_first=False, bias=True, device=None, dtype=None):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       batch_first=batch_first, bias=bias)
                self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
                self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
                self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm_first = norm_first
                self.activation = _act_fn(activation)

            def _sa(self, x, attn_mask, kpm, is_causal):
                return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=kpm,
                                      need_weights=False, is_causal=is_causal)[0]

            def _ff(self, x):
                return self.linear2(self.activation(self.linear1(x)))

            def execute(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                x = src
                if self.norm_first:
                    x = x + self._sa(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
                    x = x + self._ff(self.norm2(x))
                else:
                    x = self.norm1(x + self._sa(x, src_mask, src_key_padding_mask, is_causal))
                    x = self.norm2(x + self._ff(x))
                return x
        nn.TransformerEncoderLayer = TransformerEncoderLayer

    if not hasattr(nn, "TransformerEncoder"):
        import copy as _copy
        class TransformerEncoder(nn.Module):
            def __init__(self, encoder_layer, num_layers, norm=None, **kw):
                super().__init__()
                self.layers = nn.ModuleList([_copy.deepcopy(encoder_layer) for _ in range(num_layers)])
                self.num_layers = num_layers
                self.norm = norm

            def execute(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
                out = src
                for layer in self.layers:
                    out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                if self.norm is not None:
                    out = self.norm(out)
                return out
        nn.TransformerEncoder = TransformerEncoder

    if not hasattr(nn, "TransformerDecoderLayer"):
        class TransformerDecoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                         activation="relu", layer_norm_eps=1e-5, batch_first=False,
                         norm_first=False, bias=True, device=None, dtype=None):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                       batch_first=batch_first, bias=bias)
                self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                            batch_first=batch_first, bias=bias)
                self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
                self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
                self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
                self.norm_first = norm_first
                self.activation = _act_fn(activation)

            def _sa(self, x, m, kpm, ic):
                return self.self_attn(x, x, x, attn_mask=m, key_padding_mask=kpm,
                                      need_weights=False, is_causal=ic)[0]

            def _ca(self, x, mem, m, kpm, ic):
                return self.multihead_attn(x, mem, mem, attn_mask=m, key_padding_mask=kpm,
                                           need_weights=False, is_causal=ic)[0]

            def _ff(self, x):
                return self.linear2(self.activation(self.linear1(x)))

            def execute(self, tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        tgt_is_causal=False, memory_is_causal=False):
                x = tgt
                if self.norm_first:
                    x = x + self._sa(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                    x = x + self._ca(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
                    x = x + self._ff(self.norm3(x))
                else:
                    x = self.norm1(x + self._sa(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
                    x = self.norm2(x + self._ca(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
                    x = self.norm3(x + self._ff(x))
                return x
        nn.TransformerDecoderLayer = TransformerDecoderLayer

    if not hasattr(nn, "TransformerDecoder"):
        import copy as _copy2
        class TransformerDecoder(nn.Module):
            def __init__(self, decoder_layer, num_layers, norm=None, **kw):
                super().__init__()
                self.layers = nn.ModuleList([_copy2.deepcopy(decoder_layer) for _ in range(num_layers)])
                self.num_layers = num_layers
                self.norm = norm

            def execute(self, tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        tgt_is_causal=None, memory_is_causal=False):
                out = tgt
                for layer in self.layers:
                    out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                memory_is_causal=memory_is_causal)
                if self.norm is not None:
                    out = self.norm(out)
                return out
        nn.TransformerDecoder = TransformerDecoder

    if not hasattr(nn, "Transformer"):
        class Transformer(nn.Module):
            def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                         num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                         activation="relu", custom_encoder=None, custom_decoder=None,
                         layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                         bias=True, device=None, dtype=None):
                super().__init__()
                self.batch_first = batch_first
                self.d_model = d_model
                self.nhead = nhead
                if custom_encoder is not None:
                    self.encoder = custom_encoder
                else:
                    el = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first, bias)
                    self.encoder = nn.TransformerEncoder(el, num_encoder_layers,
                                                         nn.LayerNorm(d_model, eps=layer_norm_eps))
                if custom_decoder is not None:
                    self.decoder = custom_decoder
                else:
                    dl = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first, bias)
                    self.decoder = nn.TransformerDecoder(dl, num_decoder_layers,
                                                         nn.LayerNorm(d_model, eps=layer_norm_eps))

            def execute(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                        src_key_padding_mask=None, tgt_key_padding_mask=None,
                        memory_key_padding_mask=None, src_is_causal=None,
                        tgt_is_causal=None, memory_is_causal=False):
                memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
                return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    memory_is_causal=memory_is_causal)

            @staticmethod
            def generate_square_subsequent_mask(sz, device=None, dtype=None):
                # upper-triangular -inf mask (additive), like torch
                m = _jtm.triu(_jtm.ones((sz, sz)), 1) * (-1e30)
                return m
        nn.Transformer = Transformer

    # ---- nn.SyncBatchNorm (single-device: behaves exactly like BatchNorm) ----
    # mmdetection's rtmdet calls `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`;
    # with no real process group there is nothing to synchronise, so the convert
    # entry returns the model unchanged (BN already pools over the whole batch here).
    if not hasattr(nn, "SyncBatchNorm"):
        class SyncBatchNorm(nn.BatchNorm):
            def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                         track_running_stats=True, process_group=None, **kw):
                super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
            @classmethod
            def convert_sync_batchnorm(cls, module, process_group=None):
                return module
        nn.SyncBatchNorm = SyncBatchNorm

    # ---- nn.functional extras used by mmdetection (not auto-copied from jittor.nn) ----
    F = nn.functional
    if not hasattr(F, "_Reduction"):
        # torch's private reduction-string -> enum helper; mmdet's loss utils do
        # `F._Reduction.get_enum(reduction)` then branch 0/1/2 = none/mean/sum.
        class _Reduction:
            @staticmethod
            def get_enum(reduction):
                return {"none": 0, "mean": 1, "elementwise_mean": 1,
                        "sum": 2}.get(reduction, 1)
            @staticmethod
            def legacy_get_string(size_average, reduce, emit_warning=True):
                sa = True if size_average is None else size_average
                rd = True if reduce is None else reduce
                if not rd:
                    return "none"
                return "mean" if sa else "sum"
        F._Reduction = _Reduction
    if not hasattr(F, "adaptive_max_pool2d"):
        def _adaptive_max_pool2d(input, output_size, return_indices=False):
            out = nn.AdaptiveMaxPool2d(output_size)(input)
            return (out, None) if return_indices else out
        F.adaptive_max_pool2d = _adaptive_max_pool2d
    if not getattr(F, "_torch_linear_wrapped", False):
        # torch's F.linear accepts a 1-D weight (matrix-vector product), e.g. GFL's
        # Integral does F.linear(x[N,K], project[K]) -> [N]; jittor's linear asserts 2-D.
        _jt_linear = F.linear
        def _linear(input, weight, bias=None):
            if hasattr(weight, "ndim") and weight.ndim == 1:
                out = (input * weight).sum(-1)
                return out if bias is None else out + bias
            return _jt_linear(input, weight, bias)
        F.linear = _linear
        F._torch_linear_wrapped = True
    if not hasattr(F, "relu_"):
        F.relu_ = lambda input: nn.relu(input)   # in-place relu (graph-equivalent)
    if not hasattr(F, "upsample_bilinear"):
        # deprecated torch alias == interpolate(mode='bilinear', align_corners=True)
        def _upsample_bilinear(input, size=None, scale_factor=None):
            return F.interpolate(input, size=size, scale_factor=scale_factor,
                                 mode="bilinear", align_corners=True)
        F.upsample_bilinear = _upsample_bilinear
    if not hasattr(F, "upsample"):
        def _upsample(input, size=None, scale_factor=None, mode="nearest",
                      align_corners=None):
            return F.interpolate(input, size=size, scale_factor=scale_factor,
                                 mode=mode, align_corners=align_corners)
        F.upsample = _upsample

    # torch's nn.Conv2d exposes .transposed / .output_padding (torchvision &
    # mmcv's ConvModule read them to introspect the layer); jittor's Conv lacks
    # them. Add torch-compatible class attributes.
    for _cn in ("Conv", "Conv1d", "Conv3d"):
        _c = getattr(nn, _cn, None)
        if _c is not None:
            if not hasattr(_c, "transposed"):
                _c.transposed = False
            if not hasattr(_c, "output_padding"):
                _c.output_padding = (0, 0)
    for _cn in ("ConvTranspose", "ConvTranspose1d", "ConvTranspose3d"):
        _c = getattr(nn, _cn, None)
        if _c is not None:
            _c.transposed = True
            if not hasattr(_c, "output_padding"):
                _c.output_padding = (0, 0)

    # torch's nn.Dropout/Dropout2d/Dropout3d take an `inplace` kwarg that jittor's
    # don't (DETR-family configs pass dropout=dict(..., inplace=...)). Make the
    # constructors tolerate (and ignore) it.
    for _dn in ("Dropout", "Dropout2d", "Dropout3d"):
        _dc = getattr(nn, _dn, None)
        if _dc is not None and not getattr(_dc, "_torch_inplace_patched", False):
            def _mk_drop_init(orig):
                def _init(self, p=0.5, inplace=False, *a, **k):
                    orig(self, p, *a, **k)
                return _init
            _dc.__init__ = _mk_drop_init(_dc.__init__)
            _dc._torch_inplace_patched = True

    # jittor names several activation/layer classes lowercase or snake_case
    # (nn.ReLU.__name__ == 'relu'); torch code and mmcv's registry key layers by
    # type(layer).__name__, so normalize them to the torch class names.
    _TORCH_CLASS_NAMES = [
        "ReLU", "ReLU6", "LeakyReLU", "PReLU", "RReLU", "ELU", "CELU", "SELU",
        "GELU", "SiLU", "Mish", "Sigmoid", "Tanh", "Softmax", "Softplus",
        "Hardswish", "Hardsigmoid", "Hardtanh", "GLU", "Identity",
    ]
    for _nm in _TORCH_CLASS_NAMES:
        _cls = getattr(nn, _nm, None)
        if isinstance(_cls, type) and getattr(_cls, "__name__", None) != _nm:
            try:
                _cls.__name__ = _nm
                _cls.__qualname__ = _nm
            except Exception:
                pass

    _install_module_methods(nn)


def _install_module_methods(nn):
    """Add torch-compatible methods to jittor's nn.Module."""
    import jittor as _jt
    M = nn.Module

    # torch models define forward(); jittor calls execute(). Make the base
    # execute() delegate to a subclass-defined forward() so torch models run.
    _orig_execute = M.execute
    def _execute(self, *args, **kwargs):
        fwd = getattr(type(self), "forward", None)
        if fwd is not None and fwd is not _forward_alias:
            return fwd(self, *args, **kwargs)
        return _orig_execute(self, *args, **kwargs)
    def _forward_alias(self, *args, **kwargs):
        # if a subclass only defines execute(), forward() routes to it
        return self.execute(*args, **kwargs)
    M.execute = _execute
    if not hasattr(M, "forward"):
        M.forward = _forward_alias

    # Central dispatch fix: an HF module may SUBCLASS a jittor builtin (e.g.
    # transformers OPTLearnedPositionalEmbedding(nn.Embedding)) and override
    # forward() with a different signature. The builtin (Embedding) defines its
    # own execute(), which MRO-shadows the patched base Module.execute above, so
    # `module(...)` -> __call__ -> self.execute(...) lands on the builtin's
    # execute() and never sees the subclass forward() -> TypeError.
    #
    # Decide per class whether the OWN forward() override should take precedence
    # over the inherited builtin execute(): it should iff a real (non-alias)
    # forward() is defined at an MRO position at least as derived as the nearest
    # execute(). Conservative: classes that only define execute() (every native
    # jittor module + jittor-native subclasses of builtins) keep calling
    # execute() exactly as before; only a genuine, more-derived forward()
    # override flips dispatch.
    _dispatch_cache = {}
    def _prefer_forward(cls):
        cached = _dispatch_cache.get(cls)
        if cached is not None:
            return cached
        fwd_idx = exec_idx = None
        for i, c in enumerate(cls.__mro__):
            d = c.__dict__
            if fwd_idx is None and "forward" in d and d["forward"] is not _forward_alias:
                fwd_idx = i
            if exec_idx is None and "execute" in d and d["execute"] is not _execute:
                exec_idx = i
        # forward() wins only if it exists and is no less derived than execute()
        result = fwd_idx is not None and (exec_idx is None or fwd_idx <= exec_idx)
        _dispatch_cache[cls] = result
        return result

    _orig_call = M.__call__
    def _call(self, *args, **kwargs):
        # torch lets a module override forward per-INSTANCE (`self.forward = fn`,
        # used by vLLM's samplers / CustomOp dispatch). Honor an instance-level
        # forward before the class-level dispatch.
        inst_fwd = self.__dict__.get("forward", None)
        if inst_fwd is not None and callable(inst_fwd):
            return inst_fwd(*args, **kwargs)
        if _prefer_forward(type(self)):
            return type(self).forward(self, *args, **kwargs)
        return _orig_call(self, *args, **kwargs)
    M.__call__ = _call

    # torch's named_parameters/named_buffers/named_modules accept extra kwargs
    # (remove_duplicate, prefix, recurse) and return iterators; jittor's take
    # only `recurse` and return lists, with named_buffers defaulting recurse=
    # False (torch defaults True). Wrap to be torch-compatible.
    _orig_named_parameters = M.named_parameters
    _orig_named_buffers = M.named_buffers
    _orig_named_modules = M.named_modules

    def _named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        reg = getattr(jt, "_torch_leaf_params", None)
        if reg is None:
            reg = jt._torch_leaf_params = {}
        seen = set()
        for name, v in _orig_named_parameters(self, recurse=recurse):
            if remove_duplicate and id(v) in seen:
                continue
            seen.add(id(v))
            # register trainable params as autograd leaves so the no-optimizer
            # loss.backward() path can populate their .grad (see parameters()).
            try:
                if isinstance(v, jt.Var) and not v.is_stop_grad():
                    reg[id(v)] = v
            except Exception:
                pass
            yield (prefix + ("." if prefix else "") + name, v)
    M.named_parameters = _named_parameters

    def _named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for name, v in _orig_named_buffers(self, recurse=recurse):
            if remove_duplicate and id(v) in seen:
                continue
            seen.add(id(v))
            yield (prefix + ("." if prefix else "") + name, v)
    M.named_buffers = _named_buffers

    def _named_modules(self, memo=None, prefix="", remove_duplicate=True):
        for item in _orig_named_modules(self):
            # jittor yields (name, module) pairs
            if isinstance(item, tuple) and len(item) == 2:
                name, mod = item
            else:
                name, mod = "", item
            yield (prefix + ("." if prefix and name else "") + name, mod)
    M.named_modules = _named_modules

    # torch's Module.load_state_dict(state, strict=True, assign=False) accepts a
    # `strict` kwarg and returns a namedtuple(missing_keys, unexpected_keys);
    # jittor's takes only `params` and returns None. Wrap for torch callers
    # (peft's set_peft_model_state_dict passes strict=False).
    _orig_load_state_dict = M.load_state_dict
    import collections as _collections2
    _IncompatibleKeys = _collections2.namedtuple("IncompatibleKeys",
                                                  ["missing_keys", "unexpected_keys"])
    def _find_state_target(root, key):
        obj = root
        for part in str(key).split("."):
            if isinstance(obj, nn.Sequential):
                if part in obj.layers:
                    obj = obj.layers[part]
                elif str(part).isdigit() and int(part) in obj.layers:
                    obj = obj.layers[int(part)]
                else:
                    return None
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    def _state_source_to_var(value):
        if isinstance(value, jt.Var):
            return value
        try:
            return jt.array(value.cpu().detach().numpy())
        except Exception:
            return jt.array(value)

    def _preserve_target_dtypes_for_load(root, state_dict):
        # torch.load_state_dict(assign=False), the default used by TRELLIS.2,
        # copies checkpoint values into existing parameters/buffers and keeps
        # the destination dtype.  Jittor's native load replaces through update(),
        # so a bf16 target can be widened to fp32 when the loader had to widen a
        # BF16 safetensor through numpy. Cast the source to the live target dtype
        # before delegating to native load_state_dict.
        if not isinstance(state_dict, dict):
            return state_dict
        converted = None
        for key, value in state_dict.items():
            target = _find_state_target(root, key)
            if not isinstance(target, jt.Var):
                continue
            src = _state_source_to_var(value)
            if not isinstance(src, jt.Var):
                continue
            if src.shape != target.shape:
                continue
            target_dtype = str(target.dtype)
            if str(src.dtype) == target_dtype:
                continue
            if converted is None:
                converted = dict(state_dict)
            converted[key] = src.cast(target_dtype)
        return state_dict if converted is None else converted

    def _load_state_dict(self, state_dict, strict=True, assign=False):
        # preserve trainable flags: jittor assign can flip stop_grad
        trainable = set()
        try:
            for n, p in self.named_parameters():
                if not p.is_stop_grad():
                    trainable.add(n)
        except Exception:
            pass
        load_state = state_dict if assign else _preserve_target_dtypes_for_load(self, state_dict)
        _orig_load_state_dict(self, load_state)
        try:
            for n, p in self.named_parameters():
                if n in trainable and p.is_stop_grad():
                    p.start_grad()
        except Exception:
            pass
        return _IncompatibleKeys([], [])
    M.load_state_dict = _load_state_dict

    # torch's Module.parameters() returns an *iterator*; peft does
    # `next(model.parameters())`. jittor returns a list (needed for len()/
    # indexing by optimizers). Return a list subclass that is also an iterator
    # so both `next(...)` and `len(...)`/indexing work.
    class _ParamList(list):
        def __iter__(self):
            return list.__iter__(self)
        def __next__(self):
            it = getattr(self, "_it", None)
            if it is None:
                it = self._it = list.__iter__(self)
            return next(it)
    # Register every trainable parameter as an autograd "leaf" the first time a
    # module's params are enumerated. torch code reads param.grad only after
    # enumerating params (optimizer construction, gradient clipping, gradcheck,
    # manual inspection all call parameters()/named_parameters() first), so this
    # is the reliable hook that lets the optimizer-free loss.backward() path
    # (below) populate param.grad. jittor params are trainable-by-default and
    # almost never pass through the requires_grad setter, which is why the prior
    # registry stayed empty (bert: 0/39 grads exposed). Enumeration is also the
    # *leak-safe* hook: only declared parameters are captured -- never transient
    # forward activations, which a Module.__setattr__ hook would wrongly retain
    # and leak one Var per step. Idempotent (id-keyed); skips frozen params so
    # their .grad stays None like torch.
    def _register_leaf_params(params):
        try:
            reg = getattr(jt, "_torch_leaf_params", None)
            if reg is None:
                reg = jt._torch_leaf_params = {}
            for p in params:
                if isinstance(p, jt.Var) and not p.is_stop_grad():
                    reg[id(p)] = p
        except Exception:
            pass
    _orig_parameters = M.parameters
    def _parameters(self, recurse=True):
        pl = _orig_parameters(self, recurse=recurse)
        _register_leaf_params(pl)
        return _ParamList(pl)
    M.parameters = _parameters

    # torch's Module.train(mode=True)/eval() take a mode arg; jittor's train()
    # takes none. Wrap to accept it and toggle jittor's real training flag.
    #
    # The flag that controls layers like Dropout/BatchNorm is `is_train` -- an
    # instance attribute read by Dropout.execute (nn.py). `is_training` is a
    # *method* and `training` a *property*, so they must NEVER be assigned a
    # bool (the old code did `m.is_training = False`, which both shadowed the
    # method and failed to flip the flag the layers actually read). We set
    # `is_train` recursively on every submodule. We deliberately do NOT touch
    # parameter stop-grad state (torch's .eval() leaves requires_grad alone),
    # so this is purely a mode flip with no gradient side effects.
    def _set_is_train(self, mode):
        mode = bool(mode)
        try:
            mods = self.modules() if hasattr(self, "modules") else [self]
        except Exception:
            mods = [self]
        for m in mods:
            try:
                m.is_train = mode
            except Exception:
                pass
    def _train(self, mode=True):
        # torch semantics: set this module's flag, then recurse into DIRECT
        # children calling each child's .train(mode) so overridden train()
        # methods run (e.g. e2cnn's R2Conv.train() rebuilds/discards its cached
        # filter; a flat is_train sweep silently bypasses it, leaving stale or
        # empty filters and zero output). For ordinary modules this is
        # behaviourally identical to the old flat sweep.
        mode = bool(mode)
        try:
            self.is_train = mode
        except Exception:
            pass
        kids = None
        try:
            kids = list(self.children())
        except Exception:
            kids = None
        if kids is None:
            _set_is_train(self, mode)          # fallback: flat sweep
            return self
        for child in kids:
            tr = getattr(child, "train", None)
            if callable(tr):
                try:
                    tr(mode)
                    continue
                except Exception:
                    pass
            _set_is_train(child, mode)
        return self
    M.train = _train
    def _eval(self):
        return _train(self, False)
    M.eval = _eval

    _MODULE_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")

    def _module_cast_var_if_needed(v, ds, copy=False):
        if copy or str(v.dtype) != ds:
            return v.cast(ds)
        return v

    def _module_cast_float_dtype(self, ds):
        if ds is not None and ds in _MODULE_FLOAT_DTYPES:
            for p in self.parameters():
                if p.dtype.is_float() if hasattr(p.dtype, "is_float") else ("float" in str(p.dtype)):
                    new_p = _module_cast_var_if_needed(p, ds)
                    if new_p is not p:
                        p.assign(new_p)
        return self

    def _module_replace_vars(self, convert):
        converted = {}
        try:
            modules = list(self.modules()) if hasattr(self, "modules") else [self]
        except Exception:
            modules = [self]
        if not modules or modules[0] is not self:
            modules.insert(0, self)
        seen = set()
        for module in modules:
            mid = id(module)
            if mid in seen:
                continue
            seen.add(mid)
            attrs = []
            if hasattr(module, "params"):
                attrs.append(("params", getattr(module, "params")))
            attrs.append(("__dict__", getattr(module, "__dict__", {})))
            for _container_name, container in attrs:
                if not isinstance(container, dict):
                    continue
                buffer_names = getattr(module, "_buffer_names", set())
                for name, value in list(container.items()):
                    if isinstance(value, jt.Var):
                        if _container_name == "__dict__":
                            is_public_param = not (isinstance(name, str) and name.startswith("_"))
                            is_buffer = getattr(value, "is_buffer", False) or name in buffer_names
                            if not (is_public_param or is_buffer):
                                continue
                        vid = id(value)
                        if vid in converted:
                            new_value = converted[vid]
                        else:
                            new_value = convert(value)
                            converted[vid] = new_value
                            if new_value is not value:
                                try:
                                    new_value.persistent = getattr(value, "persistent")
                                except Exception:
                                    pass
                                try:
                                    new_value.is_buffer = getattr(value, "is_buffer")
                                except Exception:
                                    pass
                                try:
                                    new_value._torch_grad = getattr(value, "_torch_grad")
                                except Exception:
                                    pass
                                try:
                                    if value.is_stop_grad() and not new_value.is_stop_grad():
                                        new_value.stop_grad()
                                    elif (not value.is_stop_grad()) and new_value.is_stop_grad():
                                        new_value.start_grad()
                                        _torch_register_leaf(new_value)
                                except Exception:
                                    pass
                                try:
                                    reg = getattr(jt, "_torch_leaf_params", None)
                                    if isinstance(reg, dict) and vid in reg:
                                        reg.pop(vid, None)
                                        if not new_value.is_stop_grad():
                                            reg[id(new_value)] = new_value
                                except Exception:
                                    pass
                        if new_value is value:
                            continue
                        container[name] = new_value
        return self

    def _module_to(self, *args, **kwargs):
        # torch Module.to(device/dtype/...) casts floating tensors and migrates
        # tensor residency when an explicit cpu/cuda device is requested.
        ds = None
        dev = kwargs.get("device")
        copy = bool(kwargs.get("copy", False))
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, dtype):
                ds = a.name
            elif isinstance(a, device):
                dev = a
            elif isinstance(a, jt.Var):
                ds = str(a.dtype)
                dev = a.device
            elif isinstance(a, str):
                bare = a.replace("torch.", "")
                if bare in dtype._registry:
                    ds = bare
                elif bare.split(":")[0] in ("cpu", "cuda", "npu"):
                    dev = bare
        if _device_is_cuda(dev):
            jt.flags.use_cuda = 1

        def convert(v):
            out = v
            if ds is not None and ds in _MODULE_FLOAT_DTYPES:
                is_float = v.dtype.is_float() if hasattr(v.dtype, "is_float") else ("float" in str(v.dtype))
                if is_float:
                    out = _module_cast_var_if_needed(out, ds, copy=copy)
            if _device_is_cpu(dev):
                out = _make_cpu_resident(out, inplace=(out is v))
            elif _device_is_cuda(dev):
                out = _make_cuda_resident(out, force=True, inplace=(out is v))
            return out

        if dev is not None or ds is not None:
            return _module_replace_vars(self, convert)
        return self
    M.to = _module_to

    def _module_cuda(self, dev=None):
        return _module_to(self, device("cuda", dev) if isinstance(dev, int) else "cuda")
    def _module_npu(self, dev=None):
        return _module_to(self, device("npu", dev) if isinstance(dev, int) else "npu")
    M.cuda = _module_cuda
    M.npu = _module_npu
    M.cpu = lambda self: _module_to(self, "cpu")
    if not hasattr(M, "float"):
        M.float = lambda self: _module_cast_float_dtype(self, "float32")
    if not hasattr(M, "double"):
        M.double = lambda self: _module_cast_float_dtype(self, "float64")
    if not hasattr(M, "half"):
        M.half = lambda self: _module_cast_float_dtype(self, "float16")
    # torch's zero_grad() clears each param's .grad so the next backward starts
    # fresh; the optimizer-free backward path below accumulates with += (matching
    # torch), so a real reset is required. The prior no-op left grads silently
    # accumulating across steps. Clear the torch-exposed grad and, when an
    # optimizer is bridged, delegate to its zero_grad as well.
    def _zero_grad(self, set_to_none=True):
        try:
            for p in self.parameters():
                if getattr(p, "_torch_grad", None) is not None:
                    object.__setattr__(p, "_torch_grad", None)
        except Exception:
            pass
        opt = getattr(jt, "_current_optimizer", None)
        if opt is not None:
            try:
                opt.zero_grad()
            except Exception:
                pass
        return None
    M.zero_grad = _zero_grad
    if not hasattr(M, "buffers"):
        M.buffers = lambda self, recurse=True: [v for _, v in self.named_buffers()]
    if not hasattr(M, "get_submodule"):
        def _get_submodule(self, target):
            mod = self
            for part in target.split("."):
                if part:
                    mod = getattr(mod, part)
            return mod
        M.get_submodule = _get_submodule
    if not hasattr(M, "get_parameter"):
        def _get_parameter(self, target):
            mod = self
            parts = target.split(".")
            for part in parts[:-1]:
                if part:
                    mod = getattr(mod, part)
            leaf = parts[-1]
            if not hasattr(mod, leaf):
                raise AttributeError(f"`{target}` is not a parameter")
            v = getattr(mod, leaf)
            import jittor as _jtp
            # a parameter is a trainable Var directly attached to the module
            if isinstance(v, _jtp.Var) and not v.is_stop_grad():
                return v
            if isinstance(v, _jtp.Var):
                # could still be a (frozen) parameter; distinguish from buffers
                names = {n for n, _ in self.named_parameters()}
                if target in names:
                    return v
            raise AttributeError(f"`{target}` is not a parameter")
        M.get_parameter = _get_parameter
    if not hasattr(M, "get_buffer"):
        def _get_buffer(self, target):
            mod = self
            parts = target.split(".")
            for part in parts[:-1]:
                if part:
                    mod = getattr(mod, part)
            leaf = parts[-1]
            if not hasattr(mod, leaf):
                raise AttributeError(f"`{target}` is not a buffer")
            v = getattr(mod, leaf)
            import jittor as _jtp
            names = {n for n, _ in self.named_buffers()}
            if isinstance(v, _jtp.Var) and target in names:
                return v
            raise AttributeError(f"`{target}` is not a buffer")
        M.get_buffer = _get_buffer
    if not hasattr(M, "register_parameter"):
        def _register_parameter(self, name, param):
            setattr(self, name, param)
        M.register_parameter = _register_parameter
    if not hasattr(M, "type"):
        M.type = lambda self, dst_type=None: self

    # torch's nn.Module keeps `_non_persistent_buffers_set`, a set of the
    # *immediate* (non-recursive) buffer attribute names that were registered
    # with persistent=False. transformers' from_pretrained reads it via
    # `named_non_persistent_buffers()` (parent._non_persistent_buffers_set).
    # jittor instead tags each buffer Var with `.persistent`; derive the set
    # from that. It's a property so it stays correct as buffers are (de)added.
    if not isinstance(M.__dict__.get("_non_persistent_buffers_set"), property):
        import jittor as _jtb
        def _nonpersist_set(self):
            out = set()
            for k, v in self.__dict__.items():
                if (isinstance(k, str) and not k.startswith("_")
                        and isinstance(v, _jtb.Var)
                        and getattr(v, "is_buffer", False)
                        and not getattr(v, "persistent", True)):
                    out.add(k)
            return out
        M._non_persistent_buffers_set = property(_nonpersist_set)


def _install_init_aliases():
    import jittor.init as _init
    import jittor as _jt2
    # torch-style in-place initializers, tolerant of torch kwargs (e.g.
    # `generator=`, which jittor ignores). Each writes into `tensor` in place.
    def _assign(tensor, value):
        # Preserve the tensor's grad-tracking: jittor's .assign() adopts the
        # source var's stop_grad flag, and our `value` (jt.normal/zeros/...) is
        # stop_grad, which would silently freeze the parameter. Re-enable grad
        # unless the param was explicitly stop-grad before.
        was_trainable = not tensor.is_stop_grad()
        parent = getattr(tensor, "_torch_index_parent", None)
        parent_slices = getattr(tensor, "_torch_index_slices", None)
        tensor.assign(value)
        # Basic indexing materializes a Var in Jittor, while torch initializers
        # mutate a view's underlying storage. Write the initialized value back
        # through the recorded parent chain (TorchQuantum initializes U3 columns
        # via init.constant_(parameter[:, k], value)).
        if isinstance(parent, _jt2.Var):
            parent[parent_slices] = value
        if was_trainable:
            tensor.start_grad()
        return tensor

    # in-place inits are sometimes called on a NON-Var constant: jittor represents a
    # disabled affine term (e.g. LayerNorm(bias=False) -> self.bias = 0.0) as a Python
    # scalar, and a model's _init_weights may still call init.zeros_(module.bias) on it.
    # Such a constant isn't a learnable parameter, so initializing it is a no-op.
    def _not_var(t):
        return not isinstance(t, _jt2.Var)
    def normal_(tensor, mean=0.0, std=1.0, generator=None):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.normal(float(mean), float(std), tensor.shape).cast(str(tensor.dtype)))
    def uniform_(tensor, a=0.0, b=1.0, generator=None):
        if _not_var(tensor): return tensor
        return _assign(tensor, (_jt2.rand(tensor.shape) * (b - a) + a).cast(str(tensor.dtype)))
    def zeros_(tensor):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.zeros(tensor.shape, tensor.dtype))
    def ones_(tensor):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype))
    def constant_(tensor, val):
        if _not_var(tensor): return tensor
        return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype) * val)
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
        if _not_var(tensor): return tensor
        import numpy as _np
        # simple clamp of a normal sample (no scipy dependency)
        x = _np.random.normal(mean, std, tensor.shape).astype("float32")
        x = _np.clip(x, mean + a * std, mean + b * std)
        return _assign(tensor, _jt2.array(x).cast(str(tensor.dtype)))
    # override with the tolerant versions (also covers jittor's own names)
    for name, fn in [("normal_", normal_), ("uniform_", uniform_),
                     ("zeros_", zeros_), ("ones_", ones_), ("constant_", constant_),
                     ("trunc_normal_", trunc_normal_)]:
        setattr(_init, name, fn)
    # jittor's native kaiming/xavier/gauss initializers do `var.assign(src)` without
    # re-enabling grad. Under transformers' @torch.no_grad() weight init, `src` is
    # stop_grad, so .assign() silently FREEZES the parameter -- Conv2d/Linear inited
    # with kaiming (resnet/regnet/...) end up stop_grad and get zero weight grads
    # (forward stays exact, so it's invisible until you train/check gradients). Wrap
    # them with the same grad-preserving guard used by _assign() above: a no-op for
    # already-frozen params, so it can't regress anything.
    def _grad_preserving(fn):
        def wrapped(tensor, *a, **k):
            was_trainable = hasattr(tensor, "is_stop_grad") and not tensor.is_stop_grad()
            r = fn(tensor, *a, **k)
            if was_trainable and hasattr(tensor, "start_grad"):
                tensor.start_grad()
            return r
        return wrapped
    for _nm in ("kaiming_normal_", "kaiming_uniform_", "gauss_",
                "xavier_uniform_", "xavier_gauss_", "xavier_normal_",
                "relu_invariant_gauss_", "invariant_uniform_"):
        if hasattr(_init, _nm):
            setattr(_init, _nm, _grad_preserving(getattr(_init, _nm)))
            if hasattr(_jt2.Var, _nm):   # keep the Var-bound method spelling in sync
                setattr(_jt2.Var, _nm, getattr(_init, _nm))
    # keep jittor's good xavier/kaiming; add torch-name aliases for the rest
    aliases = {"xavier_normal_": "xavier_gauss_"}
    for tname, jname in aliases.items():
        if not hasattr(_init, tname) and hasattr(_init, jname):
            setattr(_init, tname, getattr(_init, jname))
    # initializers torch has that jittor lacks -- best-effort implementations
    if not hasattr(_init, "_calculate_fan_in_and_fan_out"):
        def _fan(t):
            sh = t.shape
            if len(sh) < 2:
                return sh[0], sh[0]
            num_input_fmaps, num_output_fmaps = sh[1], sh[0]
            rf = 1
            for s in sh[2:]:
                rf *= s
            return num_input_fmaps * rf, num_output_fmaps * rf
        _init._calculate_fan_in_and_fan_out = _fan
    if not hasattr(_init, "_calculate_correct_fan"):
        def _calculate_correct_fan(tensor, mode):
            mode = str(mode).lower()
            if mode not in ("fan_in", "fan_out"):
                raise ValueError("Mode %s not supported, please use fan_in or fan_out" % mode)
            fan_in, fan_out = _init._calculate_fan_in_and_fan_out(tensor)
            return fan_in if mode == "fan_in" else fan_out
        _init._calculate_correct_fan = _calculate_correct_fan
    if not hasattr(_init, "dirac_"):
        _init.dirac_ = lambda t, *a, **k: t   # best-effort no-op
    if not hasattr(_init, "orthogonal_"):
        def _orth(t, gain=1.0):
            import numpy as _np
            sh = t.shape
            flat = (sh[0], int(t.numel() // sh[0])) if len(sh) > 1 else (sh[0], 1)
            a = _np.random.randn(*flat)
            q, r = _np.linalg.qr(a)
            q = q * _np.sign(_np.diag(r))
            if flat[0] < flat[1]:
                q = q.T
            t.assign(jt.array((gain * q).reshape(sh).astype("float32")))
            return t
        _init.orthogonal_ = _orth
    if not hasattr(_init, "sparse_"):
        _init.sparse_ = lambda t, *a, **k: t  # best-effort no-op

    # torch.nn.init also exposes deprecated non-underscore spellings of the
    # in-place initializers (normal/xavier_normal/kaiming_uniform/kaiming_normal),
    # which forward to the `_` versions. Some older model code calls them. Add
    # each alias only when its `_` target exists and the alias is still missing.
    for tname in ("normal", "xavier_normal", "kaiming_uniform", "kaiming_normal"):
        target = tname + "_"
        if not hasattr(_init, tname) and hasattr(_init, target):
            setattr(_init, tname, getattr(_init, target))

    # Keep transformers/diffusers no_init_weights() from replacing jittor's
    # construction-time init functions with no-op stubs. torch.nn is jittor.nn
    # on the bare `import jittor as torch` path, and jittor.nn.Conv/Linear call
    # the same module-global init functions to allocate weights.
    import types as _types_init
    import sys as _sys_init
    class _GuardedInit(_types_init.ModuleType):
        _protected = set()
        def __setattr__(self, key, value):
            if key in self._protected:
                name = getattr(value, "__name__", "")
                if (not callable(value)) or name in ("_skip_init", "skip_init", "<lambda>"):
                    return
            object.__setattr__(self, key, value)
    guarded = _GuardedInit("torch.nn.init")
    protected = set()
    for key in dir(_init):
        if not key.startswith("__"):
            try:
                value = getattr(_init, key)
                object.__setattr__(guarded, key, value)
                if callable(value):
                    protected.add(key)
            except Exception:
                pass
    object.__setattr__(guarded, "_protected", protected)
    nn.init = guarded
    _sys_init.modules["torch.nn.init"] = guarded


_cuda_props_cache = {}


def _cuda_driver():
    try:
        import ctypes
        for n in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(n)
                lib.cuInit(0)
                return lib, ctypes
            except OSError:
                pass
    except Exception:
        pass
    return None, None


def _cuda_device_index(device=None):
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return 0
    if isinstance(device, int):
        return device
    idx = getattr(device, "index", None)
    return int(idx) if idx is not None else 0


def _cuda_device_name(device=None):
    name = _cuda_props_cache.get("name")
    if name is not None:
        return name
    name = "CUDA"
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), _cuda_device_index(device))
            buf = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(buf, len(buf), dev)
            got = buf.value.decode("utf-8", "ignore")
            if got:
                name = got
    except Exception:
        pass
    _cuda_props_cache["name"] = name
    return name


def _cuda_capability():
    """(major, minor) compute capability of the active CUDA device.

    Queried once from the CUDA driver (compute-capability of device 0); falls
    back to (8, 0) when the driver query is unavailable (e.g. Ascend NPU).
    """
    cc = _cuda_props_cache.get("cap")
    if cc is not None:
        return cc
    cc = (8, 0)
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), 0)
            maj = ctypes.c_int(0); mino = ctypes.c_int(0)
            lib.cuDeviceComputeCapability(ctypes.byref(maj), ctypes.byref(mino), dev)
            if maj.value > 0:
                cc = (maj.value, mino.value)
    except Exception:
        pass
    _cuda_props_cache["cap"] = cc
    return cc


def _cuda_sm_count():
    """SM (multiprocessor) count of CUDA device 0, queried via the driver.

    Triton-based libraries (e.g. flex_gemm's autotuner) size their grids by
    ``get_device_properties(...).multi_processor_count``; a wrong value only
    affects performance/occupancy, not correctness, so we default to 132 (an
    H100-class count) when the driver can't be queried.
    """
    n = _cuda_props_cache.get("sm")
    if n is not None:
        return n
    n = 132
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), 0)
            val = ctypes.c_int(0)
            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
            lib.cuDeviceGetAttribute(ctypes.byref(val), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
            if val.value > 0:
                n = val.value
    except Exception:
        pass
    _cuda_props_cache["sm"] = n
    return n


def _cuda_total_memory():
    total = _cuda_props_cache.get("total_memory")
    if total is not None:
        return total
    total = 64 * 1024 ** 3
    try:
        lib, ctypes = _cuda_driver()
        if lib is not None:
            dev = ctypes.c_int(0)
            lib.cuDeviceGet(ctypes.byref(dev), 0)
            val = ctypes.c_size_t(0)
            fn = getattr(lib, "cuDeviceTotalMem_v2", None) or getattr(lib, "cuDeviceTotalMem", None)
            if fn is not None:
                fn(ctypes.byref(val), dev)
            if val.value > 0:
                total = int(val.value)
    except Exception:
        pass
    _cuda_props_cache["total_memory"] = total
    return total


class _DeviceProps:
    """torch.cuda.get_device_properties(...) result.

    Exposes the attributes real-torch device props carry that libraries read:
    ``name``, ``major``/``minor``, ``total_memory``, ``multi_processor_count``
    (alias ``multiprocessor_count``), ``warp_size``, ``max_threads_per_*``.
    """
    def __init__(self):
        cap = _cuda_capability()
        self.name = "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name()
        self.major, self.minor = cap
        self.total_memory = _cuda_total_memory()
        self.multi_processor_count = _cuda_sm_count()
        self.multiprocessor_count = self.multi_processor_count
        self.warp_size = 32
        self.max_threads_per_multi_processor = 2048
        self.max_threads_per_block = 1024
        self.is_integrated = 0
        self.is_multi_gpu_board = 0
        self.regs_per_multiprocessor = 65536
        self.shared_memory_per_block = 49152
        self.shared_memory_per_multiprocessor = 102400

    def __repr__(self):
        return (f"_DeviceProps(name='{self.name}', major={self.major}, "
                f"minor={self.minor}, total_memory={self.total_memory}, "
                f"multi_processor_count={self.multi_processor_count})")


def _install_cuda(g):
    import types as _types, contextlib
    cuda = _types.ModuleType("torch.cuda")
    def _cuda_visible_devices_empty():
        import os as _os_cuda
        _cvd = _os_cuda.environ.get("CUDA_VISIBLE_DEVICES", None)
        return _cvd is not None and _cvd.strip() == ""

    def is_available():
        try:
            if _cuda_visible_devices_empty():
                return False
            return bool(getattr(jt, "has_cuda", 0)) or bool(getattr(jt.compiler, "has_cuda", 0)) \
                or bool(getattr(jt.compiler, "has_acl", 0))
        except Exception:
            return False
    def device_count():
        if not is_available():
            return 0
        try:
            import os as _os_cuda
            _cvd = _os_cuda.environ.get("CUDA_VISIBLE_DEVICES", None)
            if _cvd is not None:
                return len([_d for _d in _cvd.split(",") if _d.strip()])
        except Exception:
            pass
        return 1
    cuda.is_available = is_available
    cuda.device_count = device_count
    cuda.current_device = lambda: 0
    cuda.set_device = lambda *a, **k: None
    class _CudaDeviceContext:
        def __init__(self, device=None):
            self.device = device
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    cuda.device = _CudaDeviceContext
    cuda.is_initialized = lambda *a, **k: bool(is_available() and getattr(jt.flags, "use_cuda", 0))
    cuda._is_in_bad_fork = lambda *a, **k: False
    # Match PyTorch's empty_cache() as a memory hint instead of a forced
    # synchronization point. TRELLIS calls it inside the inference path before
    # decode; running jt.gc() there costs several seconds. Users that need
    # explicit release can opt in with JITTOR_TORCH_CUDA_EMPTY_CACHE=gc or sync.
    try:
        import os as _os_empty_cache
        _empty_cache_mode = str(_os_empty_cache.environ.get(
            "JITTOR_TORCH_CUDA_EMPTY_CACHE", "0")).strip().lower()
    except Exception:
        _empty_cache_mode = "0"

    def _empty_cache():
        if _empty_cache_mode in ("0", "false", "no", "off", "none", "noop"):
            return
        if _empty_cache_mode in ("", "1", "true", "yes", "on", "gc"):
            try:
                jt.gc()
            except Exception:
                pass
        elif _empty_cache_mode in ("sync", "full"):
            try:
                jt.sync_all(True)
            except Exception:
                pass
            try:
                jt.gc()
            except Exception:
                pass
    cuda.empty_cache = _empty_cache
    cuda.synchronize = lambda *a, **k: jt.sync_all(True)
    cuda.manual_seed = lambda s: jt.set_global_seed(int(s))
    cuda.manual_seed_all = lambda s: jt.set_global_seed(int(s))
    cuda.is_bf16_supported = lambda: True
    cuda.get_device_capability = lambda *a, **k: _cuda_capability()
    def _device_name(*a, **k):
        try:
            return "Ascend910B/NPU" if getattr(jt.compiler, "has_acl", 0) else _cuda_device_name(a[0] if a else None)
        except Exception:
            return "CUDA"
    cuda.get_device_name = _device_name
    cuda.get_device_properties = lambda *a, **k: _DeviceProps()
    class _amp:
        @staticmethod
        def autocast(*a, **k):
            return _AutocastContext()
        GradScaler = _GradScaler
        custom_fwd = staticmethod(_amp_passthrough_decorator)
        custom_bwd = staticmethod(_amp_passthrough_decorator)
    cuda.amp = _amp
    # stub classes referenced in annotations / guarded paths
    cuda.CUDAGraph = type("CUDAGraph", (), {})
    class _Stream:
        def __init__(self, *a, **k): self.cuda_stream = 0
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def synchronize(self): jt.sync_all(True)
        def wait_stream(self, *a, **k): return None
        def wait_event(self, *a, **k): return None
        def record_event(self, *a, **k): return None
        def query(self): return True
    cuda.Stream = _Stream
    cuda.Event = type("Event", (), {"__init__": lambda self, *a, **k: None,
                                      "record": lambda self, *a, **k: None,
                                      "synchronize": lambda self: None,
                                      "elapsed_time": lambda self, o: 0.0})
    g.Stream = cuda.Stream
    g.Event = cuda.Event
    g.CUDAGraph = cuda.CUDAGraph
    cuda.stream = lambda s=None: contextlib.nullcontext()
    cuda.current_stream = lambda *a, **k: _Stream()
    # report REAL device memory from jittor's MemInfo (was a 0-stub, so training-code
    # memory logging printed 0). total_cuda_used on an accelerator, else total_cpu_used.
    # jittor doesn't expose a per-reset peak, so max_* track a process-lifetime high-water
    # mark we maintain here (still real, monotone -- better than a flat 0).
    _mem_peak = [0]
    def _mem_used(*a, **k):
        try:
            mi = jt.get_mem_info()
            used = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
        except Exception:
            used = 0
        if used > _mem_peak[0]:
            _mem_peak[0] = used
        return used
    def _mem_max(*a, **k):
        _mem_used()
        return _mem_peak[0]
    cuda.memory_allocated = _mem_used
    cuda.max_memory_allocated = _mem_max
    cuda.memory_reserved = _mem_used
    cuda.max_memory_reserved = _mem_max
    cuda.memory_cached = _mem_used
    cuda.max_memory_cached = _mem_max
    def _reset_peak(*a, **k):
        try:
            mi = jt.get_mem_info()
            _mem_peak[0] = int(mi.total_cuda_used if jt.flags.use_cuda else mi.total_cpu_used)
        except Exception:
            _mem_peak[0] = 0
    cuda.reset_peak_memory_stats = _reset_peak
    cuda.reset_max_memory_allocated = _reset_peak
    cuda.memory_stats = lambda *a, **k: {"allocated_bytes.all.current": _mem_used(),
                                         "allocated_bytes.all.peak": _mem_peak[0]}
    cuda.mem_get_info = lambda *a, **k: (64*1024**3, 64*1024**3)
    cuda.ipc_collect = lambda *a, **k: None
    cuda.memory = _types.ModuleType("torch.cuda.memory")
    cuda.memory._set_allocator_settings = lambda *a, **k: None
    cuda.memory.empty_cache = cuda.empty_cache
    cuda.memory.memory_allocated = cuda.memory_allocated
    cuda.memory.max_memory_allocated = cuda.max_memory_allocated
    cuda.memory.memory_reserved = cuda.memory_reserved
    cuda.memory.max_memory_reserved = cuda.max_memory_reserved
    # rng state (trainer checkpoints save/restore it). jittor has no portable
    # CUDA rng-state handle, so use a small placeholder Var round-trip.
    cuda.get_rng_state = lambda *a, **k: jt.array([0], dtype="uint8")
    cuda.get_rng_state_all = lambda *a, **k: [jt.array([0], dtype="uint8")]
    cuda.set_rng_state = lambda *a, **k: None
    cuda.set_rng_state_all = lambda *a, **k: None
    cuda.initial_seed = lambda *a, **k: 0
    cuda.seed = lambda *a, **k: None
    cuda.seed_all = lambda *a, **k: None
    import types as _types_cuda
    _curandom = _types_cuda.ModuleType("torch.cuda.random")
    _curandom.get_rng_state = cuda.get_rng_state
    _curandom.get_rng_state_all = cuda.get_rng_state_all
    _curandom.set_rng_state = cuda.set_rng_state
    _curandom.set_rng_state_all = cuda.set_rng_state_all
    _curandom.manual_seed = cuda.manual_seed
    _curandom.manual_seed_all = cuda.manual_seed_all
    _curandom.initial_seed = cuda.initial_seed
    cuda.random = _curandom
    import sys as _sys_cuda
    _sys_cuda.modules["torch.cuda.random"] = _curandom
    g.cuda = cuda
    _sys_cuda.modules["torch.cuda"] = cuda
    _sys_cuda.modules["torch.cuda.memory"] = cuda.memory
    if hasattr(cuda, "amp"):
        _sys_cuda.modules["torch.cuda.amp"] = cuda.amp

    for _dev_ns in ("mps", "cpu", "npu", "xpu", "mtia"):
        _mod = _sys_cuda.modules.get("torch." + _dev_ns)
        if _mod is None:
            _mod = _types.ModuleType("torch." + _dev_ns)
            _sys_cuda.modules["torch." + _dev_ns] = _mod
        _mod.is_available = getattr(_mod, "is_available", lambda *a, **k: False)
        _mod.is_initialized = getattr(_mod, "is_initialized", lambda *a, **k: False)
        _mod.device_count = getattr(_mod, "device_count", lambda *a, **k: 0)
        _mod.current_device = getattr(_mod, "current_device", lambda *a, **k: 0)
        _mod.set_device = getattr(_mod, "set_device", lambda *a, **k: None)
        _mod.empty_cache = getattr(_mod, "empty_cache", lambda *a, **k: None)
        _mod.synchronize = getattr(_mod, "synchronize", lambda *a, **k: None)
        _mod.ipc_collect = getattr(_mod, "ipc_collect", lambda *a, **k: None)
        _mod.manual_seed = getattr(_mod, "manual_seed", lambda *a, **k: None)
        _mod.manual_seed_all = getattr(_mod, "manual_seed_all", lambda *a, **k: None)
        _mod.seed = getattr(_mod, "seed", lambda *a, **k: None)
        _mod.reset_peak_memory_stats = getattr(_mod, "reset_peak_memory_stats", lambda *a, **k: None)
        _mod.reset_max_memory_allocated = getattr(_mod, "reset_max_memory_allocated", lambda *a, **k: None)
        _mod.memory_allocated = getattr(_mod, "memory_allocated", lambda *a, **k: 0)
        _mod.max_memory_allocated = getattr(_mod, "max_memory_allocated", lambda *a, **k: 0)
        setattr(g, _dev_ns, _mod)

    if "torch.multiprocessing" not in _sys_cuda.modules:
        import multiprocessing as _mp
        _sys_cuda.modules["torch.multiprocessing"] = _mp
    g.multiprocessing = _sys_cuda.modules["torch.multiprocessing"]
    _mp_reductions = _types.ModuleType("torch.multiprocessing.reductions")
    _mp_reductions.reduce_tensor = lambda tensor: (lambda x: x, (tensor,))
    _mp_reductions.rebuild_cuda_tensor = lambda *a, **k: None
    _mp_reductions.rebuild_tensor = lambda *a, **k: a[0] if a else None
    _sys_cuda.modules["torch.multiprocessing.reductions"] = _mp_reductions
    try:
        g.multiprocessing.reductions = _mp_reductions
    except Exception:
        pass

    if "torch.overrides" not in _sys_cuda.modules:
        overrides = _types.ModuleType("torch.overrides")
        class TorchFunctionMode:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))
        overrides.TorchFunctionMode = TorchFunctionMode
        overrides.BaseTorchFunctionMode = TorchFunctionMode
        overrides.get_default_nowrap_functions = lambda: set()
        overrides.has_torch_function = lambda *a, **k: False
        overrides.handle_torch_function = lambda func, types, *a, **k: func(*a, **k)
        _sys_cuda.modules["torch.overrides"] = overrides
    g.overrides = _sys_cuda.modules["torch.overrides"]

    if "torch._C" not in _sys_cuda.modules:
        c_mod = _types.ModuleType("torch._C")
        c_mod._TensorMeta = type(getattr(g, "Tensor", jt.Var))
        c_mod._get_tracing_state = lambda: None
        c_mod._log_api_usage_once = lambda *a, **k: None
        c_mod._cuda_clearCublasWorkspaces = lambda *a, **k: None
        c_mod._disabled_torch_function_impl = lambda *a, **k: NotImplemented
        functorch_c = _types.ModuleType("torch._C._functorch")
        functorch_c.get_unwrapped = lambda x: x
        functorch_c.is_batchedtensor = lambda *a, **k: False
        functorch_c._add_batch_dim = lambda x, *a, **k: x
        functorch_c._remove_batch_dim = lambda x, *a, **k: x
        c_mod._distributed_c10d = _types.SimpleNamespace(Reducer=type("Reducer", (), {}))
        nn_c = _types.ModuleType("torch._C._nn")
        def _parse_to(*args, **kwargs):
            dev = kwargs.get("device", None)
            dtype_arg = kwargs.get("dtype", None)
            non_blocking = kwargs.get("non_blocking", False)
            for arg in args:
                if isinstance(arg, jt.Var):
                    dev = getattr(arg, "device", dev)
                    dtype_arg = getattr(arg, "dtype", dtype_arg)
                    continue
                if isinstance(arg, dtype) or str(arg).replace("torch.", "") in dtype._registry:
                    if dtype_arg is None:
                        dtype_arg = arg
                    continue
                if isinstance(arg, str) or hasattr(arg, "type"):
                    if dev is None:
                        dev = arg
                    continue
                if arg in getattr(dtype, "_registry", {}).values():
                    if dtype_arg is None:
                        dtype_arg = arg
            return dev, dtype_arg, non_blocking, kwargs.get("memory_format", None)
        nn_c._parse_to = _parse_to
        c_mod._nn = nn_c
        c_mod._functorch = functorch_c
        _sys_cuda.modules["torch._C"] = c_mod
        _sys_cuda.modules["torch._C._nn"] = nn_c
        _sys_cuda.modules["torch._C._functorch"] = functorch_c
    g._C = _sys_cuda.modules["torch._C"]
    if not hasattr(g._C, "_autograd"):
        g._C._autograd = _types.SimpleNamespace()
    g._C._autograd._push_saved_tensors_default_hooks = lambda *a, **k: None
    g._C._autograd._pop_saved_tensors_default_hooks = lambda *a, **k: None
    _sys_cuda.modules["torch._C._autograd"] = g._C._autograd

    backends = _sys_cuda.modules.get("torch.backends")
    if backends is None:
        backends = _types.ModuleType("torch.backends")
        _sys_cuda.modules["torch.backends"] = backends
    cudnn = _sys_cuda.modules.get("torch.backends.cudnn")
    if cudnn is None:
        cudnn = _types.ModuleType("torch.backends.cudnn")
        _sys_cuda.modules["torch.backends.cudnn"] = cudnn
    if type(cudnn).__name__ != "_CudnnBackendModule":
        class _CudnnBackendModule(_types.ModuleType):
            def __setattr__(self, name, value):
                if name == "benchmark" and not getattr(self, "_jittor_cudnn_init", False):
                    try:
                        if getattr(jt, "cudnn", None) is not None and hasattr(jt.cudnn, "set_benchmark"):
                            jt.cudnn.set_benchmark(int(bool(value)))
                    except Exception:
                        pass
                return super().__setattr__(name, value)
        cudnn.__class__ = _CudnnBackendModule
    cudnn._jittor_cudnn_init = True
    cudnn.enabled = getattr(cudnn, "enabled", True)
    cudnn.benchmark = getattr(cudnn, "benchmark", False)
    cudnn.deterministic = getattr(cudnn, "deterministic", False)
    cudnn.allow_tf32 = getattr(cudnn, "allow_tf32", True)
    cudnn.version = getattr(cudnn, "version", lambda: None)
    cudnn._jittor_cudnn_init = False
    cuda_backend = _sys_cuda.modules.get("torch.backends.cuda")
    if cuda_backend is None:
        cuda_backend = _types.ModuleType("torch.backends.cuda")
        _sys_cuda.modules["torch.backends.cuda"] = cuda_backend
    class _SDPKernel:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    cuda_backend.sdp_kernel = getattr(cuda_backend, "sdp_kernel", lambda *a, **k: _SDPKernel())
    cuda_backend.enable_flash_sdp = getattr(cuda_backend, "enable_flash_sdp", lambda *a, **k: None)
    cuda_backend.enable_mem_efficient_sdp = getattr(cuda_backend, "enable_mem_efficient_sdp", lambda *a, **k: None)
    cuda_backend.enable_math_sdp = getattr(cuda_backend, "enable_math_sdp", lambda *a, **k: None)
    class _MatmulBackend:
        @property
        def allow_tf32(self):
            cuda_tf32 = bool(getattr(jt.flags, "cuda_allow_tf32", 0))
            acl_hf32 = bool(getattr(jt, "acl_allow_hf32", False))
            return cuda_tf32 or acl_hf32

        @allow_tf32.setter
        def allow_tf32(self, value):
            enabled = bool(value)
            if hasattr(jt.flags, "cuda_allow_tf32"):
                jt.flags.cuda_allow_tf32 = int(enabled)
            jt.acl_allow_hf32 = enabled
    if not hasattr(cuda_backend, "matmul") or not isinstance(cuda_backend.matmul, _MatmulBackend):
        cuda_backend.matmul = _MatmulBackend()
    mps = _sys_cuda.modules.get("torch.backends.mps")
    if mps is None:
        mps = _types.ModuleType("torch.backends.mps")
        _sys_cuda.modules["torch.backends.mps"] = mps
    mps.is_available = getattr(mps, "is_available", lambda: False)
    cpu = _sys_cuda.modules.get("torch.backends.cpu")
    if cpu is None:
        cpu = _types.ModuleType("torch.backends.cpu")
        _sys_cuda.modules["torch.backends.cpu"] = cpu
    cpu.get_cpu_capability = getattr(cpu, "get_cpu_capability", lambda: "DEFAULT")
    mkldnn = _sys_cuda.modules.get("torch.backends.mkldnn")
    if mkldnn is None:
        mkldnn = _types.ModuleType("torch.backends.mkldnn")
        _sys_cuda.modules["torch.backends.mkldnn"] = mkldnn
    mkldnn.is_available = getattr(mkldnn, "is_available", lambda: False)
    mkldnn.enabled = getattr(mkldnn, "enabled", False)
    backends.cudnn = cudnn
    backends.cuda = cuda_backend
    backends.mps = mps
    backends.cpu = cpu
    backends.mkldnn = mkldnn
    g.backends = backends
    if not hasattr(g, "_torch_float32_matmul_precision"):
        g._torch_float32_matmul_precision = "highest"
    def _get_float32_matmul_precision():
        return getattr(g, "_torch_float32_matmul_precision", "highest")
    def _set_float32_matmul_precision(precision):
        if not isinstance(precision, str):
            raise TypeError("precision must be a string")
        precision = precision.lower()
        if precision not in ("highest", "high", "medium"):
            raise ValueError("precision must be one of 'highest', 'high', or 'medium'")
        g._torch_float32_matmul_precision = precision
        try:
            cuda_backend.matmul.allow_tf32 = precision in ("high", "medium")
        except Exception:
            pass
    g.get_float32_matmul_precision = _get_float32_matmul_precision
    g.set_float32_matmul_precision = _set_float32_matmul_precision


def _install_version(g):
    """Install torch.version for libraries that probe torch.cuda/hip versions."""
    import sys as _sys
    import types as _types
    torch_api_version = "2.11.0"
    jittor_version = getattr(g, "__jittor_version__", getattr(g, "__version__", getattr(jt, "__version__", None)))
    g.__jittor_version__ = jittor_version
    g.__torch_version__ = torch_api_version
    g.__version__ = torch_api_version
    version = _types.ModuleType("torch.version")
    version.__version__ = torch_api_version
    version.jittor = jittor_version
    try:
        nv = getattr(getattr(jt, "compiler", None), "nvcc_version", None)
        version.cuda = ".".join(map(str, nv[:2])) if nv else None
    except Exception:
        version.cuda = None
    version.hip = None
    version.git_version = "jittor"
    _sys.modules["torch.version"] = version
    g.version = version


def _install_torchdata_stateful_dataloader(g):
    """Provide torchdata.stateful_dataloader for verl trainer imports.

    Newer torchdata packages may omit the stateful_dataloader namespace while
    verl still imports it.  A single-process fallback can use the installed
    torch.utils.data.DataLoader and expose no-op state_dict hooks.
    """
    import sys as _sys
    import types as _types

    torchdata = _sys.modules.get("torchdata")
    if torchdata is None:
        torchdata = _types.ModuleType("torchdata")
        torchdata.__version__ = "0.0.jittor"
        _sys.modules["torchdata"] = torchdata

    stateful = _types.ModuleType("torchdata.stateful_dataloader")
    sampler_mod = _types.ModuleType("torchdata.stateful_dataloader.sampler")
    data_mod = getattr(getattr(g, "utils", None), "data", None)
    base_loader = getattr(data_mod, "DataLoader", object)

    class StatefulDataLoader(base_loader):
        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return None

    stateful.StatefulDataLoader = StatefulDataLoader
    if data_mod is not None:
        for name in ("RandomSampler", "SequentialSampler", "BatchSampler", "Sampler"):
            if hasattr(data_mod, name):
                setattr(sampler_mod, name, getattr(data_mod, name))
    _sys.modules["torchdata.stateful_dataloader"] = stateful
    _sys.modules["torchdata.stateful_dataloader.sampler"] = sampler_mod
    setattr(torchdata, "stateful_dataloader", stateful)


def _install_torchmetrics_fastpaths(g):
    """Patch TorchMetrics internals with jittor-safe fast paths.

    Public ``torch.bincount`` must keep PyTorch's output-length semantics, which
    require ``max(input.max()+1, minlength)`` and therefore a GPU->host sync in
    the generic compatibility implementation. TorchMetrics classification
    helpers pass a known bounded ``minlength`` (for example ``num_classes**2``)
    and then immediately reshape to that fixed size. Patch only that internal
    helper so TorchMetrics avoids the sync without changing user-visible torch
    semantics.
    """
    import builtins as _builtins
    import sys as _sys

    if getattr(g, "_torchmetrics_fastpaths_installed", False):
        return
    g._torchmetrics_fastpaths_installed = True

    def _patch_bound_torchmetrics_attr(attr, orig, fast):
        for name, mod in list(_sys.modules.items()):
            if not name.startswith("torchmetrics."):
                continue
            if getattr(mod, attr, None) is orig:
                setattr(mod, attr, fast)

    def _patch_data_mod(mod):
        if mod is None:
            return mod

        if getattr(mod, "_jittor_fast_bincount", False):
            fast = getattr(mod, "_bincount", None)
            orig = getattr(fast, "_jittor_orig_bincount", None)
            if orig is not None:
                _patch_bound_torchmetrics_attr("_bincount", orig, fast)
        else:
            orig = getattr(mod, "_bincount", None)
            if orig is not None:
                def _bounded_bincount(x, minlength=None, _orig=orig):
                    if minlength is None or not isinstance(minlength, (int, np.integer)):
                        return _orig(x, minlength=minlength)
                    ml = max(int(minlength), 0)
                    flat = x.reshape(-1).int64()
                    if flat.numel() == 0:
                        return jt.zeros((ml,), dtype=jt.int64)
                    out = jt.zeros((ml,), dtype=jt.int64)
                    src = jt.ones((flat.shape[0],), dtype=jt.int64)
                    return out.scatter_add(0, flat, src)

                _bounded_bincount._jittor_orig_bincount = orig
                mod._bincount = _bounded_bincount
                mod._jittor_fast_bincount = True
                _patch_bound_torchmetrics_attr("_bincount", orig, _bounded_bincount)

        if getattr(mod, "_jittor_fast_dim_zero_cat", False):
            fast = getattr(mod, "dim_zero_cat", None)
            orig = getattr(fast, "_jittor_orig_dim_zero_cat", None)
            if orig is not None:
                _patch_bound_torchmetrics_attr("dim_zero_cat", orig, fast)
        else:
            orig = getattr(mod, "dim_zero_cat", None)
            if orig is not None:
                def _fast_dim_zero_cat(x, _orig=orig):
                    if isinstance(x, jt.Var):
                        return x
                    try:
                        n = len(x)
                    except TypeError:
                        return _orig(x)
                    if n == 0:
                        raise ValueError("No samples to concatenate")
                    if n == 1:
                        y = x[0]
                        if not isinstance(y, jt.Var):
                            return _orig(x)
                        if y.numel() == 1 and getattr(y, "ndim", 0) == 0:
                            return y.unsqueeze(0)
                        return y.clone()
                    return _orig(x)

                _fast_dim_zero_cat._jittor_orig_dim_zero_cat = orig
                mod.dim_zero_cat = _fast_dim_zero_cat
                mod._jittor_fast_dim_zero_cat = True
                _patch_bound_torchmetrics_attr("dim_zero_cat", orig, _fast_dim_zero_cat)

        return mod

    def _patch_compute_mod(mod):
        if mod is None:
            return mod
        if getattr(mod, "_jittor_fast_safe_divide", False):
            fast = getattr(mod, "_safe_divide", None)
            orig = getattr(fast, "_jittor_orig_safe_divide", None)
            if orig is not None:
                _patch_bound_torchmetrics_attr("_safe_divide", orig, fast)
            return mod
        orig = getattr(mod, "_safe_divide", None)
        if orig is None:
            return mod

        def _fast_safe_divide(num, denom, zero_division=0.0):
            if not isinstance(zero_division, (float, int)):
                return orig(num, denom, zero_division=zero_division)
            if not hasattr(num, "is_floating_point") or not hasattr(denom, "is_floating_point"):
                return orig(num, denom, zero_division=zero_division)
            num = num if num.is_floating_point() else num.float()
            denom = denom if denom.is_floating_point() else denom.float()
            div = num / denom
            fill = jt.zeros_like(div) if zero_division == 0 else jt.zeros_like(div) + zero_division
            return g.where(denom != 0, div, fill)

        _fast_safe_divide._jittor_orig_safe_divide = orig
        mod._safe_divide = _fast_safe_divide
        mod._jittor_fast_safe_divide = True
        _patch_bound_torchmetrics_attr("_safe_divide", orig, _fast_safe_divide)
        return mod

    _patch_data_mod(_sys.modules.get("torchmetrics.utilities.data"))
    _patch_compute_mod(_sys.modules.get("torchmetrics.utilities.compute"))

    orig_import = _builtins.__import__
    if getattr(orig_import, "_jittor_torchmetrics_fastpaths", False):
        return

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = orig_import(name, globals, locals, fromlist, level)
        if name == "torchmetrics.utilities.data" or name.startswith("torchmetrics."):
            _patch_data_mod(_sys.modules.get("torchmetrics.utilities.data"))
            _patch_compute_mod(_sys.modules.get("torchmetrics.utilities.compute"))
        return mod

    _import._jittor_torchmetrics_fastpaths = True
    _builtins.__import__ = _import


def _install_fsdp2_distributed(dist, torch_module=None):
    """Install the single-process FSDP2/DTensor compatibility surface."""
    from . import torch_fsdp2_compat as _fsdp2
    return _fsdp2.install(dist, torch_module)

def _install_distributed(g):
    """Install single-process torch.distributed stubs.

    Transformers 5 imports torch.distributed at module import time for tensor
    parallel helpers even when no distributed execution is requested. The jittor
    torch shim runs TRELLIS.2 as a single process, so report distributed support
    as unavailable while keeping the imported symbols present.
    """
    import sys as _sys
    import types as _types

    dist = _sys.modules.get("torch.distributed")
    if dist is None:
        dist = _types.ModuleType("torch.distributed")
        _sys.modules["torch.distributed"] = dist
    dist.is_available = lambda *a, **k: True
    dist.is_initialized = lambda *a, **k: False
    dist.get_rank = lambda *a, **k: 0
    dist.get_world_size = lambda *a, **k: 1
    dist.init_process_group = lambda *a, **k: None
    dist.destroy_process_group = lambda *a, **k: None
    dist.barrier = lambda *a, **k: None
    dist.all_reduce = lambda *a, **k: None
    dist.all_gather = lambda *a, **k: None
    dist.broadcast = lambda *a, **k: None
    def _all_gather_object(object_list, obj, *a, **k):
        if object_list:
            object_list[0] = obj
        return None
    def _broadcast_object_list(object_list, src=0, *a, **k):
        return object_list
    def _all_gather_into_tensor(output_tensor, input_tensor, *a, **k):
        try:
            output_tensor.assign(input_tensor.reshape(output_tensor.shape))
        except Exception:
            try:
                output_tensor.assign(input_tensor)
            except Exception:
                pass
        return None
    dist.all_gather_object = _all_gather_object
    dist.broadcast_object_list = _broadcast_object_list
    dist.all_gather_into_tensor = _all_gather_into_tensor
    dist.gather_object = lambda obj, object_gather_list=None, dst=0, *a, **k: _all_gather_object(object_gather_list or [], obj)
    dist.new_group = lambda *a, **k: dist.group.WORLD
    dist.new_subgroups_by_enumeration = lambda *a, **k: ([dist.group.WORLD], dist.group.WORLD)
    dist.get_global_rank = lambda group=None, group_rank=0: int(group_rank)
    dist.is_torchelastic_launched = lambda *a, **k: False
    class _ReduceOp:
        SUM = 0
        MEAN = 1
        AVG = 1
        MAX = 2
        MIN = 3
        PRODUCT = 4
    _ReduceOp.RedOpType = _ReduceOp
    dist.ReduceOp = getattr(dist, "ReduceOp", _ReduceOp)
    if not hasattr(dist.ReduceOp, "RedOpType"):
        dist.ReduceOp.RedOpType = dist.ReduceOp
    dist.GroupMember = getattr(dist, "GroupMember", type("GroupMember", (), {"WORLD": None}))
    dist.group = getattr(dist, "group", type("group", (), {"WORLD": None}))

    for sub in ("tensor", "fsdp", "device_mesh", "algorithms", "_composable",
                "checkpoint", "_shard", "nn"):
        name = "torch.distributed." + sub
        mod = _sys.modules.get(name)
        if mod is None:
            mod = _types.ModuleType(name)
            _sys.modules[name] = mod
        setattr(dist, sub, mod)

    dist.algorithms.__path__ = getattr(dist.algorithms, "__path__", [])
    const_mod = _sys.modules.get("torch.distributed.constants")
    if const_mod is None:
        const_mod = _types.ModuleType("torch.distributed.constants")
        _sys.modules["torch.distributed.constants"] = const_mod
    try:
        import datetime as _datetime_dist
        const_mod.default_pg_timeout = getattr(
            const_mod, "default_pg_timeout", _datetime_dist.timedelta(minutes=30)
        )
    except Exception:
        const_mod.default_pg_timeout = getattr(const_mod, "default_pg_timeout", None)
    dist.constants = const_mod
    join_mod = _sys.modules.get("torch.distributed.algorithms.join")
    if join_mod is None:
        join_mod = _types.ModuleType("torch.distributed.algorithms.join")
        _sys.modules["torch.distributed.algorithms.join"] = join_mod
    class JoinHook:
        def main_hook(self):
            return None
        def post_hook(self, is_last_joiner):
            return None
    class Joinable:
        def __init__(self, *a, **k):
            pass
        @property
        def join_hook(self):
            return JoinHook()
        @property
        def join_device(self):
            return None
        @property
        def join_process_group(self):
            return dist.group.WORLD
    class Join:
        def __init__(self, joinables, enable=True, throw_on_early_termination=False, **kwargs):
            self.joinables = list(joinables) if joinables is not None else []
            self.enable = enable
            self.throw_on_early_termination = throw_on_early_termination
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        @staticmethod
        def notify_join_context(joinable):
            return None
        @staticmethod
        def notify_join_context_enabled(joinable):
            return False
    join_mod.Join = Join
    join_mod.Joinable = Joinable
    join_mod.JoinHook = JoinHook
    dist.algorithms.join = join_mod

    class _DeviceMesh:
        def __init__(self, device_type=None, mesh=None, *, mesh_dim_names=None, **k):
            self.device_type = device_type
            self.mesh = mesh
            self.mesh_dim_names = mesh_dim_names
        def __getitem__(self, *a, **k): return self
        def size(self, *a, **k): return 1
        def get_rank(self, *a, **k): return 0
        def get_group(self, *a, **k): return None
        def get_local_rank(self, *a, **k): return 0
    def _init_device_mesh(device_type=None, mesh_shape=None, *, mesh_dim_names=None, **k):
        return _DeviceMesh(device_type=device_type, mesh=mesh_shape,
                           mesh_dim_names=mesh_dim_names)
    dist.device_mesh.DeviceMesh = getattr(dist.device_mesh, "DeviceMesh", _DeviceMesh)
    dist.device_mesh.init_device_mesh = getattr(dist.device_mesh, "init_device_mesh", _init_device_mesh)
    dist.DeviceMesh = getattr(dist, "DeviceMesh", _DeviceMesh)
    dist.init_device_mesh = getattr(dist, "init_device_mesh", _init_device_mesh)
    dist.ProcessGroup = getattr(dist, "ProcessGroup", type("ProcessGroup", (), {
        "__init__": lambda self, *a, **k: None,
        "size": lambda self, *a, **k: 1,
        "rank": lambda self, *a, **k: 0,
    }))

    c10d = _sys.modules.get("torch.distributed.distributed_c10d")
    if c10d is None:
        c10d = _types.ModuleType("torch.distributed.distributed_c10d")
        _sys.modules["torch.distributed.distributed_c10d"] = c10d
    for name in dir(dist):
        if not name.startswith("__"):
            setattr(c10d, name, getattr(dist, name))
    for name in ("is_xccl_available", "is_nccl_available", "is_gloo_available",
                 "is_mpi_available", "is_ucc_available"):
        setattr(c10d, name, lambda *a, **k: False)
    c10d.ProcessGroup = dist.ProcessGroup
    c10d._get_default_group = lambda *a, **k: None
    c10d._get_default_store = lambda *a, **k: None
    c10d.Work = getattr(c10d, "Work", type("Work", (), {}))
    c10d.default_pg_timeout = getattr(c10d, "default_pg_timeout", None)
    dist.distributed_c10d = c10d

    rpc = _sys.modules.get("torch.distributed.rpc")
    if rpc is None:
        rpc = _types.ModuleType("torch.distributed.rpc")
        _sys.modules["torch.distributed.rpc"] = rpc
    rpc.is_available = lambda *a, **k: False
    rpc.init_rpc = lambda *a, **k: None
    rpc.shutdown = lambda *a, **k: None
    dist.rpc = rpc

    optim = _sys.modules.get("torch.distributed.optim")
    if optim is None:
        optim = _types.ModuleType("torch.distributed.optim")
        _sys.modules["torch.distributed.optim"] = optim
    dist.optim = optim

    dist.nn.all_reduce = lambda input, *a, **k: input
    _sys.modules["torch.distributed.nn"] = dist.nn

    futures = _sys.modules.get("torch.futures")
    if futures is None:
        futures = _types.ModuleType("torch.futures")
        _sys.modules["torch.futures"] = futures
    class Future:
        def __init__(self, devices=None):
            self._value = None
        def set_result(self, value):
            self._value = value
            return self
        def value(self):
            return self._value
        def wait(self):
            return self._value
        def then(self, callback):
            return callback(self)
    futures.Future = Future
    g.futures = futures

    checkpoint = dist.checkpoint
    checkpoint.__path__ = getattr(checkpoint, "__path__", [])
    class FileSystemReader:
        def __init__(self, path, *a, **k):
            self.path = path
    class FileSystemWriter:
        def __init__(self, path, *a, **k):
            self.path = path
    checkpoint.FileSystemReader = FileSystemReader
    checkpoint.FileSystemWriter = FileSystemWriter
    checkpoint.load_state_dict = lambda state_dict, *a, **k: state_dict
    checkpoint.save_state_dict = lambda state_dict, *a, **k: state_dict
    checkpoint.load = lambda state_dict=None, *a, **k: state_dict
    checkpoint.save = lambda state_dict=None, *a, **k: state_dict
    checkpoint_sd = _types.ModuleType("torch.distributed.checkpoint.state_dict")
    class StateDictOptions:
        def __init__(self, *, full_state_dict=False, cpu_offload=False,
                     ignore_frozen_params=False, keep_submodule_prefixes=True,
                     strict=True, broadcast_from_rank0=False, flatten_optimizer_state_dict=False):
            self.full_state_dict = bool(full_state_dict)
            self.cpu_offload = bool(cpu_offload)
            self.ignore_frozen_params = bool(ignore_frozen_params)
            self.keep_submodule_prefixes = bool(keep_submodule_prefixes)
            self.strict = bool(strict)
            self.broadcast_from_rank0 = bool(broadcast_from_rank0)
            self.flatten_optimizer_state_dict = bool(flatten_optimizer_state_dict)
    def _get_model_state_dict(model, *a, options=None, **k):
        return model.state_dict(*a, **k) if hasattr(model, "state_dict") else {}
    def _set_model_state_dict(model, state_dict, *a, options=None, **k):
        if hasattr(model, "load_state_dict"):
            return model.load_state_dict(state_dict, strict=getattr(options, "strict", True))
        return None
    checkpoint_sd.StateDictOptions = StateDictOptions
    checkpoint_sd.get_model_state_dict = _get_model_state_dict
    checkpoint_sd.set_model_state_dict = _set_model_state_dict
    checkpoint_sd.get_state_dict = lambda model, optimizers=None, *a, **k: (
        _get_model_state_dict(model, *a, **k),
        optimizers.state_dict() if hasattr(optimizers, "state_dict") else {},
    )
    checkpoint_sd.set_state_dict = lambda model, optimizers=None, model_state_dict=None, optim_state_dict=None, *a, **k: (
        _set_model_state_dict(model, model_state_dict or {}, *a, **k)
    )
    checkpoint_fs = _types.ModuleType("torch.distributed.checkpoint.filesystem")
    checkpoint_fs.FileSystemReader = FileSystemReader
    checkpoint_fs.FileSystemWriter = FileSystemWriter
    checkpoint_fs.SerializationFormat = type("SerializationFormat", (), {
        "TORCH_SAVE": "torch_save",
        "SAFETENSORS": "safetensors",
    })
    checkpoint_fs._write_item = lambda *a, **k: None
    _sys.modules["torch.distributed.checkpoint"] = checkpoint
    _sys.modules["torch.distributed.checkpoint.state_dict"] = checkpoint_sd
    _sys.modules["torch.distributed.checkpoint.filesystem"] = checkpoint_fs
    checkpoint.state_dict = checkpoint_sd
    checkpoint.filesystem = checkpoint_fs

    shard = dist._shard
    shard.__path__ = getattr(shard, "__path__", [])
    sharded_tensor = _types.ModuleType("torch.distributed._shard.sharded_tensor")
    class ShardedTensor:
        pass
    sharded_tensor.ShardedTensor = ShardedTensor
    sharded_tensor.init_from_local_shards = lambda shards, *a, **k: shards[0] if shards else None
    sharded_tensor.empty = lambda *a, **k: jt.empty(*a, **{kk: vv for kk, vv in k.items() if kk == "dtype"})
    shard.sharded_tensor = sharded_tensor
    _sys.modules["torch.distributed._shard"] = shard
    _sys.modules["torch.distributed._shard.sharded_tensor"] = sharded_tensor
    for _sub in ("api", "metadata", "reshard", "shard"):
        _m = _types.ModuleType("torch.distributed._shard.sharded_tensor." + _sub)
        _m.ShardedTensor = ShardedTensor
        _sys.modules[_m.__name__] = _m

    class TCPStore:
        _data = {}
        def __init__(self, *a, **k):
            pass
        def set(self, key, value):
            self._data[str(key)] = value
        def get(self, key):
            return self._data.get(str(key), b"")
        def add(self, key, num):
            v = int(self._data.get(str(key), 0)) + int(num)
            self._data[str(key)] = v
            return v
        def wait(self, keys, *a, **k):
            return None
        def delete_key(self, key):
            self._data.pop(str(key), None)
            return True
    dist.TCPStore = TCPStore

    if not hasattr(g, "_C"):
        class _Accel:
            type = "cuda" if getattr(jt.compiler, "has_cuda", 0) else "cpu"
        class _CNS:
            @staticmethod
            def _get_accelerator():
                return _Accel()
        g._C = _CNS()
    _install_fsdp2_distributed(dist, getattr(g, "__dict__", None))
    g.distributed = dist


def _install_flash_attn_shim():
    """Register the Jittor-backed flash_attn stub for the bare jittor path."""
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    torch_mod = _sys.modules.get("torch")
    if torch_mod is not None and torch_mod is not jt:
        # When the deployed `import torch` shim imports jittor, the `torch`
        # package body is still half-built. The optional flash_attn stub imports
        # torch.nn.functional, so importing it at that point can abort the core
        # torch_compat install. deploy.py installs a normal flash_attn package
        # for that path; only register this direct stub for `import jittor as
        # torch` flows.
        return
    mod = _sys.modules.get("flash_attn")
    if mod is not None and getattr(mod, "_jittor_flash_attn_stub", False):
        return
    src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "torch_shim", "stubs", "flash_attn", "__init__.py")
    if not _os.path.isfile(src):
        return
    spec = _ilu.spec_from_file_location("flash_attn", src)
    shim = _ilu.module_from_spec(spec)
    old_flash = _sys.modules.get("flash_attn")
    _sys.modules["flash_attn"] = shim
    _sys.modules["torch"] = jt
    try:
        spec.loader.exec_module(shim)
    except Exception:
        if old_flash is None:
            _sys.modules.pop("flash_attn", None)
        else:
            _sys.modules["flash_attn"] = old_flash
        raise
    shim._jittor_flash_attn_stub = True


def _install_tensor_methods(g, Var, _DTYPE_OBJS=None):
    # Var.dtype natively returns jittor's NanoString, which is unhashable and
    # not == to torch dtype objects. Wrap it to return our hashable `dtype`
    # (str subclass), so `t.dtype in {torch.float16, ...}` and dict keys work.
    if _DTYPE_OBJS is not None and not getattr(Var, "_dtype_wrapped", False):
        try:
            _native_desc = Var.__dict__.get("dtype")  # C getset_descriptor
            if _native_desc is not None:
                def _dtype_get(self, _d=_native_desc):
                    name = str(_d.__get__(self, type(self)))
                    return _DTYPE_OBJS.get(name, name)
                Var.dtype = property(_dtype_get)
                Var._dtype_wrapped = True
        except Exception:
            pass

    # torch parity for `x[bool_mask] = value` when the mask has lower rank than
    # `x` and `value` carries redundant leading size-1 batch axes. Torch assigns
    # a RHS shaped like (1, N, C) into the selected region shaped (N, C); jittor's
    # native setitem rejects the extra leading axis. This path is used by
    # TRELLIS/o_voxel texture baking (`attrs[mask] = grid_sample_3d(...)`).
    _orig_setitem = Var.__setitem__
    if not getattr(_orig_setitem, "_torch_mask_bcast", False):
        def _torch_setitem(self, slices, value):
            try:
                mask = slices
                if isinstance(mask, Var) and mask.dtype in ("bool", "uint8") \
                        and isinstance(value, Var) \
                        and len(mask.shape) < len(self.shape):
                    # Region selected by a lower-rank bool mask has shape
                    # (N, *self.shape[mask.ndim:]). Drop only provably redundant
                    # leading singleton axes from value until ranks agree.
                    region_rank = 1 + (len(self.shape) - len(mask.shape))
                    while len(value.shape) > region_rank and value.shape[0] == 1:
                        value = value.squeeze(0)
            except Exception:
                pass
            result = _orig_setitem(self, slices, value)
            _write_index_parent(self, self)
            return result
        _torch_setitem._torch_mask_bcast = True
        Var.__setitem__ = _torch_setitem

    def _write_index_parent(view, value):
        parent = getattr(view, "_torch_index_parent", None)
        parent_slices = getattr(view, "_torch_index_slices", None)
        if not isinstance(parent, Var):
            return
        parent_was_trainable = not parent.is_stop_grad()
        # Bypass the compatibility wrapper here; this helper owns the one
        # explicit ancestor walk. Calling patched __setitem__ would recurse once
        # implicitly and once below, duplicating graph nodes at every depth.
        _orig_setitem(parent, parent_slices, value)
        if parent_was_trainable and parent.is_stop_grad():
            parent.start_grad()
        elif not parent_was_trainable and not parent.is_stop_grad():
            parent.stop_grad()
        # Jittor basic indexing materializes a separate Var. Propagate a
        # mutation through every retained view so x[0][1].zero_() reaches x.
        _write_index_parent(parent, parent)

    # in-place tensor ops torch code uses heavily (jittor exposes assign()).
    # _ip() preserves grad-tracking: jittor's assign() adopts the source's
    # stop_grad flag, which would freeze a trainable parameter.
    def _ip(self, value):
        # In-place op x.OP_(...) -> x becomes `value` (which usually depends on x,
        # e.g. div_/mul_/add_). assign() ALREADY keeps x grad-connected when `value`
        # is grad-connected, so grad flows through the in-place op (torch parity).
        # But start_grad() RESETS x's grad node and SEVERS that just-built graph
        # (the same start_grad-severing bug behind the DPO/requires_grad fix), which
        # silently zeroed grads through x.div_()/etc (GRPO temperature scaling).
        # So only start_grad if assign actually left x stopped (a constant value like
        # fill_/zero_ on a previously-trainable leaf) -- never on an already-connected x.
        was_trainable = not self.is_stop_grad()
        _write_index_parent(self, value)
        self.assign(value)
        if was_trainable and self.is_stop_grad():
            self.start_grad()
        elif not was_trainable and not self.is_stop_grad():
            self.stop_grad()
        return self
    def _copy_(self, other, non_blocking=False):
        src = other if isinstance(other, Var) else jt.array(other)
        return _ip(self, src.cast(str(self.dtype)) if hasattr(self, "dtype") else src)
    if not hasattr(Var, "copy_"):
        Var.copy_ = _copy_

    # torch's new_*(size, *, dtype=, device=, requires_grad=) factory methods.
    # jittor's native new_ones/new_zeros only take a size, so override to accept
    # torch kwargs (dtype defaults to self's dtype, like torch).
    def _norm_size(args):
        # torch allows new_ones(2,3), new_ones((2,3)), or new_ones(<NanoVector/Size>)
        # -- unwrap any single iterable that isn't itself a scalar int/Var.
        if len(args) == 1 and not isinstance(args[0], (int, jt.Var)) \
                and hasattr(args[0], "__len__"):   # tuple/list/NanoVector/Size
            args = tuple(args[0])
        # torch accepts 0-d int Vars / numpy ints as sizes (e.g. longformer computes
        # dims via torch.div); jittor's factories need plain ints -- coerce.
        return tuple(int(s.item()) if isinstance(s, jt.Var) else int(s) for s in args)
    def _resolve_size(size, kw):
        # torch allows new_ones(2,3), new_ones((2,3)) AND the keyword form
        # new_ones(size=(2,3)) (used by longformer's new_ones(size=mask.size())).
        if not size and "size" in kw:
            return (kw["size"],)
        return size
    def _new_finish(v, device=None, requires_grad=False):
        if _device_is_cpu(device):
            v = _make_cpu_resident(v)
        elif _device_is_cuda(device):
            jt.flags.use_cuda = 1
            v = _make_cuda_resident(v, force=True)
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        return v
    def _new_ones(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(jt.ones(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_zeros(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(jt.zeros(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        # size may be a tuple/list/torch.Size OR a jittor NanoVector (e.g. from
        # x.new_full(x.shape, v)); both are iterable with __len__.
        shp = tuple(int(s) for s in size) if hasattr(size, "__len__") else (int(size),)
        return _new_finish(jt.full(shp, fill_value).cast(dt), device, requires_grad)
    def _new_empty(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(jt.empty(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_tensor(self, data, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        # torch's new_tensor accepts a python list whose elements are 0-d tensors
        # (e.g. centernet_update_head builds start_coord_pre_level by accumulating
        # `_start = _start + batch * area_per_level[level]`, where the indexed term
        # is a scalar). jittor has no 0-d tensors, so those scalars are [1] Vars and
        # jt.array([int, Var, Var, ...]) raises "inhomogeneous shape". Coerce any
        # numel-1 Var element to a python number first so the list is homogeneous.
        if isinstance(data, (list, tuple)):
            def _coerce(v):
                if isinstance(v, jt.Var):
                    return v.item() if v.numel() == 1 else v.tolist()
                if isinstance(v, (list, tuple)):
                    return [_coerce(e) for e in v]
                return v
            data = [_coerce(v) for v in data]
        return _new_finish(jt.array(data).cast(dt), device, requires_grad)
    Var.new_ones = _new_ones
    Var.new_zeros = _new_zeros
    Var.new_full = _new_full
    Var.new_empty = _new_empty
    Var.new_tensor = _new_tensor
    # Override the native methods even when they already exist. Transformers
    # initializes parameters through ``param.data.normal_()/zero_()/fill_()``
    # inside @torch.no_grad(); Jittor's native bound initializers adopt the
    # constant source's stop-grad flag and permanently freeze the parameter.
    Var.fill_ = lambda self, val: _ip(self, jt.ones(self.shape, self.dtype) * val)
    Var.zero_ = lambda self: _ip(self, jt.zeros(self.shape, self.dtype))
    Var.add_ = lambda self, o, alpha=1: _ip(self, self + (o * alpha))
    Var.sub_ = lambda self, o, alpha=1: _ip(self, self - (o * alpha))
    Var.mul_ = lambda self, o: _ip(self, self * o)
    Var.div_ = lambda self, o: _ip(self, self / o)
    # in-place unary math ops (recurrent_gemma uses x.log_(); common torch idioms)
    for _name, _fn in (("log_", jt.log), ("exp_", jt.exp), ("sqrt_", jt.sqrt),
                       ("neg_", lambda x: -x), ("abs_", jt.abs), ("sigmoid_", jt.sigmoid),
                       ("tanh_", jt.tanh), ("reciprocal_", lambda x: 1.0 / x),
                       ("rsqrt_", lambda x: 1.0 / jt.sqrt(x))):
        if not hasattr(Var, _name):
            setattr(Var, _name, (lambda fn: lambda self: _ip(self, fn(self)))(_fn))
    # torch.clamp(input, min=None, max=None) and Tensor.clamp(min=, max=)
    # accept min/max as keyword args, either of which may be None. jittor's
    # native clamp only takes them positionally and rejects the keywords (it
    # also exposes `low`/`high` names, not `min`/`max`). Wrap both the
    # top-level op and the method so torch's keyword form works, while plain
    # positional calls (jittor's own usage) pass straight through unchanged.
    _native_clamp = jt.clamp
    def _clamp(input, min=None, max=None, min_v=None, max_v=None):
        # accept BOTH torch (min/max) and jittor-native (min_v/max_v) kwarg names:
        # this override REPLACES jt.clamp, and jittor's own ops (e.g. nn.hardswish ->
        # jt.clamp(x+3, min_v=0, max_v=6)) call it with min_v/max_v.
        return _native_clamp(input, min if min is not None else min_v,
                             max if max is not None else max_v)
    g.clamp = _clamp
    g.clip = _clamp                      # torch.clip is an alias of torch.clamp
    # torch.clamp_min / clamp_max free functions (3DGS gm:159 clamps distCUDA2)
    g.clamp_min = lambda input, v: _clamp(input, min=v)
    g.clamp_max = lambda input, v: _clamp(input, max=v)
    Var.clamp = lambda self, min=None, max=None, min_v=None, max_v=None: _clamp(self, min, max, min_v, max_v)
    Var.clip = Var.clamp
    Var.clamp_ = lambda self, min=None, max=None, min_v=None, max_v=None: _ip(self, _clamp(self, min, max, min_v, max_v))
    Var.clip_ = Var.clamp_

    def _torch_ne(input, other):
        a = input if isinstance(input, Var) else jt.array(input)
        b = other if isinstance(other, Var) else jt.array(other)
        if str(a.dtype) == "bool":
            a = a.int32()
        if isinstance(b, Var) and str(b.dtype) == "bool":
            b = b.int32()
        diff = (a - b).abs()
        out = diff > 0
        if "float" in str(a.dtype) or (isinstance(b, Var) and "float" in str(b.dtype)):
            try:
                out = out | jt.isnan(a) | jt.isnan(b)
            except Exception:
                pass
        return out

    g.ne = _torch_ne
    g.not_equal = _torch_ne
    Var.ne = lambda self, other: _torch_ne(self, other)
    Var.__ne__ = lambda self, other: _torch_ne(self, other)

    # torch's Tensor.nonzero(as_tuple=False) returns an (N, ndim) index matrix;
    # nonzero(as_tuple=True) instead returns a tuple of ndim 1-D index Vars (one
    # per dimension) -- transformers/diffusers use the tuple form for advanced
    # indexing. jittor's nonzero only returns the matrix and rejects as_tuple.
    _native_nonzero = getattr(jt, "_vj_native_nonzero", jt.nonzero)
    def _nonzero(self, as_tuple=False, **kw):
        idx = _native_nonzero(self)
        if not as_tuple:
            return idx
        # idx is (N, ndim); split into one 1-D index Var per dimension. For a
        # 0/1-D input torch still returns a 1-tuple of the flat indices.
        ndim = idx.shape[1] if idx.ndim == 2 else 1
        if idx.ndim != 2:
            return (idx.reshape(-1),)
        return tuple(idx[:, d] for d in range(ndim))
    Var.nonzero = _nonzero
    g.nonzero = lambda input, as_tuple=False, **kw: _nonzero(input, as_tuple=as_tuple)
    # torch-compat: torch.argwhere(input) / Tensor.argwhere() -> the indices of the
    # nonzero elements as an (N, ndim) matrix (identical to nonzero(as_tuple=False)).
    if not hasattr(g, "argwhere"):
        g.argwhere = lambda input: _nonzero(input, as_tuple=False)
    if not hasattr(Var, "argwhere"):
        Var.argwhere = lambda self: _nonzero(self, as_tuple=False)
    Var.normal_ = lambda self, mean=0.0, std=1.0, generator=None: _ip(self, jt.normal(float(mean), float(std), self.shape).cast(str(self.dtype)))
    Var.uniform_ = lambda self, a=0.0, b=1.0, generator=None: _ip(self, (jt.rand(self.shape)*(b-a)+a).cast(str(self.dtype)))

    # torch tensors are hashable by identity (they define __eq__ elementwise but
    # keep an id-based __hash__). jittor's Var defines __eq__ and so becomes
    # unhashable, breaking `var in set_of_vars` / dict keys in peft. Restore an
    # identity hash. Membership tests use hash first, then `is`, so this matches
    # torch semantics without invoking elementwise __eq__.
    if Var.__hash__ is None:
        Var.__hash__ = lambda self: id(self)

    # element_size / nelement (torch byte-accounting helpers)
    _DTYPE_BYTES = {
        "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
        "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
        "uint16": 2, "uint32": 4, "uint64": 8, "bool": 1,
        "float8_e4m3fn": 1, "float8_e5m2": 1,
        "complex64": 8, "complex128": 16,
    }
    if not hasattr(Var, "element_size"):
        def _element_size(self):
            return _DTYPE_BYTES.get(str(self.dtype), 4)
        Var.element_size = _element_size
    if not hasattr(Var, "nelement"):
        Var.nelement = lambda self: int(self.numel())

    # torch dtype predicates on the tensor itself. transformers computes
    # model.dtype via `next(p.dtype for p in params if p.is_floating_point())`,
    # so save_pretrained needs these. jittor has no native complex, so
    # is_complex is always False here.
    _FP_DTYPES = {"float16", "float32", "float64", "bfloat16",
                  "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2",
                  "float8_e5m2fnuz", "float8_e8m0fnu", "float4_e2m1fn_x2"}
    if not hasattr(Var, "is_floating_point"):
        Var.is_floating_point = lambda self: str(self.dtype) in _FP_DTYPES
    if not hasattr(Var, "is_complex"):
        Var.is_complex = lambda self: str(self.dtype) in ("complex64", "complex128")
    if not hasattr(Var, "is_signed"):
        Var.is_signed = lambda self: str(self.dtype) not in (
            "bool", "uint8", "uint16", "uint32", "uint64")

    # torch storage introspection: peft/safetensors call tensor.storage()
    # .data_ptr() / .untyped_storage().nbytes() to detect shared/tied weights.
    # jittor has no exposed storage object; expose identity-based stand-ins so
    # save_pretrained's tied-weight detection works (each Var is its own storage).
    class _Storage:
        def __init__(self, var):
            self._var = var
        def data_ptr(self):
            return id(self._var)
        def size(self):
            return int(self._var.numel())
        def nbytes(self):
            return int(self._var.numel()) * _DTYPE_BYTES.get(str(self._var.dtype), 4)
    if not hasattr(Var, "storage"):
        Var.storage = lambda self: _Storage(self)
    if not hasattr(Var, "untyped_storage"):
        Var.untyped_storage = lambda self: _Storage(self)
    if not hasattr(Var, "data_ptr"):
        Var.data_ptr = lambda self: id(self)
    # torch tensors expose is_contiguous()/contiguous(); jittor Vars are always
    # contiguous in the sense safetensors cares about.
    if not hasattr(Var, "is_contiguous"):
        Var.is_contiguous = lambda self, *a, **k: True

    # cumsum: ACL's aclnnCumsum SEGFAULTS on bool input (transformers builds
    # position_ids via mask.cumsum(-1)). torch.cumsum promotes bool/uint8 to
    # int64 anyway, so cast before the native op to match torch AND avoid the
    # crash. Override both torch.cumsum and Var.cumsum (g IS the jittor module).
    _native_cumsum = jt.cumsum
    def _assign_out(out, value):
        out.assign(value)
        _write_index_parent(out, out)
        return out

    def _cumsum(x, dim=-1, dtype=None, out=None, **kw):
        if isinstance(x, jt.Var) and str(x.dtype) in ("bool", "uint8"):
            x = x.cast("int64")
        r = _native_cumsum(x, dim)
        if dtype is not None:
            r = r.cast(_dtype_to_str(dtype))
        if out is not None:
            return _assign_out(out, r)
        return r
    g.cumsum = _cumsum
    Var.cumsum = lambda self, dim=-1, dtype=None, out=None, **kw: _cumsum(self, dim, dtype, out=out)
    # cumprod has the same ACL fragility; guard it the same way if present.
    if hasattr(jt, "cumprod"):
        _native_cumprod = jt.cumprod
        def _cumprod(x, dim=-1, dtype=None, out=None, **kw):
            if isinstance(x, jt.Var) and str(x.dtype) in ("bool", "uint8"):
                x = x.cast("int64")
            r = _native_cumprod(x, dim)
            if dtype is not None:
                r = r.cast(_dtype_to_str(dtype))
            if out is not None:
                return _assign_out(out, r)
            return r
        g.cumprod = _cumprod
        Var.cumprod = lambda self, dim=-1, dtype=None, out=None, **kw: _cumprod(self, dim, dtype, out=out)

    # bitwise/logical operators torch supports on tensors
    if not hasattr(Var, "__invert__"):
        def _invert(self):
            if str(self.dtype) == "bool":
                return self.logical_not()
            return jt.logical_not(self) if str(self.dtype) == "bool" else (-self - 1)
        Var.__invert__ = _invert

    def _device(self):
        # Inside a `with torch.device("meta")` block (transformers'
        # from_pretrained), report "meta" so its meta-context detection
        # fires and eager weight init is skipped. See device.__enter__.
        if _DEVICE_CTX_STACK:
            return _DEVICE_CTX_STACK[-1]
        # Report the Var's ACTUAL memory residency (matches jtorch's C++
        # is_cpu()/device()): a Var built/migrated to host -- e.g. via
        # torch.zeros(device='cpu') or .cpu() -- is "cpu" even while the
        # global use_cuda flag is 1. Only fall back to the global flag when
        # CUDA is on and the Var is genuinely device-resident.
        if (jt.flags.use_cuda or getattr(jt.compiler, "has_acl", 0)):
            return device("cpu") if _var_is_cpu_resident(self) else device("cuda", 0)
        return device("cpu")
    Var.device = property(_device)

    _orig_getitem = getattr(Var, "__getitem__", None)
    if _orig_getitem is not None and not getattr(_orig_getitem, "_torch_cpu_residency", False):
        def _is_basic_index(index):
            if isinstance(index, tuple):
                return all(_is_basic_index(item) for item in index)
            if index is None or index is Ellipsis or isinstance(index, slice):
                return True
            return isinstance(index, numbers.Integral) and not isinstance(index, (bool, np.bool_))

        def _torch_getitem(self, slices):
            out = _orig_getitem(self, slices)
            if isinstance(out, Var) and _var_has_cpu_residency_hint(self):
                out = _mark_cpu_like(out, self)
            # Only basic indexing returns a view in PyTorch. Retaining the
            # parent for advanced-index copies creates optimizer-state chains
            # across Gaussian Splatting densification generations.
            if isinstance(out, Var) and _is_basic_index(slices):
                try:
                    out._torch_index_parent = self
                    out._torch_index_slices = slices
                except Exception:
                    pass
            return out
        _torch_getitem._torch_cpu_residency = True
        Var.__getitem__ = _torch_getitem

    for _op_name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                     "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__"):
        _orig_op = getattr(Var, _op_name, None)
        if _orig_op is None or getattr(_orig_op, "_torch_cpu_residency", False):
            continue
        def _make_cpu_binary_wrapper(orig):
            def _wrapped(self, other):
                out = orig(self, other)
                return _mark_cpu_like(out, self, other)
            _wrapped._torch_cpu_residency = True
            return _wrapped
        setattr(Var, _op_name, _make_cpu_binary_wrapper(_orig_op))

    # torch's Tensor.data returns a detached *tensor* (and is assignable:
    # `param.data = new_tensor`). jittor's native Var.data returns a numpy
    # ndarray, breaking `param.data.to(...)`. Override to torch semantics.
    if not getattr(Var, "_data_wrapped", False):
        def _data_get(self):
            # torch's .data SHARES storage with the param: in-place writes
            # (`p.data[:] = x`, `p.data.copy_(x)`, `p.data.normal_()`) write
            # through. Returning self preserves that (a detached copy would
            # silently drop the write — e.g. e2cnn inits weights via
            # `self.weights.data[:] = ...`). Forward *values* are identical to
            # detach(); only grad-tracking-on-reads differs (rare in practice).
            return self
        def _data_set(self, value):
            src = value if isinstance(value, Var) else jt.array(value)
            was_trainable = not self.is_stop_grad()
            self.assign(src)
            if was_trainable:
                self.start_grad()
        Var.data = property(_data_get, _data_set)
        Var._data_wrapped = True

    # jittor's native Var.__reduce__ is `(Var, (self.data,))`, which assumes
    # .data is a numpy ndarray. The shim above redefines .data to return a Var,
    # so the stock reduce recurses forever (pickle re-reduces the Var arg). Make
    # Vars picklable by serializing through numpy + dtype (needed for Ray to
    # ship token tensors to reward actors, torch.multiprocessing, etc.).
    if not getattr(Var, "_reduce_wrapped", False):
        Var.__reduce__ = lambda self: (
            _rebuild_var_from_numpy, (self.numpy(), str(self.dtype)))
        Var._reduce_wrapped = True

    # Leaf registry for the no-optimizer backward() path (below): torch's
    # loss.backward() accumulates grads into the .grad of every leaf that
    # requires grad, but jittor has no graph-walk to recover those leaves. So
    # track Vars whose grad was explicitly enabled through the torch-facing
    # API (requires_grad=True / requires_grad_()). Keyed by id() to dedupe;
    # jittor Vars are not weak-referenceable, so we hold strong refs (leaf
    # params are long-lived anyway) and prune entries that drop stop-grad.
    if not hasattr(jt, "_torch_leaf_params"):
        jt._torch_leaf_params = {}
    def _register_leaf(v):
        _torch_register_leaf(v)

    # Override requires_grad with a Python property even though jittor exposes a
    # native getset descriptor: the native setter maps directly to start_grad/
    # stop_grad (identical semantics), but we additionally register the Var as a
    # leaf so the no-optimizer loss.backward() path (below) can find it. This is
    # behavior-preserving for the getter/setter; it only adds leaf bookkeeping.
    if not isinstance(Var.__dict__.get("requires_grad"), property):
        def _rg_get(self):
            try:
                return not self.is_stop_grad()
            except Exception:
                return False
        def _rg_set(self, v):
            # CRITICAL: jittor's start_grad()/stop_grad() RESET the Var's grad node,
            # which SEVERS any already-built autograd graph that depends on it --
            # even start_grad() on an already-grad Var wipes prior computations
            # ([2,2,2]->[0,0,0]). torch's requires_grad_ is idempotent and never
            # severs existing graphs. So only flip when the flag ACTUALLY changes;
            # a no-op set (the common case, e.g. peft set_adapter re-asserting
            # requires_grad=True every disable_adapter exit) must NOT touch the node.
            v = bool(v)
            fsdp_entry = getattr(self, "_jittor_fsdp2_entry", None)
            fsdp_state = getattr(self, "_jittor_fsdp2_state", None)
            if fsdp_entry is not None and fsdp_state is not None:
                fsdp_entry.requires_grad = v
                for peer in (getattr(fsdp_entry, "shard", None),
                             getattr(fsdp_entry, "full_param", None)):
                    if not isinstance(peer, Var) or peer is self:
                        continue
                    if v:
                        if peer.is_stop_grad():
                            peer.start_grad()
                        _register_leaf(peer)
                    elif not peer.is_stop_grad():
                        peer.stop_grad()
                if getattr(fsdp_state, "true_fsdp_flat", False):
                    flat = getattr(fsdp_state, "true_fsdp_flat_shard", None)
                    any_trainable = any(getattr(entry, "requires_grad", True)
                                        for entry in fsdp_state.true_fsdp_params)
                    if isinstance(flat, Var):
                        if any_trainable:
                            if flat.is_stop_grad():
                                flat.start_grad()
                            _register_leaf(flat)
                        elif not flat.is_stop_grad():
                            flat.stop_grad()
            if v:
                if self.is_stop_grad():
                    self.start_grad()
                _register_leaf(self)
            else:
                if not self.is_stop_grad():
                    self.stop_grad()
        Var.requires_grad = property(_rg_get, _rg_set)

    def requires_grad_(self, v=True):
        self.requires_grad = v
        if v:
            _register_leaf(self)
        return self
    Var.requires_grad_ = requires_grad_

    # ------------------------------------------------------------------
    # torch-style autograd bridge: loss.backward() / param.grad
    # ------------------------------------------------------------------
    # jittor has no tensor-level backward(); gradients flow through
    # `optimizer.backward(loss)` then `optimizer.step()`. torch/accelerate
    # instead call `loss.backward()`, read/modify `param.grad` (grad clipping),
    # then call `optimizer.step()` with no loss. We bridge the two:
    #   * loss.backward(): route to the active optimizer's backward(loss),
    #     which fills pg["grads"]; then expose those grad Vars on each param.
    #   * param.grad: getter returns the optimizer-held grad Var (so in-place
    #     clipping mutates the very Var that step() consumes); setter stores it.
    def _fill_opt_grads(opt, grad_by_id, filled_param_ids=None):
        # Replicate the grad-storage half of jittor's Optimizer.backward() but
        # from an already-computed {id(param): grad} map (so a SINGLE jt.grad
        # pass feeds every optimizer + every leaf — no N-times-repeated backward).
        # Honors the per-optimizer __zero_grad flag (post_step zeros it, so the
        # next backward overwrites rather than accumulates) and tolerates a param
        # whose shape changed (3DGS densify replaces params) by replacing — not
        # .update()-ing — the stored grad Var.
        zero = getattr(opt, "_Optimizer__zero_grad", True)
        if filled_param_ids is None:
            filled_param_ids = set()
        for pg in opt.param_groups:
            grads_list = pg.get("grads")
            if grads_list is None:
                grads_list = pg["grads"] = [None] * len(pg["params"])
            for i, p in enumerate(pg["params"]):
                if not isinstance(p, Var) or p.is_stop_grad():
                    continue
                g = grad_by_id.get(id(p))
                if g is None:
                    continue
                if id(p) in filled_param_ids:
                    while len(grads_list) <= i:
                        grads_list.append(None)
                    grads_list[i] = getattr(p, "_torch_grad", None)
                    continue
                g = g.stop_grad()
                existing = grads_list[i] if i < len(grads_list) else None
                if not isinstance(existing, Var):
                    existing = getattr(p, "_torch_grad", None)
                if isinstance(existing, Var) and list(existing.shape) == list(g.shape):
                    if not zero:
                        g = g + existing
                    existing.update(g)
                    stored = existing
                else:
                    stored = g
                while len(grads_list) <= i:
                    grads_list.append(None)
                grads_list[i] = stored
                object.__setattr__(p, "_torch_grad", stored)
                filled_param_ids.add(id(p))
        object.__setattr__(opt, "_Optimizer__zero_grad", False)
        try:
            opt._build_grad_map()
        except Exception:
            pass

    def _optimizer_maybe_has_fsdp_params(opt):
        for _pg in getattr(opt, "param_groups", []):
            for _p in _pg.get("params", []):
                if getattr(_p, "_jittor_fsdp2_state", None) is not None:
                    return True
        return False

    def _backward(self, gradient=None, retain_graph=False, create_graph=False, **kw):
        # torch defaults retain_graph to create_graph. In the common
        # loss.backward() case both are false, so the graph must be freed.
        retain_graph = bool(create_graph) if retain_graph is None else bool(retain_graph)
        # Materialize the loss's FORWARD graph before computing gradients. A custom
        # CUDA-ext Function (3DGS rasterizer / fused-ssim) writes its outputs
        # out-of-band; if the forward is left lazy, jt.grad recomputes that
        # subgraph during the backward pass and the ext's lazy "empty/full"
        # factory op re-runs WITHOUT the kernel's writes -> garbage/NaN loss
        # (proven: a plain float(loss) before backward makes train.py finite).
        # Forcing the forward to settle once here decouples it from the grad pass.
        try:
            self.sync()
        except Exception:
            pass
        # Collect EVERY live optimizer (torch allows several at once — 3DGS uses a
        # Gaussian Adam + an exposure Adam; routing to just _current_optimizer
        # left the other's params with .grad=None -> KeyError 'grads' in step()).
        reg = getattr(jt, "_active_optimizers", None)
        opts = []
        if reg:
            alive = []
            for r in reg:
                o = r() if callable(r) else r
                if o is not None:
                    alive.append(r)
                    opts.append(o)
            reg[:] = alive
        # The union of grad targets: every optimizer's trainable params, plus
        # retain_grad'd non-leaves (3DGS's screenspace `means2D`, read by
        # densification as .grad). Without optimizers, fall back to the global
        # leaf registry so standalone Tensor.backward() still works.
        #
        # When optimizers are live, their current param_groups are authoritative:
        # torch code such as 3DGS replaces parameters during densification, and
        # stale strong refs in the registry would otherwise keep old params and
        # their Jittor graphs alive until OOM.
        fsdp_opts = [o for o in opts if _optimizer_maybe_has_fsdp_params(o)]
        if fsdp_opts:
            try:
                from . import torch_fsdp2_compat as _fsdp2_backward
            except Exception:
                _fsdp2_backward = None
        else:
            _fsdp2_backward = None
        fsdp_opt_ids = {id(o) for o in fsdp_opts} if _fsdp2_backward is not None else set()
        leaf_map = {}
        opt_ids = set()
        filled_param_ids = set()
        for o in opts:
            for pg in getattr(o, "param_groups", []):
                for p in pg.get("params", []):
                    if not isinstance(p, Var) or p.is_stop_grad():
                        continue
                    if _fsdp2_backward is not None and _fsdp2_backward.is_fsdp_managed_param(p):
                        opt_ids.add(id(p))
                        continue
                    leaf_map.setdefault(id(p), p)
                    opt_ids.add(id(p))
        if _fsdp2_backward is not None and fsdp_opts:
            for p in _fsdp2_backward.collect_fsdp_full_params_for_backward(fsdp_opts):
                if isinstance(p, Var) and not p.is_stop_grad():
                    leaf_map.setdefault(id(p), p)
                    opt_ids.add(id(p))
        retained = getattr(jt, "_torch_retained", None)
        retained_ids = set()
        if retained:
            for v in list(retained.values()):
                if isinstance(v, Var) and not v.is_stop_grad():
                    leaf_map.setdefault(id(v), v)
                    retained_ids.add(id(v))
        if opts:
            _torch_prune_leaf_registry(opt_ids | retained_ids)
        else:
            _torch_prune_leaf_registry()
            for v in list(jt._torch_leaf_params.values()):
                if isinstance(v, Var) and not v.is_stop_grad():
                    leaf_map.setdefault(id(v), v)
        if not leaf_map:
            return None
        leaves = list(leaf_map.values())
        # torch leaves a disconnected target at grad=None. Keep jt.grad's
        # historical zero-materialization untouched and use the compatibility
        # core entry point that preserves missing gradients explicitly.
        grads = jt.core.grad_optional(self, leaves, retain_graph)
        grad_by_id = {}
        for p, gr in zip(leaves, grads):
            if gr is None:
                continue
            grad_by_id[id(p)] = gr
            if id(p) not in opt_ids:
                # non-optimizer leaf (retain_grad screenspace etc.): accumulate
                # onto .grad like torch (zeroed externally / per render).
                prev = getattr(p, "_torch_grad", None)
                object.__setattr__(p, "_torch_grad",
                                   gr if prev is None else (prev + gr))
        # fill each optimizer's pg["grads"] so its step(loss=None) consumes them
        if _fsdp2_backward is not None and fsdp_opts:
            _fsdp2_backward.fill_fsdp_optimizer_grads_from_grad_map(fsdp_opts, grad_by_id)
        for o in opts:
            if _fsdp2_backward is not None and id(o) in fsdp_opt_ids \
                    and not _fsdp2_backward.optimizer_has_non_fsdp_params(o):
                continue
            _fill_opt_grads(o, grad_by_id, filled_param_ids)
        # retain_grad is per-forward in torch; clear so the next iteration's fresh
        # screenspace tensor doesn't leak (jittor Vars aren't weak-referenceable).
        if retained:
            retained.clear()
        return None
    Var.backward = _backward

    def _grad_get(self):
        # _backward publishes _torch_grad on every leaf (for optimizer params it
        # points AT pg["grads"][i], so in-place grad clipping mutates the very Var
        # step() consumes). Fall back to any live optimizer's grad map if a param
        # hasn't gone through _backward yet.
        g = getattr(self, "_torch_grad", None)
        if g is not None:
            return g
        for r in getattr(jt, "_active_optimizers", None) or []:
            o = r() if callable(r) else r
            if o is None:
                continue
            try:
                return o.find_grad(self)
            except Exception:
                pass
        return None
    def _grad_set(self, value):
        object.__setattr__(self, "_torch_grad", value)
        fsdp_entry = getattr(self, "_jittor_fsdp2_entry", None)
        fsdp_role = getattr(self, "_jittor_fsdp2_role", None)
        if fsdp_entry is not None:
            try:
                if value is None:
                    fsdp_entry.last_grad = None
                    fsdp_entry.full_public_grad = None
                    object.__setattr__(fsdp_entry.shard, "_torch_grad", None)
                    full = getattr(fsdp_entry, "full_param", None)
                    if full is not None and full is not self:
                        object.__setattr__(full, "_torch_grad", None)
                elif fsdp_role != "full":
                    fsdp_entry.last_grad = value
                    full = getattr(fsdp_entry, "full_param", None)
                    if full is not None and full is not self:
                        object.__setattr__(full, "_torch_grad", None)
            except Exception:
                pass
        # Write through by identity so step() sees manual grad assignment and,
        # critically, p.grad=None cannot leave an old optimizer slot behind.
        for r in getattr(jt, "_active_optimizers", None) or []:
            o = r() if callable(r) else r
            if o is None:
                continue
            changed = False
            for pg in getattr(o, "param_groups", []):
                params = list(pg.get("params", []))
                for i, p in enumerate(params):
                    same_fsdp_entry = fsdp_entry is not None and getattr(
                        p, "_jittor_fsdp2_entry", None) is fsdp_entry
                    if p is not self and not same_fsdp_entry:
                        continue
                    if fsdp_role == "full" and value is not None and p is not self:
                        continue
                    if value is None:
                        grads = pg.get("grads")
                        if grads is not None and i < len(grads):
                            grads[i] = None
                    else:
                        grads = pg.get("grads")
                        if grads is None:
                            grads = pg["grads"] = [None] * len(params)
                        while len(grads) < len(params):
                            grads.append(None)
                        grads[i] = value
                    changed = True
            if changed:
                try:
                    object.__setattr__(o, "_grad_map", {})
                    if value is None:
                        object.__setattr__(o, "_torch_backward_advanced_n_step", False)
                    if value is not None:
                        object.__setattr__(o, "_Optimizer__zero_grad", False)
                except Exception:
                    pass
    Var.grad = property(_grad_get, _grad_set)

    # torch's `is_leaf`: True for tensors not produced by a grad-tracked op
    # (user-created params/inputs). jittor has no autograd-graph leaf concept;
    # treat every Var as a leaf so peft's `if param.is_leaf:` guards pass.
    if not hasattr(Var, "is_leaf"):
        Var.is_leaf = property(lambda self: True)
    # torch's nested-tensor flag; jittor has no nested tensors -> always False.
    if not hasattr(Var, "is_nested"):
        Var.is_nested = property(lambda self: False)
    # torch's `grad_fn` is None for leaves; libs check `t.grad_fn is None`.
    if not hasattr(Var, "grad_fn"):
        Var.grad_fn = property(lambda self: None)
    # torch's retain_grad() marks a NON-leaf tensor so its .grad is populated
    # after backward (normally only leaves keep .grad). 3DGS relies on this for
    # the screenspace `means2D` tensor (`zeros_like(xyz)+0` then retain_grad()),
    # whose .grad drives densification. Register into a per-forward set the
    # _backward pass includes as a grad target; cleared each backward so the
    # next iteration's fresh tensor doesn't accumulate (jittor Vars can't be
    # weak-ref'd, so a persistent dict would leak one Var per iteration).
    if not hasattr(jt, "_torch_retained"):
        jt._torch_retained = {}
    def _retain_grad(self):
        try:
            jt._torch_retained[id(self)] = self
        except Exception:
            pass
        return self
    Var.retain_grad = _retain_grad

    def _to(self, *args, **kwargs):
        ds = None
        dev = None
        copy = bool(kwargs.get("copy", False))
        # device passed as a keyword (torch's .to(device=..., dtype=...))
        if "device" in kwargs:
            dev = kwargs["device"]
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, dtype):
                ds = a.name
            elif isinstance(a, device):
                dev = a
            elif isinstance(a, Var):
                # .to(other) copies other's dtype AND device.
                ds = str(a.dtype)
                dev = a.device
            elif isinstance(a, str):
                bare = a.replace("torch.", "")
                if bare in dtype._registry:
                    ds = bare
                elif bare.split(":")[0] in ("cpu", "cuda", "npu"):
                    dev = bare
        if ds is not None:
            out = self.cast(ds) if copy else _cast_if_needed(self, ds)
        else:
            out = self.clone() if copy else self
        # Honor an explicit device= target by migrating residency. device=None
        # (the common .to(dtype) call) leaves placement on the global default.
        if _device_is_cpu(dev):
            out = _make_cpu_resident(out)
        elif _device_is_cuda(dev):
            out = _make_cuda_resident(out, force=True)
        if getattr(self, "_torch_0d", False):
            out._torch_0d = True
        return out
    Var.to = _to

    # Jittor stores torch 0-D scalars as one-element Vars. Preserve a lightweight
    # provenance marker through the copy-like methods used before host export,
    # then expose the scalar shape only at the Python/NumPy boundary.
    _native_detach = Var.detach
    def _var_detach(self):
        out = _native_detach(self)
        if getattr(self, "_torch_0d", False):
            out._torch_0d = True
        return out
    Var.detach = _var_detach

    _native_numpy = Var.numpy
    def _var_numpy(self, *args, **kwargs):
        out = _native_numpy(self, *args, **kwargs)
        if getattr(self, "_torch_0d", False) and getattr(out, "size", 0) == 1:
            return out.reshape(())
        return out
    Var.numpy = _var_numpy
    Var.tolist = lambda self: (self.item() if getattr(self, "_torch_0d", False)
                               else self.numpy().tolist())

    # torch's Tensor.cpu()/.cuda() MIGRATE the tensor's residency (native exts
    # check tensor.is_cpu()). jittor's base Var.cpu just clones (stays on GPU)
    # and Var.cuda only flips the global flag, so override both to actually move
    # the data: .cpu() rebuilds the Var under the host allocator, .cuda() under
    # the device allocator. Var.location()/jtorch's C++ is_cpu() then agree.
    def _var_cpu(self, *a, **k):
        out = _make_cpu_resident(self)
        try:
            out._jittor_torch_force_cpu = True
            if getattr(self, "_torch_0d", False):
                out._torch_0d = True
        except Exception:
            pass
        return out
    Var.cpu = _var_cpu
    def _var_cuda(self, device=None, *a, **k):
        jt.flags.use_cuda = 1
        out = _make_cuda_resident(self, force=True)
        if getattr(self, "_torch_0d", False):
            out._torch_0d = True
        return out
    Var.cuda = _var_cuda

    # ---- integer/float dtype cast methods (torch parity) ----
    # jittor aliases Var.long = Var.int32 and Var.int = Var.int32, so BOTH
    # .long() and (from a non-int32 input) the torch dtype is wrong: torch's
    # .long() is int64, .int() is int32. It also lacks .short()/.byte()/.char().
    # Pin every cast method to torch's EXACT dtype. (.bool()/.half()/.double()/
    # .float()/.float32()/.int64()/... were already correct, but reassigning
    # them through .cast is behavior-identical and keeps the mapping in one place.)
    _CAST_METHOD_DTYPE = {
        "byte": "uint8", "char": "int8", "short": "int16", "int": "int32",
        "long": "int64", "half": "float16", "float": "float32",
        "double": "float64", "bfloat16": "bfloat16", "bool": "bool",
    }
    def _cast_if_needed(tensor, dtype):
        return tensor if str(tensor.dtype) == dtype else tensor.cast(dtype)

    for _mname, _mdt in _CAST_METHOD_DTYPE.items():
        setattr(Var, _mname, (lambda dt: lambda self: _cast_if_needed(self, dt))(_mdt))

    # torch's Tensor.type(): with a dtype/typed-tensor-name it casts; with no
    # argument it returns the torch type-NAME string ('torch.FloatTensor' ...).
    _DTYPE_TO_TYPENAME = {
        "float32": "torch.FloatTensor", "float64": "torch.DoubleTensor",
        "float16": "torch.HalfTensor", "bfloat16": "torch.BFloat16Tensor",
        "int64": "torch.LongTensor", "int32": "torch.IntTensor",
        "int16": "torch.ShortTensor", "int8": "torch.CharTensor",
        "uint8": "torch.ByteTensor", "bool": "torch.BoolTensor",
    }
    _TYPENAME_TO_DTYPE = {v: k for k, v in _DTYPE_TO_TYPENAME.items()}
    _TYPENAME_TO_DTYPE.update({v.replace("torch.", "torch.cuda."): k
                               for k, v in _DTYPE_TO_TYPENAME.items()})
    def _var_type(self, dst_type=None, non_blocking=False, **kw):
        if dst_type is None:
            return _DTYPE_TO_TYPENAME.get(str(self.dtype), "torch.FloatTensor")
        if isinstance(dst_type, str) and dst_type in _TYPENAME_TO_DTYPE:
            return _cast_if_needed(self, _TYPENAME_TO_DTYPE[dst_type])
        ds = _dtype_to_str(dst_type)
        return _cast_if_needed(self, ds) if ds is not None else self
    Var.type = _var_type

    # ---- torch-parity binary-op type promotion ----
    # jittor's native arithmetic operators keep the LEFT/narrower operand's dtype
    # for mixed-dtype Var op Var (int32+int64 -> int32, float32+float64 -> float32,
    # float16+int64 -> float32, uint8+int8 -> int8), silently losing range/precision
    # vs torch. torch instead promotes BOTH operands to result_type, then computes.
    # Wrap the affected operators to do exactly that: when the other operand is a Var
    # of a DIFFERENT dtype, cast both to the promoted dtype and call the original
    # native op (now same-dtype -> jittor returns the promoted dtype). All other
    # paths -- matching dtypes, or a Python scalar (jittor already matches torch:
    # int scalar keeps the int dtype, float scalar lifts int->float32) -- pass
    # straight through to the native op, so nothing else changes.
    # True division ('/') has its OWN rule (always float) and is wrapped separately
    # just below; the operators wrapped here follow the plain promotion lattice.
    # jittor's native binary ops ALSO corrupt unsigned dtypes even when both
    # operands match (uint8+uint8 -> int8, uint16+uint16 -> int16) -- a C++
    # binary_dtype_infer quirk we cannot touch. So the wrapper post-corrects the
    # native result to the torch-expected dtype whenever they differ, which both
    # restores unsigned results and double-guards the mixed-dtype promotion.
    def _complex_scalar_var(value):
        # Python/NumPy complex scalars are not accepted by Jittor's automatic
        # Var converter. Materialize the torch-default complex64 scalar first;
        # the actual arithmetic remains a normal device op.
        return jt.array(np.asarray([value], dtype=np.complex64))

    def _make_promoting_op(opname, reflected):
        native = Var.__dict__.get(opname)
        if native is None:
            return None
        def _op(self, other):
            if isinstance(other, (complex, np.complexfloating)):
                other = _complex_scalar_var(other)
            if isinstance(other, Var):
                da, db = str(self.dtype), str(other.dtype)
                if da == db and not da.startswith("uint"):
                    return native(self, other)
                res = g._torch_promote_pair(da, db)
                a = self if da == res else self.cast(res)
                b = other if db == res else other.cast(res)
                out = native(a, b)
                # native may still mis-infer (unsigned -> signed); fix it up.
                if isinstance(out, Var) and str(out.dtype) != res:
                    out = out.cast(res)
                return out
            # torch defers numeric ops against a Python sequence to the sequence's
            # own protocol: `Tensor.__mul__([x])` / `__rmul__([x])` return
            # NotImplemented, so `[x] * t` becomes list-repeat (via Tensor.__index__)
            # and `t * [x]` raises. jittor's native op would instead broadcast the
            # list into a Var (e.g. `[tok] * grid.prod()` -> Var, breaking ms-swift's
            # `_extend_tokens` list concatenation). Match torch: defer to the sequence.
            if isinstance(other, (list, tuple)):
                return NotImplemented
            return native(self, other)
        _op.__name__ = opname
        return _op
    # (opname, reflected?) -- reflected ops receive the *other* operand as the left
    # value, but promotion is symmetric so the same body is correct.
    for _opn, _refl in [("__add__", False), ("__radd__", True),
                        ("__sub__", False), ("__rsub__", True),
                        ("__mul__", False), ("__rmul__", True),
                        ("__floordiv__", False), ("__rfloordiv__", True),
                        ("__mod__", False), ("__rmod__", True),
                        ("__pow__", False), ("__rpow__", True)]:
        _wrapped = _make_promoting_op(_opn, _refl)
        if _wrapped is not None:
            setattr(Var, _opn, _wrapped)

    # True division ('/') is the documented special case: torch ALWAYS yields a
    # float. The result dtype is result_type(a, b) when that is already floating
    # (so float16/int64 -> float16, float32/float64 -> float64), otherwise the
    # default float dtype (so every integral pair, incl. int64/int32 and int8/int8,
    # -> float32). jittor instead follows numpy's "int -> float of matching width"
    # (int64/int32 -> float64, int8/int8 -> float16, float16/int64 -> float64),
    # which loses torch parity. Cast operands to the torch target float, then div.
    def _truediv_target(da, db):
        r = g._torch_promote_pair(da, db)
        if r.startswith(("float", "bfloat", "complex")):
            return r
        return _dtype_to_str(g.get_default_dtype()) or "float32"
    def _scalar_dtype_name(x):
        if isinstance(x, bool):
            return "bool"
        if isinstance(x, int):
            return "int64"
        if isinstance(x, float):
            return _dtype_to_str(g.get_default_dtype()) or "float32"
        if isinstance(x, complex):
            return "complex64"
        return None
    def _make_truediv(opname):
        native = Var.__dict__.get(opname)
        if native is None:
            return None
        def _op(self, other):
            if isinstance(other, (complex, np.complexfloating)):
                other = _complex_scalar_var(other)
            if isinstance(other, Var):
                da, db = str(self.dtype), str(other.dtype)
                if da == db and da.startswith(("float", "bfloat", "complex")):
                    return native(self, other)
                tgt = _truediv_target(da, db)
                a = self if da == tgt else self.cast(tgt)
                b = other if db == tgt else other.cast(tgt)
                out = native(a, b)
                if isinstance(out, Var) and str(out.dtype) != tgt:
                    out = out.cast(tgt)
                return out
            # python sequence: defer to it (torch returns NotImplemented), matching
            # the integer-op behaviour above.
            if isinstance(other, (list, tuple)):
                return NotImplemented
            sd = _scalar_dtype_name(other)
            if sd is not None:
                tgt = _truediv_target(str(self.dtype), sd)
                src_dt = str(self.dtype)
                # PyTorch's Python-float scalar division keeps the result dtype
                # but uses the scalar value with enough precision to differ from
                # division by a float32 tensor by 1 ulp in common cases. 3DGS hits
                # both uint8 image normalization and RGB2SH float32/C0 this way.
                use_wide = sd.startswith("float") and src_dt != "float64"
                calc_dt = "float64" if use_wide else tgt
                a = self if src_dt == calc_dt else self.cast(calc_dt)
                b = jt.array(other, dtype=calc_dt) if use_wide else other
                out = native(a, b)
                if isinstance(out, Var) and str(out.dtype) != tgt:
                    out = out.cast(tgt)
                return out
            return native(self, other)
        _op.__name__ = opname
        return _op
    for _opn in ("__truediv__", "__rtruediv__"):
        _w = _make_truediv(_opn)
        if _w is not None:
            setattr(Var, _opn, _w)

    # Jittor Vars do not expose PyTorch-style strided non-contiguous storage;
    # materialized op outputs are already laid out for their logical shape. The
    # old fake torch hook from misc.py implemented contiguous() as clone(), which
    # adds avoidable graph nodes and copies in PyTorch code that calls
    # transpose(...).contiguous() before export or parameter construction.
    Var.contiguous = lambda self: self
    # torch's Tensor.is_cuda / .is_cpu report the tensor's ACTUAL residency.
    # A Var built/migrated to host (torch.zeros(device='cpu'), .cpu()) is on the
    # CPU even under global use_cuda=1, so read Var.location() rather than the
    # global flag (matches jtorch's C++ is_cuda()/is_cpu()). When CUDA is off
    # everything is host-resident.
    def _is_cuda(self):
        if not (jt.flags.use_cuda or getattr(jt.compiler, "has_acl", 0)):
            return False
        return not _var_is_cpu_resident(self)
    Var.is_cuda = property(_is_cuda)
    Var.is_cpu = property(lambda self: not _is_cuda(self))
    Var.is_mps = property(lambda self: False)
    Var.is_xpu = property(lambda self: False)
    Var.is_meta = property(lambda self: getattr(self.device, "type", None) == "meta")
    # torch's Tensor.get_device(): CUDA device index, or -1 for CPU tensors.
    # 3DGS's fallback ssim (utils/loss_utils.py) does window.cuda(img.get_device()).
    if not hasattr(Var, "get_device"):
        Var.get_device = lambda self: (0 if _is_cuda(self) else -1)

    # torch's Tensor.narrow(dim, start, length): a view of `length` elements
    # starting at `start` along `dim` (jittor has no narrow; use a slice).
    if not hasattr(Var, "narrow"):
        def _narrow(self, dim, start, length):
            nd = self.ndim
            d = dim if dim >= 0 else dim + nd
            if start < 0:
                start += self.shape[d]
            sl = [slice(None)] * nd
            sl[d] = slice(start, start + length)
            return self[tuple(sl)]
        Var.narrow = _narrow

    # torch's Tensor.stride()/.as_strided(): jittor Vars are always materialized
    # contiguous (row-major) -- `.contiguous` above is a no-op -- so a Var's strides
    # are exactly the row-major strides of its shape (this matches torch's strides
    # right after a `.view()`/`.reshape()`, which is where this is used, e.g.
    # longformer's `_chunk` sliding-window attention).
    if not hasattr(Var, "stride"):
        def _stride(self, dim=None):
            shape = self.shape
            st = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                st[i] = st[i + 1] * shape[i + 1]
            if dim is None:
                return tuple(st)
            return st[dim if dim >= 0 else dim + len(shape)]
        Var.stride = _stride
    if not hasattr(Var, "storage_offset"):
        Var.storage_offset = lambda self: 0
    # as_strided over a contiguous buffer == gather at linear offsets
    #   out[i0,i1,...] = flat[storage_offset + sum_d i_d * stride[d]]
    # Built with broadcast arange grids; routed through jittor advanced-indexing so
    # the backward is the correct scatter-add (overlapping windows read shared inputs).
    if not hasattr(Var, "as_strided"):
        def _as_strided(self, size, stride, storage_offset=0):
            size = [int(s) for s in size]
            stride = [int(s) for s in stride]
            flat = self.reshape(-1)
            idx = None
            for d in range(len(size)):
                ar = jt.arange(size[d], dtype="int64") * stride[d]
                shp = [1] * len(size)
                shp[d] = size[d]
                ar = ar.reshape(shp)
                idx = ar if idx is None else idx + ar
            if storage_offset:
                idx = idx + int(storage_offset)
            return flat[idx.reshape(-1)].reshape(size)
        Var.as_strided = _as_strided

    # torch's Tensor.where(condition, other): elements of *self* where condition is
    # True, else from `other`. jittor's native Var.where treats *self* as the condition
    # (ternary(self, a, b)) -- the opposite role -- so `t.where(cond, other)` silently
    # returned `cond` cast to t's dtype (breaks e.g. longformer's _mask_invalid_locations
    # edge masking). Add the torch 2-arg method semantics while preserving jittor's
    # native 0/1-arg form (nonzero indices), used by contrib.py. No jittor-core caller
    # uses the 2-arg method form, so this only fixes, never regresses.
    if not getattr(Var.where, "_torch_where_compat", False):
        _jt_var_where = Var.where
        def _torch_where(self, *args):
            if len(args) == 2:
                condition, other = args
                return _torch_where_select(condition, self, other)
            return _jt_var_where(self, *args)
        _torch_where._torch_where_compat = True
        Var.where = _torch_where

    # torch's Tensor.tile(*dims): like numpy.tile -- when fewer dims than the
    # tensor rank are given, dims are left-padded with 1. jittor's repeat
    # already implements exactly this padding, so route tile through it.
    if not hasattr(Var, "tile"):
        def _tile(self, *dims):
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = tuple(dims[0])
            return self.repeat(*dims)
        Var.tile = _tile

    # torch's Tensor.squeeze(dim=None): differs from jittor's in two ways --
    #   * squeeze(dim) where that dim's size != 1 is a NO-OP in torch, but
    #     jittor asserts (AssertionError). Models call x.squeeze(d) defensively.
    #   * torch 2.0+ accepts a tuple/list of dims (squeeze((0,2))); jittor's
    #     native squeeze only takes a single int (raises TypeError on a tuple).
    # Wrap to match torch while delegating the actual op to jittor's squeeze.
    _native_squeeze = Var.squeeze
    def _squeeze(self, dim=None):
        if dim is None:
            out = _native_squeeze(self)
            logical_0d = all(int(s) == 1 for s in self.shape)
            if logical_0d:
                out._torch_0d = True
            return out
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        nd = self.ndim
        # normalize negatives and keep only the dims whose size is 1 (torch
        # silently ignores the rest). Remove from highest index to lowest so
        # earlier removals don't shift the indices of later ones.
        norm = sorted({(d if d >= 0 else d + nd) for d in dims}, reverse=True)
        out = self
        for d in norm:
            if 0 <= d < out.ndim and out.shape[d] == 1:
                out = _native_squeeze(out, d)
        removed = {d for d in norm if 0 <= d < nd and self.shape[d] == 1}
        if len(removed) == nd:
            out._torch_0d = True
        return out
    Var.squeeze = _squeeze

    # torch's Tensor.baddbmm(batch1, batch2, *, beta=1, alpha=1):
    #   out = beta * self + alpha * (batch1 @ batch2)   (batched matmul)
    # jittor exposes a module-level baddbmm but no Var method (bloom calls
    # the method form). Mirror torch's keyword-only beta/alpha here.
    if not hasattr(Var, "baddbmm"):
        def _baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
            res = jt.matmul(batch1, batch2)
            if alpha != 1:
                res = res * alpha
            if beta == 0:
                return res
            return beta * self + res
        Var.baddbmm = _baddbmm
    # torch's Tensor.addmm(mat1, mat2, *, beta=1, alpha=1):
    #   out = beta * self + alpha * (mat1 @ mat2)   (2-D matmul)
    if not hasattr(Var, "addmm"):
        def _addmm_method(self, mat1, mat2, *, beta=1, alpha=1):
            res = jt.matmul(mat1, mat2)
            if alpha != 1:
                res = res * alpha
            if beta == 0:
                return res
            return beta * self + res
        Var.addmm = _addmm_method

    # torch's Tensor.T: reverse ALL dims (a deprecated-but-ubiquitous alias for
    # x.permute(reversed(range(ndim)))); a no-op for ndim < 2. jittor lacks it.
    if not isinstance(getattr(Var, "T", None), property):
        def _T(self):
            nd = self.ndim
            if nd < 2:
                return self
            return self.permute(*range(nd - 1, -1, -1))
        Var.T = property(_T)
    # torch's Tensor.mT: swap the last two dims (batched matrix transpose);
    # requires ndim >= 2. Used by modern attention code (q.mT @ k etc.).
    if not isinstance(getattr(Var, "mT", None), property):
        def _mT(self):
            return self.transpose(-1, -2)
        Var.mT = property(_mT)

    # torch's Tensor.norm(p='fro', dim=None, keepdim=False, dtype=None):
    # default (dim=None) reduces over ALL dims to a 0-dim scalar -- but jittor's
    # native Var.norm defaults to dim=-1 (per-row). Override to torch semantics
    # while STAYING compatible with jittor's internal positional convention
    #   jt.norm(x, p=2, dim=-1, keepdims=False, eps=1e-30, keepdim=False)
    # which callers like misc.normalize use as input.norm(p, dim, True, eps).
    # The collision is the 4th positional: torch=dtype, jittor=eps. Disambiguate
    # by type (a number -> jittor eps; a dtype/str/None -> torch dtype). When dim
    # is given explicitly (the only way internal callers reach here) behavior is
    # identical to before; only the dim=None default changes to a full reduce.
    _norm_via = _torch_norm_impl
    _native_norm = Var.norm  # jittor's native Var.norm (eps-floored, dim=-1)
    def _var_norm(self, p="fro", dim=None, keepdims=None, *rest,
                  keepdim=False, dtype=None, eps=None, **kw):
        # jittor's internal convention is norm(p, dim, keepdims, eps): when a
        # 4th positional eps (a non-bool number) or an explicit eps= is present,
        # this is an internal call -- delegate verbatim to the native op so its
        # eps-floor (used by misc.normalize/weightnorm to avoid div-by-zero) is
        # preserved exactly.
        fourth = rest[0] if rest else None
        is_internal = eps is not None or (
            isinstance(fourth, (int, float)) and not isinstance(fourth, bool))
        if is_internal:
            kdv = bool(keepdims) if keepdims is not None else keepdim
            ev = eps if eps is not None else (fourth if fourth is not None else 1e-30)
            d = -1 if dim is None else dim
            return _native_norm(self, p if p != "fro" else 2, d, kdv, ev)
        # torch convention: norm(p='fro', dim=None, keepdim=False, dtype=None)
        kd = bool(keepdims) if keepdims is not None else keepdim
        if fourth is not None:
            dtype = fourth
        return _norm_via(self, p=p, dim=dim, keepdim=kd, dtype=dtype)
    Var.norm = _var_norm


def _install_misc(g, Var, _DTYPE_OBJS=None):
    if _DTYPE_OBJS is None:
        _DTYPE_OBJS = dtype._registry
    import sys as _sys_misc
    import types as _types_misc
    _types2 = _types_misc

    if "torch.storage" not in _sys_misc.modules:
        _storage_mod = _types_misc.ModuleType("torch.storage")

        class UntypedStorage:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def _typed_storage(self):
                return TypedStorage(wrap_storage=self)

        class TypedStorage:
            def __init__(self, *args, wrap_storage=None, dtype=None, device=None,
                         _internal=False, **kwargs):
                self._untyped_storage = wrap_storage
                self.dtype = dtype if dtype is not None else getattr(g, "float32", "float32")
                self.device = device
                self.args = args
                self.kwargs = kwargs

            def untyped(self):
                return self._untyped_storage

        _storage_mod.UntypedStorage = UntypedStorage
        _storage_mod.TypedStorage = TypedStorage
        _sys_misc.modules["torch.storage"] = _storage_mod
    else:
        _storage_mod = _sys_misc.modules["torch.storage"]

    g.storage = _storage_mod
    g.UntypedStorage = getattr(_storage_mod, "UntypedStorage")
    g.TypedStorage = getattr(_storage_mod, "TypedStorage")
    for _name in (
        "DoubleStorage", "FloatStorage", "HalfStorage", "BFloat16Storage",
        "LongStorage", "IntStorage", "ShortStorage", "CharStorage",
        "ByteStorage", "BoolStorage",
    ):
        if not hasattr(g, _name):
            setattr(g, _name, type(_name, (g.TypedStorage,), {"__module__": "torch"}))
        if not hasattr(_storage_mod, _name):
            setattr(_storage_mod, _name, getattr(g, _name))

    import types as _types_random
    _native_random_fn = getattr(g, "random", None)
    class _RandomModule(_types_random.ModuleType):
        def __call__(self, *args, **kwargs):
            if callable(_native_random_fn):
                return _native_random_fn(*args, **kwargs)
            raise TypeError("torch.random is not callable")
    _random_mod = _RandomModule("torch.random")
    _random_mod._seed = int(getattr(g, "_torch_initial_seed", 0))
    def _manual_seed(s):
        s = int(s)
        _random_mod._seed = s
        g._torch_initial_seed = s
        if hasattr(jt, "set_global_seed"):
            jt.set_global_seed(s)
        return g
    def _get_rng_state():
        return jt.array([int(getattr(_random_mod, "_seed", 0))], dtype="int64")
    def _set_rng_state(state):
        try:
            if isinstance(state, Var):
                state = int(state.reshape(-1)[0].item())
            elif hasattr(state, "__len__"):
                state = int(list(state)[0])
            else:
                state = int(state)
        except Exception:
            state = int(getattr(_random_mod, "_seed", 0))
        _manual_seed(state)
    g.manual_seed = _manual_seed
    g.initial_seed = lambda: int(getattr(_random_mod, "_seed", 0))
    g.seed = lambda: int(getattr(_random_mod, "_seed", 0))
    g.get_rng_state = _get_rng_state
    g.set_rng_state = _set_rng_state
    _random_mod.manual_seed = _manual_seed
    _random_mod.initial_seed = g.initial_seed
    _random_mod.seed = g.seed
    _random_mod.get_rng_state = _get_rng_state
    _random_mod.set_rng_state = _set_rng_state
    g.random = _random_mod
    _sys_misc.modules["torch.random"] = _random_mod
    g.is_tensor = lambda x: isinstance(x, Var)
    if not hasattr(g, "numel"):
        g.numel = lambda x: x.numel()
    if not hasattr(g, "PyTorchFileReader"):
        class PyTorchFileReader:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    "torch.PyTorchFileReader is not implemented by the jittor torch shim; use torch.load instead")
        g.PyTorchFileReader = PyTorchFileReader

    # torch.norm(input, p='fro', dim=None, keepdim=False, dtype=None, out=None):
    # default reduces over ALL dims to a 0-dim scalar. jittor's jt.norm defaults
    # to dim=-1 (per-row), so torch.norm(x)/x.norm() silently returned a vector.
    # Override the torch-facing top-level norm (NOT jt.norm's internal default,
    # which jittor relies on) to match torch.
    def norm(input, p="fro", dim=None, keepdim=False, dtype=None, out=None, **kw):
        return _torch_norm_impl(input, p=p, dim=dim, keepdim=keepdim, dtype=dtype)
    g.norm = norm

    # autocast / grad-mode query helpers
    g.is_autocast_enabled = lambda *a, **k: False
    g.set_autocast_enabled = lambda *a, **k: None
    g.is_grad_enabled = lambda: not bool(getattr(jt.flags, "no_grad", 0))
    g.set_grad_enabled = lambda mode: (g.enable_grad() if mode else g.no_grad())
    g.get_autocast_dtype = lambda *a, **k: getattr(g, "float32", "float32")
    g.get_autocast_gpu_dtype = lambda *a, **k: getattr(g, "float16", "float16")
    g.is_autocast_available = lambda *a, **k: False
    g.are_deterministic_algorithms_enabled = lambda: False
    g.use_deterministic_algorithms = lambda *a, **k: None
    g.is_floating_point = lambda x: ("float" in str(x.dtype))

    def where(condition, input=None, other=None, *, out=None):
        if input is None and other is None:
            native_where = getattr(jt, "_vj_native_where", None)
            idx = native_where(condition) if native_where is not None else _native_nonzero(condition)
            if isinstance(idx, tuple):
                return idx
            if getattr(idx, "ndim", 0) == 2:
                return tuple(idx[:, d] for d in range(idx.shape[1]))
            return (idx.reshape(-1),)
        if input is None or other is None:
            raise TypeError("torch.where expected either 1 or 3 arguments")
        return _torch_where_select(condition, input, other)
    g.where = where

    # torch-compat: torch.bincount(input, weights=None, minlength=0). Counts the
    # occurrences of each non-negative integer in a 1-D `input`; with `weights`,
    # sums the weights per bin instead. Output length = max(input.max()+1, minlength)
    # (0 for an empty input, honoring minlength). Implemented with jittor's native
    # out-of-place scatter_add (reduce='add' accumulates duplicate indices).
    if not hasattr(g, "bincount"):
        def bincount(input, weights=None, minlength=0):
            x = input.reshape(-1).int64()
            ml = max(int(minlength), 0)
            if x.numel() == 0:
                wdtype = weights.dtype if weights is not None else jt.int64
                return jt.zeros((ml,), dtype=wdtype)
            n = max(int(x.max().item()) + 1, ml)
            if weights is not None:
                out = jt.zeros((n,), dtype=weights.dtype)
                src = weights.reshape(-1).cast(str(weights.dtype))
            else:
                out = jt.zeros((n,), dtype=jt.int64)
                src = jt.ones((x.shape[0],), dtype=jt.int64)
            # scatter_add is out-of-place and accumulates at duplicate indices.
            return out.scatter_add(0, x, src)
        g.bincount = bincount
        Var.bincount = lambda self, weights=None, minlength=0: bincount(self, weights, minlength)

    # torch-compat: torch.segment_reduce(data, reduce, *, lengths) -- reduce over
    # contiguous variable-length segments along dim 0 (lengths-based form). torch
    # supports reduce in {sum, mean, prod, max/amax, min/amin}; the per-segment
    # result is stacked back into a (num_segments, *data.shape[1:]) tensor.
    if not hasattr(g, "segment_reduce"):
        def segment_reduce(data, reduce="sum", *, lengths=None, **kw):
            assert lengths is not None, "torch_compat segment_reduce requires lengths="
            lengths_list = [int(l) for l in lengths]
            tail = list(data.shape[1:])     # per-element shape; segments reduce dim 0
            segs = []
            start = 0
            for l in lengths_list:
                chunk = data[start:start + l]
                start += l
                if reduce == "sum":
                    r = chunk.sum(dim=0)
                elif reduce == "mean":
                    r = chunk.mean(dim=0)
                elif reduce == "prod":
                    r = chunk.prod(dim=0)
                elif reduce in ("max", "amax"):
                    r = chunk.amax(dim=0)    # values-only (Var.max here is the (values,idx) shim)
                elif reduce in ("min", "amin"):
                    r = chunk.amin(dim=0)
                else:
                    raise ValueError(f"Unsupported segment_reduce op: {reduce}")
                # jittor's reduce over dim 0 leaves a leading size-1 axis; normalise
                # each segment to (1, *tail) so concat gives torch's output shape:
                # 1-D data -> (num_segments,), N-D data -> (num_segments, *data[1:]).
                segs.append(r.reshape([1] + tail))
            return jt.concat(segs, dim=0)
        g.segment_reduce = segment_reduce

    # torch unary math jittor lacks at top level (it has log2 but not exp2/log10/trunc/sign)
    import math as _math_la
    _LN2 = _math_la.log(2.0); _INV_LN10 = 1.0 / _math_la.log(10.0)
    def _sign(x): return (x > 0) * 1.0 - (x < 0) * 1.0   # jittor has no jt.sign
    g.exp2 = lambda x: jt.exp(x * _LN2)
    g.log10 = lambda x: jt.log(x) * _INV_LN10
    g.sign = _sign
    g.trunc = lambda x: _sign(x) * jt.floor(jt.abs(x))
    Var.exp2 = lambda self: jt.exp(self * _LN2)
    Var.log10 = lambda self: jt.log(self) * _INV_LN10
    if not hasattr(Var, "sign"): Var.sign = lambda self: _sign(self)
    Var.trunc = lambda self: _sign(self) * jt.floor(jt.abs(self))


    # ---- finfo / iinfo ----
    import numpy as _np
    # hardcoded specs for dtypes numpy can't represent: (min, max, eps, tiny, bits)
    _FINFO_SPECIAL = {
        "bfloat16": (-3.3895313892515355e38, 3.3895313892515355e38, 0.0078125, 1.1754943508222875e-38, 16),
        "float8_e4m3fn": (-448.0, 448.0, 0.125, 0.015625, 8),
        "float8_e4m3fnuz": (-240.0, 240.0, 0.125, 0.0078125, 8),
        "float8_e5m2": (-57344.0, 57344.0, 0.25, 6.103515625e-05, 8),
        "float8_e5m2fnuz": (-57344.0, 57344.0, 0.25, 6.103515625e-05, 8),
        "float8_e8m0fnu": (-3.4e38, 3.4e38, 1.0, 1e-38, 8),
        "float4_e2m1fn_x2": (-6.0, 6.0, 0.5, 0.5, 4),
    }
    class finfo:
        def __init__(self, dt):
            ds = _dtype_to_str(dt) or "float32"
            if ds in _FINFO_SPECIAL:
                mn, mx, eps, tiny, bits = _FINFO_SPECIAL[ds]
                self.min, self.max, self.eps, self.tiny, self.smallest_normal = mn, mx, eps, tiny, tiny
                self.bits, self.dtype = bits, ds
                self.resolution = eps
                return
            info = _np.finfo(_np.dtype(ds))
            self.min = float(info.min); self.max = float(info.max)
            self.eps = float(info.eps); self.tiny = float(info.tiny)
            self.smallest_normal = float(info.tiny)
            self.resolution = float(info.resolution)
            self.bits = info.bits; self.dtype = ds
    class iinfo:
        def __init__(self, dt):
            ds = _dtype_to_str(dt) or "int64"
            info = _np.iinfo(_np.dtype(ds))
            self.min = int(info.min); self.max = int(info.max); self.bits = info.bits
    g.finfo = finfo
    g.iinfo = iinfo

    # ---- type promotion (torch.result_type / promote_types / can_cast) ----
    # Encodes torch's documented `_promoteTypesLookup` lattice (c10/core/
    # ScalarType.cpp). Rules, verified against torch's docs:
    #   * category order  bool < (signed/unsigned int) < float < complex;
    #   * same-category ints: the wider wins, BUT mixing signed+unsigned of the
    #     same OR smaller width promotes to a SIGNED type wide enough to hold both
    #     (uint8+int8 -> int16, uint8+int16 -> int16, uint8+int32 -> int32,
    #     uint8+int64 -> int64);  uint8+uint8 stays uint8;
    #   * a float of ANY width absorbs an int of ANY width WITHOUT widening
    #     (float16+int64 -> float16);  int+bfloat16 -> bfloat16 (torch parity,
    #     incl. its known low-mantissa caveat);
    #   * floats: the wider wins, except float16+bfloat16 -> float32 (neither can
    #     represent the other; matches torch/JAX).
    # This is the SAME table torch's binary ops consult, so wrapping the Var
    # arithmetic operators (below, in _install_tensor_methods) to cast both
    # operands to result_type before the native op reproduces torch exactly.
    _PROMO_ORDER = ["bool", "uint8", "int8", "int16", "int32", "int64",
                    "float16", "bfloat16", "float32", "float64"]
    # The lower-triangular promotion matrix (symmetric); rows/cols in _PROMO_ORDER.
    # b1 u1 i1 i2 i4 i8 f2 bf f4 f8
    _PROMO_ROWS = {
        "bool":     ["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        "uint8":    ["uint8", "uint8", "int16", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        "int8":     ["int8", "int16", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        "int16":    ["int16", "int16", "int16", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        "int32":    ["int32", "int32", "int32", "int32", "int32", "int64", "float16", "bfloat16", "float32", "float64"],
        "int64":    ["int64", "int64", "int64", "int64", "int64", "int64", "float16", "bfloat16", "float32", "float64"],
        "float16":  ["float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float32", "float64"],
        "bfloat16": ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16", "float32", "bfloat16", "float32", "float64"],
        "float32":  ["float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float64"],
        "float64":  ["float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64"],
    }
    _PROMO_IDX = {n: i for i, n in enumerate(_PROMO_ORDER)}

    def _promote_pair(a, b):
        # a, b are bare dtype-name strings. Unknown/complex types fall back to the
        # wider of the two by category index when possible, else to a.
        if a == b:
            return a
        ia, ib = _PROMO_IDX.get(a), _PROMO_IDX.get(b)
        if ia is not None and ib is not None:
            return _PROMO_ROWS[a][ib]
        # complex (jittor has no native complex compute, but keep the lattice sane)
        if a.startswith("complex") or b.startswith("complex"):
            wide = "complex128" if ("128" in a or "128" in b or "float64" in (a, b)) else "complex64"
            return wide
        return a if ib is None else b

    def promote_types(t1, t2):
        return _DTYPE_OBJS.get(_promote_pair(_dtype_to_str(t1), _dtype_to_str(t2)),
                               _promote_pair(_dtype_to_str(t1), _dtype_to_str(t2)))
    g.promote_types = promote_types

    def _category(name):
        # 0 bool, 1 int, 2 float, 3 complex -- torch's scalar-promotion categories.
        if name == "bool":
            return 0
        if name.startswith(("int", "uint")):
            return 1
        if name.startswith("complex"):
            return 3
        return 2

    def result_type(a, b):
        # torch.result_type(a, b): a/b may each be a Tensor, a dtype, or a Python
        # number. Two tensors (or dtypes) -> promote_types. A Python scalar follows
        # torch's "wrapped number" rule: it only bumps the result if it is a HIGHER
        # category than the tensor; a same-or-lower-category scalar keeps the
        # tensor's dtype (an int scalar does NOT widen, a float scalar lifts an int
        # tensor to the default float).
        def info(x):
            if isinstance(x, Var):
                return (_dtype_to_str(x.dtype), False)
            if isinstance(x, dtype) or (isinstance(x, str) and _dtype_to_str(x) in _DTYPE_OBJS):
                return (_dtype_to_str(x), False)
            if isinstance(x, bool):
                return ("bool", True)
            if isinstance(x, int):
                return ("int64", True)
            if isinstance(x, float):
                return (_dtype_to_str(g.get_default_dtype()) or "float32", True)
            if isinstance(x, complex):
                return ("complex64", True)
            return (_dtype_to_str(x) or "float32", False)
        (na, sa), (nb, sb) = info(a), info(b)
        if sa and not sb:
            # scalar a vs tensor b: bump only if a is a strictly higher category
            res = _promote_pair(na, nb) if _category(na) > _category(nb) else nb
        elif sb and not sa:
            res = _promote_pair(na, nb) if _category(nb) > _category(na) else na
        else:
            res = _promote_pair(na, nb)
        return _DTYPE_OBJS.get(res, res)
    g.result_type = result_type

    def can_cast(from_dtype, to_dtype):
        # torch.can_cast(from, to): True iff `from` can promote into `to` without
        # leaving its (or a lower) category -- i.e. promote(from, to) == to.
        f, t = _dtype_to_str(from_dtype), _dtype_to_str(to_dtype)
        return _promote_pair(f, t) == t
    g.can_cast = can_cast

    # Expose the promoter for the operator wrappers installed on Var.
    g._torch_promote_pair = _promote_pair

    # ---- default dtype/device ----
    _state = {"dtype": getattr(g, "float32", "float32")}
    g.get_default_dtype = lambda: _state["dtype"]
    def set_default_dtype(d):
        _state["dtype"] = d
    g.set_default_dtype = set_default_dtype
    def get_default_device():
        return g.device("cuda", 0) if (jt.flags.use_cuda or getattr(jt.compiler, "has_acl", 0)) else g.device("cpu")
    g.get_default_device = get_default_device
    g.set_default_device = lambda *a, **k: None

    # ---- save / load ----
    # torch.save must handle BOTH tensor state-dicts AND arbitrary Python
    # objects (e.g. TrainingArguments). jittor's jt.save is numpy/pickle based
    # but chokes on live Vars and some objects, so we use standard pickle with
    # Vars converted to a portable (numpy) form and restored on load.
    import os as _os_pickle, pickle as _pickle
    _VAR_TAG = "__jt_var__"
    def _to_portable(obj, _seen=None):
        if isinstance(obj, jt.Var):
            # Var.numpy() can materialize a CUDA Var on CPU in-place. Checkpoint
            # serialization must not change the live module/optimizer tensors.
            return {_VAR_TAG: True, "data": obj.clone().numpy(), "dtype": str(obj.dtype)}
        # Drop non-picklable callables (e.g. an LR scheduler's local lr_lambda
        # closure in an extra/scheduler state_dict). torch's LambdaLR.state_dict
        # does the same -- the lambda is rebuilt on load, not restored.
        import types as _t
        if isinstance(obj, (_t.FunctionType, _t.LambdaType, _t.MethodType, _t.BuiltinFunctionType)):
            return None
        if isinstance(obj, dict):
            return {k: _to_portable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            items = [_to_portable(v) for v in obj]
            # Coerce list/tuple SUBCLASSES (e.g. the shim's local _ParamList in
            # an optimizer state_dict) to plain list/tuple -- local classes are
            # not picklable. Preserve namedtuples.
            if isinstance(obj, tuple):
                if hasattr(obj, "_fields"):
                    try:
                        return type(obj)(*items)
                    except Exception:
                        return tuple(items)
                return tuple(items)
            return list(items)
        return obj
    def _from_portable(obj):
        if isinstance(obj, dict):
            if obj.get(_VAR_TAG):
                # from_numpy preserves wide dtypes (float64/int64); jt.array narrows
                # them to float32/int32 -> torch.save/load silently downcast checkpoints.
                return g.from_numpy(obj["data"])
            return {k: _from_portable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_from_portable(v) for v in obj)
        return obj
    def save(obj, f, *a, **k):
        try:
            jt.sync_all(True)
        except Exception:
            pass
        portable = _to_portable(obj)
        if hasattr(f, "write"):
            _pickle.dump(portable, f)
            return
        with open(f, "wb") as fh:
            _pickle.dump(portable, fh)

    # ---- load a REAL torch .pt checkpoint (zip archive w/ persistent-id storages) ----
    # torch.save writes a zip: <name>/data.pkl (object graph; tensors are
    # persistent_id refs to storages) + <name>/data/<key> (raw storage bytes).
    # Reconstruct tensors as jittor Vars without needing real torch.
    import io as _io, zipfile as _zipfile
    import numpy as _np_pt
    _TORCH_STORAGE_DTYPE = {
        "DoubleStorage": "float64", "FloatStorage": "float32", "HalfStorage": "float16",
        "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
        "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8",
        "BoolStorage": "bool",
    }
    class _StorageMarker:
        def __init__(self, dtype_str): self.dtype_str = dtype_str
    def _np_from_storage(raw, dtype_str, numel):
        if dtype_str == "bfloat16":
            u16 = _np_pt.frombuffer(raw, dtype=_np_pt.uint16, count=numel).astype(_np_pt.uint32)
            return (u16 << 16).view(_np_pt.float32)   # widen bf16 -> f32 (ACL has no bf16 numpy)
        npd = {"float64": _np_pt.float64, "float32": _np_pt.float32, "float16": _np_pt.float16,
               "int64": _np_pt.int64, "int32": _np_pt.int32, "int16": _np_pt.int16,
               "int8": _np_pt.int8, "uint8": _np_pt.uint8, "bool": _np_pt.bool_}[dtype_str]
        return _np_pt.frombuffer(raw, dtype=npd, count=numel)
    def _load_torch_pt(path_or_file):
        zf = _zipfile.ZipFile(path_or_file, "r")
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        data_dir = pkl_name[:-len("data.pkl")] + "data/"
        cache = {}
        def _persistent_load(pid):
            assert pid[0] == "storage", pid
            marker, key, numel = pid[1], str(pid[2]), int(pid[4])
            if key not in cache:
                cache[key] = (zf.read(data_dir + key), marker.dtype_str, numel)
            return cache[key]
        def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                               requires_grad=False, backward_hooks=None, metadata=None):
            raw, dtype_str, numel = storage
            arr = _np_from_storage(raw, dtype_str, numel)
            size = tuple(int(s) for s in size)
            n = 1
            for s in size: n *= s
            sub = arr[storage_offset:storage_offset + n]
            sub = _np_pt.ascontiguousarray(sub).reshape(size) if size else sub.reshape(())
            return jt.array(sub)
        def _rebuild_parameter(data, requires_grad=True, backward_hooks=None, *a, **k):
            return data
        class _Unpick(_pickle.Unpickler):
            def persistent_load(self, pid):
                return _persistent_load(pid)
            def find_class(self, module, name):
                if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return _rebuild_tensor_v2
                if module == "torch._utils" and name.startswith("_rebuild_parameter"):
                    return _rebuild_parameter
                if name.endswith("Storage") and module.startswith("torch"):
                    return _StorageMarker(_TORCH_STORAGE_DTYPE.get(name, "float32"))
                if module == "collections" and name == "OrderedDict":
                    from collections import OrderedDict
                    return OrderedDict
                if module == "torch" and name == "Size":
                    return tuple
                if module == "torch" and name == "device":
                    return lambda *a, **k: "cpu"
                try:
                    m = __import__(module, fromlist=[name]); return getattr(m, name)
                except Exception:
                    return type(name, (), {})
        return _Unpick(_io.BytesIO(zf.read(pkl_name))).load()

    def _is_zip(f):
        if hasattr(f, "read"):
            pos = f.tell(); head = f.read(2); f.seek(pos)
            return head[:2] == b"PK"
        with open(f, "rb") as fh:
            return fh.read(2)[:2] == b"PK"
    def _is_legacy_torch_pickle(path):
        # PyTorch's pre-zip serialization starts with three pickles:
        # MAGIC_NUMBER, PROTOCOL_VERSION and sys_info, then uses persistent
        # storage records. Plain torch-shim checkpoints are regular pickles and
        # must continue down the portable-pickle path.
        try:
            with open(path, "rb") as fh:
                magic = _pickle.load(fh, encoding="utf-8")
        except Exception:
            return False
        return magic == 0x1950a86a20f9469cfc6c
    def load(f, *a, **k):
        # accept map_location/weights_only/pickle_module kwargs (ignored).
        # Real torch .pt is a zip archive -> use the torch-format loader;
        # our own torch.save output is plain pickle -> _from_portable.
        path = None
        if not hasattr(f, "read"):
            path = _os_pickle.fspath(f)
            native_load = getattr(g, "_vj_native_load", None)
            if native_load is not None and path.startswith(("jittorhub://", "http://", "https://")):
                return native_load(path)
        try:
            if _is_zip(f):
                return _load_torch_pt(f)
        except Exception as _e:
            pass
        if path is not None and path.lower().endswith((".pth", ".pt", ".bin")) and _is_legacy_torch_pickle(path):
            from jittor_utils.load_pytorch import load_pytorch as _load_pytorch
            return _load_pytorch(path)
        try:
            if hasattr(f, "read"):
                obj = _pickle.load(f)
            else:
                with open(f, "rb") as fh:
                    obj = _pickle.load(fh)
        except Exception:
            native_load = getattr(g, "_vj_native_load", None)
            if native_load is not None and path is not None and path.lower().endswith(".pkl"):
                return native_load(path)
            raise
        return _from_portable(obj)
    g.save = save
    g.load = load
    # Stash the real pickle loader so adapters can restore it if torch.load gets
    # shadowed later (the torch_shim exposes cpp_extension.load(name, sources,...)
    # at the torch top level, which can mask this in some worker processes).
    g._vj_pickle_load = load
    g._vj_pickle_save = save

    # ---- torch.distributions package layout ----
    # jittor.distributions already implements the common distribution classes;
    # expose the PyTorch package/submodule names that transformers imports.
    import types as _types_dist, sys as _sys_dist
    try:
        import jittor.distributions as _dist
        _dist.__path__ = getattr(_dist, "__path__", [])
        if not hasattr(_dist, "constraints"):
            _constraints = _types_dist.ModuleType("torch.distributions.constraints")
            class _Constraint:
                def __init__(self, *a, **k): pass
                def check(self, x):
                    try:
                        return jt.ones_like(x).bool()
                    except Exception:
                        return True
            for _cn in ("positive", "real", "nonnegative", "nonnegative_integer",
                        "positive_integer", "unit_interval", "simplex",
                        "lower_cholesky", "positive_definite", "boolean",
                        "real_vector", "dependent", "independent"):
                setattr(_constraints, _cn, _Constraint())
            _constraints.Constraint = _Constraint
            _dist.constraints = _constraints
        _sys_dist.modules["torch.distributions"] = _dist
        _sys_dist.modules["torch.distributions.constraints"] = _dist.constraints
        g.distributions = _dist
        _dist_utils = _types_dist.ModuleType("torch.distributions.utils")
        _dist_utils.broadcast_all = getattr(_dist, "broadcast_all")
        _sys_dist.modules["torch.distributions.utils"] = _dist_utils
        _dist.utils = _dist_utils
        for _cls_name, _mod_suffix in (
            ("Distribution", "distribution"),
            ("Bernoulli", "bernoulli"),
            ("Categorical", "categorical"),
            ("OneHotCategorical", "one_hot_categorical"),
            ("Normal", "normal"),
            ("Uniform", "uniform"),
            ("RelaxedBernoulli", "relaxed_bernoulli"),
            ("LogitRelaxedBernoulli", "relaxed_bernoulli"),
            ("RelaxedOneHotCategorical", "relaxed_categorical"),
            ("Beta", "beta"),
            ("Gamma", "gamma"),
            ("Poisson", "poisson"),
            ("Dirichlet", "dirichlet"),
            ("LogNormal", "log_normal"),
            ("LogisticNormal", "logistic_normal"),
            ("MultivariateNormal", "multivariate_normal"),
        ):
            if hasattr(_dist, _cls_name):
                _sub = _types_dist.ModuleType("torch.distributions." + _mod_suffix)
                setattr(_sub, _cls_name, getattr(_dist, _cls_name))
                _sys_dist.modules["torch.distributions." + _mod_suffix] = _sub
                setattr(_dist, _mod_suffix, _sub)
        if hasattr(_dist, "RelaxedBernoulli") or hasattr(_dist, "LogitRelaxedBernoulli"):
            _relaxed_bernoulli = _types_dist.ModuleType("torch.distributions.relaxed_bernoulli")
            if hasattr(_dist, "RelaxedBernoulli"):
                _relaxed_bernoulli.RelaxedBernoulli = _dist.RelaxedBernoulli
            if hasattr(_dist, "LogitRelaxedBernoulli"):
                _relaxed_bernoulli.LogitRelaxedBernoulli = _dist.LogitRelaxedBernoulli
            _sys_dist.modules["torch.distributions.relaxed_bernoulli"] = _relaxed_bernoulli
            _dist.relaxed_bernoulli = _relaxed_bernoulli
        if hasattr(_dist, "RelaxedOneHotCategorical"):
            _relaxed_categorical = _types_dist.ModuleType("torch.distributions.relaxed_categorical")
            _relaxed_categorical.RelaxedOneHotCategorical = _dist.RelaxedOneHotCategorical
            _sys_dist.modules["torch.distributions.relaxed_categorical"] = _relaxed_categorical
            _dist.relaxed_categorical = _relaxed_categorical
        if hasattr(_dist, "kl_divergence"):
            _kl = _types_dist.ModuleType("torch.distributions.kl")
            _kl.kl_divergence = _dist.kl_divergence
            _kl.register_kl = getattr(_dist, "register_kl", lambda *a, **k: (lambda f: f))
            _sys_dist.modules["torch.distributions.kl"] = _kl
            _dist.kl = _kl

        class Gumbel:
            def __init__(self, loc, scale, validate_args=None):
                self.loc = loc
                self.scale = scale
                self.batch_shape = self._batch_shape(loc, scale)
            @staticmethod
            def _batch_shape(*params):
                shapes = []
                for p in params:
                    if hasattr(p, "shape"):
                        shape = tuple(p.shape)
                        n = 1
                        for s in shape:
                            n *= int(s)
                        shapes.append(() if n == 1 else shape)
                    else:
                        shapes.append(())
                out = ()
                for shape in shapes:
                    res = []
                    for i in range(1, max(len(out), len(shape)) + 1):
                        a = out[-i] if i <= len(out) else 1
                        b = shape[-i] if i <= len(shape) else 1
                        res.append(b if a == 1 else a if b == 1 or a == b else max(a, b))
                    out = tuple(reversed(res))
                return out
            @staticmethod
            def _sample_shape(sample_shape, batch_shape=()):
                if sample_shape is None:
                    sample_shape = ()
                elif isinstance(sample_shape, int):
                    sample_shape = (sample_shape,)
                else:
                    sample_shape = tuple(int(s) for s in sample_shape)
                out = sample_shape + tuple(batch_shape)
                return out if out else (1,)
            def rsample(self, sample_shape=None):
                u = jt.random(self._sample_shape(sample_shape, self.batch_shape))
                eps = 1e-6
                u = jt.clamp(u, eps, 1.0 - eps)
                loc = self.loc if isinstance(self.loc, jt.Var) else jt.array(self.loc)
                scale = self.scale if isinstance(self.scale, jt.Var) else jt.array(self.scale)
                return loc - scale * jt.log(-jt.log(u))
            def sample(self, sample_shape=None):
                return self.rsample(sample_shape).stop_grad()

        class RelaxedBernoulli:
            def __init__(self, temperature, probs=None, logits=None, validate_args=None):
                if probs is None and logits is None:
                    raise ValueError("Either probs or logits must be specified")
                self.temperature = temperature
                if logits is None:
                    probs_v = probs if isinstance(probs, jt.Var) else jt.array(probs)
                    self.probs = probs_v
                    self.logits = jt.log(probs_v) - jt.log(1.0 - probs_v)
                else:
                    self.logits = logits if isinstance(logits, jt.Var) else jt.array(logits)
                    self.probs = jt.sigmoid(self.logits)
            def rsample(self, sample_shape=None):
                shape = tuple(self.logits.shape)
                if sample_shape is None:
                    sample_shape = ()
                elif isinstance(sample_shape, int):
                    sample_shape = (sample_shape,)
                else:
                    sample_shape = tuple(int(s) for s in sample_shape)
                u = jt.random(sample_shape + shape)
                eps = 1e-6
                u = jt.clamp(u, eps, 1.0 - eps)
                temp = self.temperature if isinstance(self.temperature, jt.Var) else jt.array(self.temperature)
                return jt.sigmoid((self.logits + jt.log(u) - jt.log(1.0 - u)) / temp)
            def sample(self, sample_shape=None):
                return self.rsample(sample_shape).stop_grad()

        class RelaxedOneHotCategorical:
            def __init__(self, temperature, probs=None, logits=None, validate_args=None):
                if probs is None and logits is None:
                    raise ValueError("Either probs or logits must be specified")
                self.temperature = temperature
                if logits is None:
                    probs_v = probs if isinstance(probs, jt.Var) else jt.array(probs)
                    self.probs = probs_v / probs_v.sum(-1, keepdims=True)
                    self.logits = jt.log(self.probs)
                else:
                    self.logits = logits if isinstance(logits, jt.Var) else jt.array(logits)
                    self.probs = nn.softmax(self.logits, dim=-1)
            def rsample(self, sample_shape=None):
                shape = tuple(self.logits.shape)
                if sample_shape is None:
                    sample_shape = ()
                elif isinstance(sample_shape, int):
                    sample_shape = (sample_shape,)
                else:
                    sample_shape = tuple(int(s) for s in sample_shape)
                u = jt.random(sample_shape + shape)
                eps = 1e-6
                u = jt.clamp(u, eps, 1.0 - eps)
                gumbels = -jt.log(-jt.log(u))
                temp = self.temperature if isinstance(self.temperature, jt.Var) else jt.array(self.temperature)
                return nn.softmax((self.logits + gumbels) / temp, dim=-1)
            def sample(self, sample_shape=None):
                return self.rsample(sample_shape).stop_grad()

        _dist.Gumbel = getattr(_dist, "Gumbel", Gumbel)
        _dist.RelaxedBernoulli = getattr(_dist, "RelaxedBernoulli", RelaxedBernoulli)
        _dist.RelaxedOneHotCategorical = getattr(_dist, "RelaxedOneHotCategorical", RelaxedOneHotCategorical)
        for _cls_name, _mod_suffix in (
            ("Gumbel", "gumbel"),
            ("RelaxedBernoulli", "relaxed_bernoulli"),
            ("RelaxedOneHotCategorical", "relaxed_categorical"),
        ):
            _sub = _types_dist.ModuleType("torch.distributions." + _mod_suffix)
            setattr(_sub, _cls_name, getattr(_dist, _cls_name))
            _sys_dist.modules[_sub.__name__] = _sub
            setattr(_dist, _mod_suffix, _sub)
    except Exception:
        pass

    # ---- torch._utils ----
    import types as _types2
    _tutils = _types2.ModuleType("torch._utils")
    def _flatten_dense_tensors(tensors):
        tensors = list(tensors)
        if len(tensors) == 1:
            return tensors[0].reshape(-1).clone()
        return jt.concat([t.reshape(-1) for t in tensors]) if tensors else jt.array([])
    def _unflatten_dense_tensors(flat, tensors):
        outputs, offset = [], 0
        for t in tensors:
            n = 1
            for s in t.shape:
                n *= int(s)
            outputs.append(flat[offset:offset + n].reshape(t.shape))
            offset += n
        return outputs
    def _take_tensors(tensors, size_limit):
        buckets = {}
        for t in tensors:
            key = str(getattr(t, "dtype", "object"))
            b = buckets.setdefault(key, [[], 0])
            n = int(t.numel()) if hasattr(t, "numel") else 1
            b[0].append(t)
            b[1] += n * 4
            if b[1] >= size_limit:
                yield b[0]
                buckets[key] = [[], 0]
        for b in buckets.values():
            if b[0]:
                yield b[0]
    def _get_available_device_type():
        if hasattr(g, "cuda") and g.cuda.is_available():
            return "cuda"
        if hasattr(g, "npu") and g.npu.is_available():
            return "npu"
        if hasattr(g, "mps") and g.mps.is_available():
            return "mps"
        return None
    def _get_device_module(device_type):
        if device_type is None:
            return None
        return getattr(g, str(device_type), None)
    _tutils._flatten_dense_tensors = _flatten_dense_tensors
    _tutils._unflatten_dense_tensors = _unflatten_dense_tensors
    _tutils._take_tensors = _take_tensors
    _tutils._get_available_device_type = _get_available_device_type
    _tutils._get_device_module = _get_device_module
    _tutils._rebuild_tensor = lambda data, *a, **k: data
    _tutils._rebuild_tensor_v2 = lambda data, *a, **k: data
    _tutils._rebuild_parameter = lambda data, requires_grad=True, *a, **k: _torch_make_parameter(data, requires_grad)
    _tutils._rebuild_parameter_with_state = lambda data, requires_grad=True, backward_hooks=None, state=None: _torch_make_parameter(data, requires_grad)
    _sys_dist.modules["torch._utils"] = _tutils
    g._utils = _tutils

    # ---- torch.hub ----
    import types as _types_hub, os as _os_hub, urllib.request as _urlreq_hub
    from urllib.parse import urlparse as _urlparse_hub
    hub = _types_hub.ModuleType("torch.hub")
    def _hub_dir():
        return _os_hub.path.expanduser(_os_hub.environ.get("TORCH_HOME", "~/.cache/torch"))
    def _hub_checkpoints_dir():
        path = _os_hub.path.join(_hub_dir(), "hub", "checkpoints")
        _os_hub.makedirs(path, exist_ok=True)
        return path
    def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
        _os_hub.makedirs(_os_hub.path.dirname(_os_hub.path.abspath(dst)), exist_ok=True)
        tmp = dst + ".partial"
        if _os_hub.path.exists(tmp):
            try:
                _os_hub.remove(tmp)
            except OSError:
                pass
        _urlreq_hub.urlretrieve(url, tmp)
        if (not _os_hub.path.isfile(tmp)) or _os_hub.path.getsize(tmp) == 0:
            try:
                _os_hub.remove(tmp)
            except OSError:
                pass
            raise RuntimeError(f"downloaded empty checkpoint from {url}")
        _os_hub.replace(tmp, dst)
    def _load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True,
                                  check_hash=False, file_name=None, weights_only=False):
        if model_dir is None:
            model_dir = _hub_checkpoints_dir()
        _os_hub.makedirs(model_dir, exist_ok=True)
        filename = file_name or _os_hub.path.basename(_urlparse_hub(url).path)
        cached_file = _os_hub.path.join(model_dir, filename)
        if (not _os_hub.path.isfile(cached_file)) or _os_hub.path.getsize(cached_file) == 0:
            if _os_hub.path.exists(cached_file):
                try:
                    _os_hub.remove(cached_file)
                except OSError:
                    pass
            _download_url_to_file(url, cached_file, progress=progress)
        return g.load(cached_file, map_location=map_location, weights_only=weights_only)
    hub.download_url_to_file = _download_url_to_file
    hub.load_state_dict_from_url = _load_state_dict_from_url
    hub.get_dir = lambda: _os_hub.path.join(_hub_dir(), "hub")
    import re as _re_hub
    hub.HASH_REGEX = _re_hub.compile(r"-([a-f0-9]*)\\.")
    hub.tqdm = None
    hub.urlparse = _urlparse_hub
    hub.urlopen = _urlreq_hub.urlopen
    hub.Request = _urlreq_hub.Request
    hub._get_torch_home = hub.get_dir
    g.hub = hub
    import sys as _sys_hub
    _sys_hub.modules.setdefault("torch.hub", hub)

    # ---- elementwise / reduction helpers that may be missing ----
    def _alias(name, fn):
        if not hasattr(g, name):
            setattr(g, name, fn)
    _alias("rsqrt", lambda x: 1.0 / jt.sqrt(x))
    _alias("empty_like", lambda x, **k: jt.empty(x.shape, x.dtype))
    # module-level comparison ops (torch.gt(a,b) etc.); .gt methods already exist.
    _alias("gt", lambda a, b: a > b)
    _alias("lt", lambda a, b: a < b)
    _alias("ge", lambda a, b: a >= b)
    _alias("le", lambda a, b: a <= b)
    _alias("eq", lambda a, b: a == b)
    # torch.compile: jittor already JIT-compiles every op, so this is a pass-through.
    # Handles torch.compile(model), @torch.compile, and torch.compile(mode=...)(model).
    def _compile(model=None, *a, **k):
        return model if model is not None else (lambda m: m)
    _alias("compile", _compile)
    # torch.jit: jittor has no TorchScript; the script/trace decorators are pass-throughs
    # (the eager fn already runs), and is_scripting/is_tracing report False.
    import types as _types2
    _compiler = getattr(g, "compiler", None) or _types2.ModuleType("torch.compiler")
    _cid = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda h: h))
    _compiler.is_compiling = lambda: False
    _compiler.is_dynamo_compiling = lambda: False
    _compiler.is_exporting = lambda: False
    _compiler.disable = _cid
    _compiler.allow_in_graph = _cid
    _compiler.assume_constant_result = _cid
    _compiler.wrap_numpy = _cid
    _compiler.reset = lambda *a, **k: None
    _compiler.cudagraph_mark_step_begin = lambda *a, **k: None
    import sys as _sys_compiler
    _sys_compiler.modules["torch.compiler"] = _compiler
    if not hasattr(g, "compiler"):
        g.compiler = _compiler
    _jit = _types2.SimpleNamespace()
    _jit.script = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _jit.trace = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))
    _jit.script_if_tracing = lambda f: f
    _jit.ignore = lambda f=None, **k: (f if callable(f) else (lambda g: g))
    _jit.unused = lambda f: f
    _jit.export = lambda f: f
    _jit.is_scripting = lambda: False
    _jit.is_tracing = lambda: False
    _jit.ScriptModule = jt.nn.Module
    _jit.interface = lambda c: c
    _alias("jit", _jit)
    _alias("ScriptModule", _jit.ScriptModule)
    _sys_compiler.modules.setdefault("torch.jit", _jit)
    _fx = _types2.ModuleType("torch.fx")
    _fx.Graph = type("Graph", (), {})
    _fx.GraphModule = type("GraphModule", (), {})
    _fx.Proxy = type("Proxy", (), {})
    _fx.Node = type("Node", (), {})
    _fx.wrap = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda h: h))
    _sys_compiler.modules["torch.fx"] = _fx
    g.fx = _fx
    # torch._dynamo: minimal importable stubs for libraries that probe or
    # decorate with Dynamo APIs. Jittor runs eagerly/JIT through its own stack.
    _dynamo = _types2.ModuleType("torch._dynamo")
    _dynamo.disable = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.allow_in_graph = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.disallow_in_graph = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.assume_constant_result = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.is_compiling = lambda: False
    _dynamo.is_dynamo_compiling = lambda: False
    _dynamo.config = _types2.SimpleNamespace()
    _dynamo.mark_static_address = lambda *a, **k: None
    _dynamo.mark_dynamic = lambda *a, **k: None
    _dynamo.graph_break = lambda *a, **k: None
    _dynamo.reset = lambda *a, **k: None
    _sys_compiler.modules["torch._dynamo"] = _dynamo
    setattr(g, "_dynamo", _dynamo)
    _eval_frame = _types2.ModuleType("torch._dynamo.eval_frame")
    _eval_frame.OptimizedModule = type("OptimizedModule", (jt.nn.Module,), {})
    _eval_frame.is_dynamo_supported = lambda: False
    _dynamo.OptimizedModule = _eval_frame.OptimizedModule
    _dynamo.eval_frame = _eval_frame
    _sys_compiler.modules["torch._dynamo.eval_frame"] = _eval_frame
    _twh = _types2.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")
    class TransformGetItemToIndex:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    _twh.TransformGetItemToIndex = TransformGetItemToIndex
    _sys_compiler.modules["torch._dynamo._trace_wrapped_higher_order_op"] = _twh
    _functorch_pkg = _types2.ModuleType("torch._functorch")
    _functorch_vmap = _types2.ModuleType("torch._functorch.vmap")
    _functorch_vmap._maybe_remove_batch_dim = lambda x, *a, **k: x
    _functorch_vmap._add_batch_dim = lambda x, *a, **k: x
    _functorch_vmap._remove_batch_dim = lambda x, *a, **k: x
    def _vmap_tree_flatten(x, *args, **kwargs):
        return g.utils._pytree.tree_flatten(x)
    def _vmap_tree_unflatten(leaves, spec):
        return g.utils._pytree.tree_unflatten(leaves, spec)
    def _vmap_broadcast_to_and_flatten(in_dims, spec):
        leaves = g.utils._pytree.tree_leaves(spec)
        n = len(leaves) if leaves else 1
        if isinstance(in_dims, (list, tuple)):
            flat, _ = g.utils._pytree.tree_flatten(in_dims)
            return flat if len(flat) == n else None
        return [in_dims] * n
    def _vmap_validate_and_get_batch_size(flat_in_dims, flat_args):
        for in_dim, arg in zip(flat_in_dims, flat_args):
            if in_dim is not None and hasattr(arg, "shape"):
                return int(arg.shape[in_dim])
        return 0
    _functorch_vmap._broadcast_to_and_flatten = _vmap_broadcast_to_and_flatten
    _functorch_vmap._get_name = lambda func: getattr(func, "__name__", str(func))
    _functorch_vmap._validate_and_get_batch_size = _vmap_validate_and_get_batch_size
    _functorch_vmap.Tensor = getattr(g, "Tensor", jt.Var)
    _functorch_vmap.tree_flatten = _vmap_tree_flatten
    _functorch_vmap.tree_unflatten = _vmap_tree_unflatten
    _functorch_pkg.vmap = _functorch_vmap
    _sys_compiler.modules["torch._functorch"] = _functorch_pkg
    _sys_compiler.modules["torch._functorch.vmap"] = _functorch_vmap
    setattr(g, "_functorch", _functorch_pkg)
    _library = _types2.ModuleType("torch.library")
    class _OpNamespace:
        def __init__(self, ns):
            object.__setattr__(self, "_ns", ns)
            object.__setattr__(self, "_ops", {})
        def _register(self, name, fn):
            object.__getattribute__(self, "_ops")[name] = fn
        def __getattr__(self, name):
            ops = object.__getattribute__(self, "_ops")
            if name in ops:
                return ops[name]
            raise AttributeError("torch.ops.%s has no op '%s'" % (
                object.__getattribute__(self, "_ns"), name))
    class _OpsDispatcher:
        def __init__(self, base):
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "_ns", {})
        def _register(self, ns, name, fn):
            namespaces = object.__getattribute__(self, "_ns")
            namespaces.setdefault(ns, _OpNamespace(ns))._register(name, fn)
        def __getattr__(self, name):
            namespaces = object.__getattribute__(self, "_ns")
            if name in namespaces:
                return namespaces[name]
            base = object.__getattribute__(self, "_base")
            if base is not None:
                return getattr(base, name)
            raise AttributeError(name)
    _ops_dispatcher = getattr(g, "ops", None)
    if not isinstance(_ops_dispatcher, _OpsDispatcher):
        _ops_dispatcher = _OpsDispatcher(_ops_dispatcher)
    def _grouped_mm_fallback(input, weight, offs, *a, **k):
        out = jt.zeros((input.shape[0], weight.shape[2]), dtype=input.dtype)
        offs_list = offs.numpy().tolist() if hasattr(offs, "numpy") else list(offs)
        start = 0
        for i, end in enumerate(offs_list):
            end = int(end)
            if end > start:
                out[start:end] = jt.matmul(input[start:end], weight[i])
            start = end
        return out
    def _custom_op(name=None, fn=None, *a, **k):
        def deco(impl):
            if isinstance(name, str) and "::" in name:
                ns, op = name.split("::", 1)
                real = _grouped_mm_fallback if name == "transformers::grouped_mm_fallback" else impl
                _ops_dispatcher._register(ns, op, real)
            return impl
        return deco(fn) if fn is not None else deco
    _library.custom_op = _custom_op
    _library.register_fake = lambda *a, **k: (lambda f: f)
    _library.register_kernel = lambda *a, **k: (lambda f: f)
    _library.impl = lambda *a, **k: (lambda f: f)
    _library.register_autograd = lambda *a, **k: (lambda f: f)
    _library.register_torch_dispatch = lambda *a, **k: (lambda f: f)
    _library.register_vmap = lambda *a, **k: (lambda f: f)
    _library.opcheck = lambda *a, **k: None
    _library.get_ctx = lambda: None
    _library.Library = type("Library", (), {
        "__init__": lambda self, *a, **k: None,
        "define": lambda self, *a, **k: None,
        "impl": lambda self, *a, **k: None,
    })
    import sys as _sys_library
    _sys_library.modules["torch.library"] = _library
    g.library = _library
    g.ops = _ops_dispatcher

    # torch.profiler: accelerate/transformers reference this namespace at
    # import-time for type annotations and optional profiling config. Do not
    # expose jittor_core.profiler here; it lacks PyTorch's ProfilerActivity API.
    _profiler = _types2.ModuleType("torch.profiler")
    class ProfilerActivity:
        CPU = "cpu"
        CUDA = "cuda"
        XPU = "xpu"
        HPU = "hpu"
        MTIA = "mtia"
    class _ProfilerAction:
        NONE = "none"
        WARMUP = "warmup"
        RECORD = "record"
        RECORD_AND_SAVE = "record_and_save"
    class _ProfileContext:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def step(self):
            pass
        def export_chrome_trace(self, *args, **kwargs):
            pass
    _profiler.ProfilerActivity = ProfilerActivity
    _profiler.ProfilerAction = _ProfilerAction
    _profiler.profile = lambda *args, **kwargs: _ProfileContext()
    _profiler.schedule = lambda *args, **kwargs: (lambda step: _ProfilerAction.NONE)
    _profiler.tensorboard_trace_handler = lambda *args, **kwargs: (lambda *a, **k: None)
    _profiler.record_function = lambda *args, **kwargs: _ProfileContext()
    _profiler.kineto_available = lambda: False
    _sys_library.modules["torch.profiler"] = _profiler
    g.profiler = _profiler

    if "torch.utils.tensorboard" not in _sys_library.modules:
        _tb = _types2.ModuleType("torch.utils.tensorboard")
        class SummaryWriter:
            def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10,
                         flush_secs=120, filename_suffix="", *args, **kwargs):
                self.log_dir = log_dir
                self.comment = comment
                self.purge_step = purge_step
                self.max_queue = max_queue
                self.flush_secs = flush_secs
                self.filename_suffix = filename_suffix
                self.args = args
                self.kwargs = kwargs
            def add_scalar(self, *a, **k): return None
            def add_scalars(self, *a, **k): return None
            def add_image(self, *a, **k): return None
            def add_images(self, *a, **k): return None
            def add_graph(self, *a, **k): return None
            def add_histogram(self, *a, **k): return None
            def add_text(self, *a, **k): return None
            def flush(self): return None
            def close(self): return None
            def __enter__(self): return self
            def __exit__(self, *exc): self.close(); return False
        _tb.SummaryWriter = SummaryWriter
        _sys_library.modules["torch.utils.tensorboard"] = _tb

    _amp = _types2.ModuleType("torch.amp")
    _amp.autocast = lambda *args, **kwargs: _AutocastContext()
    _amp.GradScaler = _GradScaler
    _amp.custom_fwd = _amp_passthrough_decorator
    _amp.custom_bwd = _amp_passthrough_decorator
    _sys_library.modules["torch.amp"] = _amp
    g.amp = _amp
    try:
        if hasattr(g, "cuda"):
            if not hasattr(g.cuda, "amp"):
                g.cuda.amp = _types2.ModuleType("torch.cuda.amp")
            g.cuda.amp.autocast = _amp.autocast
            g.cuda.amp.GradScaler = _GradScaler
            g.cuda.amp.custom_fwd = _amp_passthrough_decorator
            g.cuda.amp.custom_bwd = _amp_passthrough_decorator
            _sys_library.modules["torch.cuda.amp"] = g.cuda.amp
    except Exception:
        pass

    # `import jittor as torch; torch.utils.data.Dataset` (attribute access, used by some
    # HF/training code as a base class) needs a `utils` namespace on the jittor module --
    # the `from torch.utils.data import X` form already resolves via sys.modules. Lazily
    # resolve torch.utils.<sub> (data/checkpoint/rnn/...) on access.
    import sys as _sys_utils
    if not hasattr(g, "utils") or not isinstance(getattr(g, "utils"), _types2.ModuleType):
        class _UtilsNS(_types2.ModuleType):
            def __getattr__(self, name):
                full = "torch.utils." + name
                if full in _sys_utils.modules:
                    return _sys_utils.modules[full]
                raise AttributeError(name)
        g.utils = _UtilsNS("torch.utils")
    g.utils.__path__ = []
    g.utils.__package__ = "torch"
    _sys_utils.modules["torch.utils"] = g.utils
    if "torch.utils.tensorboard" in _sys_utils.modules:
        g.utils.tensorboard = _sys_utils.modules["torch.utils.tensorboard"]
    if "torch.utils.data" not in _sys_utils.modules:
        _data = _types2.ModuleType("torch.utils.data")
        class _TorchDataset:
            def __getitem__(self, i):
                raise NotImplementedError
            def __add__(self, other):
                return _ConcatDataset([self, other])
        class _IterableDataset(_TorchDataset):
            def __iter__(self):
                raise NotImplementedError
        class _TensorDataset(_TorchDataset):
            def __init__(self, *tensors):
                self.tensors = tensors
            def __getitem__(self, i):
                return tuple(t[i] for t in self.tensors)
            def __len__(self):
                return len(self.tensors[0]) if self.tensors else 0
        class _ConcatDataset(_TorchDataset):
            def __init__(self, datasets):
                self.datasets = list(datasets)
                self.cumulative_sizes = []
                total = 0
                for dataset in self.datasets:
                    total += len(dataset)
                    self.cumulative_sizes.append(total)
            def __len__(self):
                return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
            def __getitem__(self, idx):
                import bisect as _bisect
                dataset_idx = _bisect.bisect_right(self.cumulative_sizes, idx)
                prev = self.cumulative_sizes[dataset_idx - 1] if dataset_idx else 0
                return self.datasets[dataset_idx][idx - prev]
        class _Subset(_TorchDataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = list(indices)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        class _Sampler:
            def __init__(self, data_source=None):
                self.data_source = data_source
            def __iter__(self):
                raise NotImplementedError
        class _SequentialSampler(_Sampler):
            def __iter__(self):
                return iter(range(len(self.data_source)))
            def __len__(self):
                return len(self.data_source)
        class _RandomSampler(_Sampler):
            def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
                self.data_source = data_source
                self.replacement = replacement
                self._num_samples = num_samples
                self.generator = generator
            @property
            def num_samples(self):
                return len(self.data_source) if self._num_samples is None else self._num_samples
            def __iter__(self):
                import random as _random
                n = len(self.data_source)
                if self.replacement:
                    return iter(_random.randrange(n) for _ in range(self.num_samples))
                indices = list(range(n))
                _random.shuffle(indices)
                return iter(indices[:self.num_samples])
            def __len__(self):
                return self.num_samples
        class _SubsetRandomSampler(_Sampler):
            def __init__(self, indices, generator=None):
                self.indices = list(indices)
                self.generator = generator
            def __iter__(self):
                import random as _random
                indices = list(self.indices)
                _random.shuffle(indices)
                return iter(indices)
            def __len__(self):
                return len(self.indices)
        class _BatchSampler(_Sampler):
            def __init__(self, sampler, batch_size, drop_last):
                self.sampler = sampler
                self.batch_size = int(batch_size)
                self.drop_last = bool(drop_last)
            def __iter__(self):
                batch = []
                for idx in self.sampler:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                if batch and not self.drop_last:
                    yield batch
            def __len__(self):
                n = len(self.sampler)
                return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        class _DistributedSampler(_Sampler):
            def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                         seed=0, drop_last=False):
                import math as _math
                self.dataset = dataset
                self.num_replicas = 1 if num_replicas is None else int(num_replicas)
                self.rank = 0 if rank is None else int(rank)
                self.shuffle = bool(shuffle)
                self.seed = int(seed)
                self.drop_last = bool(drop_last)
                self.epoch = 0
                if self.drop_last and len(self.dataset) % self.num_replicas != 0:
                    self.num_samples = _math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
                else:
                    self.num_samples = _math.ceil(len(self.dataset) / self.num_replicas)
                self.total_size = self.num_samples * self.num_replicas
            def __iter__(self):
                import random as _random
                indices = list(range(len(self.dataset)))
                if self.shuffle:
                    rng = _random.Random(self.seed + self.epoch)
                    rng.shuffle(indices)
                if not self.drop_last:
                    padding = self.total_size - len(indices)
                    if padding > 0:
                        indices += (indices * ((padding + len(indices) - 1) // len(indices)))[:padding]
                else:
                    indices = indices[:self.total_size]
                return iter(indices[self.rank:self.total_size:self.num_replicas])
            def __len__(self):
                return self.num_samples
            def set_epoch(self, epoch):
                self.epoch = int(epoch)
        def _default_collate(batch):
            import numpy as _np
            elem = batch[0]
            if isinstance(elem, jt.Var):
                return jt.stack(list(batch), dim=0)
            if isinstance(elem, _np.ndarray):
                return jt.array(_np.stack(batch))
            if isinstance(elem, (type(0), type(0.0), _np.number)):
                return jt.array(_np.array(batch))
            if isinstance(elem, (tuple, list)):
                return [_default_collate(list(items)) for items in zip(*batch)]
            if isinstance(elem, dict):
                return {key: _default_collate([d[key] for d in batch]) for key in elem}
            return batch
        class _BaseDataLoaderIter:
            def __iter__(self):
                return self

        class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
            def __init__(self, loader):
                self._loader = loader
                self._batch_iter = iter(loader.batch_sampler)

            def __next__(self):
                batch_indices = next(self._batch_iter)
                return self._loader.collate_fn([self._loader.dataset[i] for i in batch_indices])

        class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
            pass

        class _DataLoader:
            def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                         batch_sampler=None, num_workers=0, collate_fn=None,
                         pin_memory=False, drop_last=False, timeout=0,
                         worker_init_fn=None, generator=None, prefetch_factor=None,
                         persistent_workers=False, **kwargs):
                self.dataset = dataset
                self.batch_size = batch_size
                self.drop_last = drop_last
                self.num_workers = num_workers
                self.pin_memory = pin_memory
                self.timeout = timeout
                self.prefetch_factor = prefetch_factor
                self.persistent_workers = persistent_workers
                self.multiprocessing_context = kwargs.get("multiprocessing_context", None)
                self.shuffle = shuffle
                self.collate_fn = collate_fn if collate_fn is not None else _default_collate
                self.worker_init_fn = worker_init_fn
                self.generator = generator
                if batch_sampler is not None:
                    self.batch_sampler = batch_sampler
                    self.sampler = None
                else:
                    self.sampler = sampler if sampler is not None else (
                        _RandomSampler(dataset, generator=generator) if shuffle else _SequentialSampler(dataset)
                    )
                    self.batch_sampler = _BatchSampler(self.sampler, batch_size, drop_last)
                self._iterator = None
            def __iter__(self):
                self._iterator = _SingleProcessDataLoaderIter(self)
                return self._iterator
            def __len__(self):
                return len(self.batch_sampler)
        for _name, _value in {
            "Dataset": _TorchDataset,
            "IterableDataset": _IterableDataset,
            "TensorDataset": _TensorDataset,
            "ConcatDataset": _ConcatDataset,
            "Subset": _Subset,
            "Sampler": _Sampler,
            "SequentialSampler": _SequentialSampler,
            "RandomSampler": _RandomSampler,
            "SubsetRandomSampler": _SubsetRandomSampler,
            "BatchSampler": _BatchSampler,
            "DistributedSampler": _DistributedSampler,
            "DataLoader": _DataLoader,
            "default_collate": _default_collate,
            "default_convert": lambda x: x,
            "get_worker_info": lambda: None,
        }.items():
            setattr(_data, _name, _value)
        _sys_utils.modules["torch.utils.data"] = _data
        g.utils.data = _data
        _du = _types2.ModuleType("torch.utils.data._utils")
        _duc = _types2.ModuleType("torch.utils.data._utils.collate")
        _duw = _types2.ModuleType("torch.utils.data._utils.worker")
        def _generate_state(base_seed, worker_id):
            import random as _random_worker
            rng = _random_worker.Random(int(base_seed) + int(worker_id))
            return [rng.randrange(0, 2**32) for _ in range(4)]
        _duc.default_collate = _default_collate
        _du.collate = _duc
        _duw._generate_state = _generate_state
        _du.worker = _duw
        _sys_utils.modules["torch.utils.data._utils"] = _du
        _sys_utils.modules["torch.utils.data._utils.collate"] = _duc
        _sys_utils.modules["torch.utils.data._utils.worker"] = _duw
        _data._utils = _du
        _dist_data = _types2.ModuleType("torch.utils.data.distributed")
        _dist_data.DistributedSampler = _DistributedSampler
        _sys_utils.modules["torch.utils.data.distributed"] = _dist_data
        _data.distributed = _dist_data
        _dataset_mod = _types2.ModuleType("torch.utils.data.dataset")
        for _name in ("Dataset", "IterableDataset", "TensorDataset", "ConcatDataset", "Subset"):
            setattr(_dataset_mod, _name, getattr(_data, _name))
        _sys_utils.modules["torch.utils.data.dataset"] = _dataset_mod
        _data.dataset = _dataset_mod
        _sampler_mod = _types2.ModuleType("torch.utils.data.sampler")
        for _name in ("Sampler", "SequentialSampler", "RandomSampler", "SubsetRandomSampler", "BatchSampler", "DistributedSampler"):
            setattr(_sampler_mod, _name, getattr(_data, _name))
        _sys_utils.modules["torch.utils.data.sampler"] = _sampler_mod
        _data.sampler = _sampler_mod
        _dataloader_mod = _types2.ModuleType("torch.utils.data.dataloader")
        _dataloader_mod.DataLoader = _DataLoader
        _dataloader_mod.default_collate = _default_collate
        _dataloader_mod._DatasetKind = type("_DatasetKind", (), {"Iterable": 0, "Map": 1})
        _dataloader_mod._BaseDataLoaderIter = _BaseDataLoaderIter
        _dataloader_mod._SingleProcessDataLoaderIter = _SingleProcessDataLoaderIter
        _dataloader_mod._MultiProcessingDataLoaderIter = _MultiProcessingDataLoaderIter
        _sys_utils.modules["torch.utils.data.dataloader"] = _dataloader_mod
        _data.dataloader = _dataloader_mod
    else:
        g.utils.data = _sys_utils.modules["torch.utils.data"]
    if "torch.utils.checkpoint" not in _sys_utils.modules:
        _ckpt = _types2.ModuleType("torch.utils.checkpoint")
        def _checkpoint(fn, *args, use_reentrant=None, **kwargs):
            return fn(*args, **kwargs)
        _ckpt.checkpoint = _checkpoint
        _sys_utils.modules["torch.utils.checkpoint"] = _ckpt
        g.utils.checkpoint = _ckpt
    if "torch.utils._pytree" not in _sys_utils.modules:
        _pytree = _types2.ModuleType("torch.utils._pytree")
        class LeafSpec:
            pass
        class TreeSpec:
            def __init__(self, type, context, children_specs):
                self.type = type
                self.context = context
                self.children_specs = list(children_specs)
        class MappingKey:
            def __init__(self, key):
                self.key = key
            def __hash__(self):
                return hash(self.key)
            def __eq__(self, other):
                return isinstance(other, MappingKey) and self.key == other.key
            def __repr__(self):
                return f"[{self.key!r}]"
        class SequenceKey:
            def __init__(self, idx):
                self.idx = idx
            def __hash__(self):
                return hash(self.idx)
            def __eq__(self, other):
                return isinstance(other, SequenceKey) and self.idx == other.idx
            def __repr__(self):
                return f"[{self.idx}]"
        class GetAttrKey:
            def __init__(self, name):
                self.name = name
            def __hash__(self):
                return hash(self.name)
            def __eq__(self, other):
                return isinstance(other, GetAttrKey) and self.name == other.name
            def __repr__(self):
                return "." + str(self.name)
        _NodeDef = type("_NodeDef", (), {})
        def _list_flatten(x):
            return list(x), None
        def _list_unflatten(values, context):
            return list(values)
        def _list_flatten_with_keys(x):
            return [(i, v) for i, v in enumerate(x)], None
        def _tuple_flatten(x):
            return list(x), None
        def _dict_flatten(x):
            keys = list(x.keys())
            return [x[k] for k in keys], keys
        def _dict_unflatten(values, context):
            return {k: v for k, v in zip(context, values)}
        def _get_node_type(x):
            return dict if isinstance(x, dict) else type(x)
        SUPPORTED_NODES = {
            list: _NodeDef(),
            tuple: _NodeDef(),
            dict: _NodeDef(),
        }
        SUPPORTED_NODES[list].flatten_fn = _list_flatten
        SUPPORTED_NODES[tuple].flatten_fn = _tuple_flatten
        SUPPORTED_NODES[dict].flatten_fn = _dict_flatten
        def _tree_flatten(x):
            leaves = []
            def rec(o):
                node_type = _get_node_type(o)
                if node_type not in SUPPORTED_NODES:
                    leaves.append(o)
                    return LeafSpec()
                child_pytrees, context = SUPPORTED_NODES[node_type].flatten_fn(o)
                child_specs = [rec(c) for c in child_pytrees]
                return TreeSpec(node_type, context, child_specs)
            return leaves, rec(x)
        def _tree_unflatten(leaves, spec):
            it = iter(leaves)
            def rec(s):
                if isinstance(s, LeafSpec):
                    return next(it)
                children = [rec(c) for c in s.children_specs]
                if s.type is tuple:
                    return tuple(children)
                if s.type is dict:
                    return {k: v for k, v in zip(s.context, children)}
                return list(children)
            return rec(spec)
        _pytree.SUPPORTED_NODES = SUPPORTED_NODES
        _pytree.LeafSpec = LeafSpec
        _pytree.TreeSpec = TreeSpec
        _pytree.PyTree = object
        _pytree.Context = object
        _pytree.MappingKey = MappingKey
        _pytree.SequenceKey = SequenceKey
        _pytree.GetAttrKey = GetAttrKey
        _pytree.KeyEntry = (MappingKey, SequenceKey, GetAttrKey)
        _pytree.FlattenFunc = object
        _pytree.UnflattenFunc = object
        _pytree._get_node_type = _get_node_type
        _pytree._list_flatten = _list_flatten
        _pytree._list_unflatten = _list_unflatten
        _pytree._list_flatten_with_keys = _list_flatten_with_keys
        _pytree._dict_flatten = _dict_flatten
        _pytree._dict_unflatten = _dict_unflatten
        _pytree.tree_flatten = _tree_flatten
        _pytree.tree_unflatten = _tree_unflatten
        _pytree.tree_map = lambda f, x: f(x)
        _pytree.tree_map_only = lambda typ, f, x: f(x) if isinstance(x, typ) else x
        _pytree.tree_leaves = lambda x: _tree_flatten(x)[0]
        _pytree.register_pytree_node = lambda *a, **k: None
        _pytree._register_pytree_node = lambda *a, **k: None
        _sys_utils.modules["torch.utils._pytree"] = _pytree
    g.utils._pytree = _sys_utils.modules["torch.utils._pytree"]
    if "torch.utils._contextlib" not in _sys_utils.modules:
        _contextlib_mod = _types2.ModuleType("torch.utils._contextlib")
        import contextlib as _ctxlib_utils
        class _DecoratorContextManager(_ctxlib_utils.ContextDecorator):
            def clone(self):
                return type(self)()
            def __call__(self, orig_func):
                return super().__call__(orig_func)
        _contextlib_mod._DecoratorContextManager = _DecoratorContextManager
        _sys_utils.modules["torch.utils._contextlib"] = _contextlib_mod
    g.utils._contextlib = _sys_utils.modules["torch.utils._contextlib"]
    if "torch.utils.hooks" not in _sys_utils.modules:
        _hooks = _types2.ModuleType("torch.utils.hooks")
        class RemovableHandle:
            def __init__(self, hooks_dict=None, *args, **kwargs):
                self.hooks_dict = hooks_dict
                try:
                    self.id = max(hooks_dict.keys(), default=0) + 1 if hooks_dict is not None else 0
                except Exception:
                    self.id = 0
            def remove(self):
                try:
                    if self.hooks_dict is not None:
                        self.hooks_dict.pop(self.id, None)
                except Exception:
                    pass
        _hooks.RemovableHandle = RemovableHandle
        _sys_utils.modules["torch.utils.hooks"] = _hooks
    g.utils.hooks = _sys_utils.modules["torch.utils.hooks"]
    if "torch.utils.dlpack" not in _sys_utils.modules:
        _dlpack = _types2.ModuleType("torch.utils.dlpack")
        def _dlpack_not_implemented(*args, **kwargs):
            raise NotImplementedError("torch.utils.dlpack is not implemented by jittor torch_compat")
        _dlpack.from_dlpack = _dlpack_not_implemented
        _dlpack.to_dlpack = _dlpack_not_implemented
        _sys_utils.modules["torch.utils.dlpack"] = _dlpack
    g.utils.dlpack = _sys_utils.modules["torch.utils.dlpack"]
    if "torch._subclasses.fake_tensor" not in _sys_utils.modules:
        _subclasses = _types2.ModuleType("torch._subclasses")
        _fake_tensor = _types2.ModuleType("torch._subclasses.fake_tensor")
        _functional_tensor = _types2.ModuleType("torch._subclasses.functional_tensor")
        _fake_tensor.FakeTensor = type("FakeTensor", (), {})
        _fake_tensor.FakeTensorMode = type("FakeTensorMode", (), {
            "__init__": lambda self, *a, **k: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: False,
        })
        _functional_tensor.FunctionalTensor = type("FunctionalTensor", (), {})
        _subclasses.fake_tensor = _fake_tensor
        _subclasses.functional_tensor = _functional_tensor
        _sys_utils.modules["torch._subclasses"] = _subclasses
        _sys_utils.modules["torch._subclasses.fake_tensor"] = _fake_tensor
        _sys_utils.modules["torch._subclasses.functional_tensor"] = _functional_tensor
        setattr(g, "_subclasses", _subclasses)
    if "torch.utils.flop_counter" not in _sys_utils.modules:
        _flop_counter = _types2.ModuleType("torch.utils.flop_counter")
        class FlopCounterMode:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_total_flops(self):
                return 0

            def get_flop_counts(self):
                return {}
        _flop_counter.FlopCounterMode = FlopCounterMode
        _sys_utils.modules["torch.utils.flop_counter"] = _flop_counter
    g.utils.flop_counter = _sys_utils.modules["torch.utils.flop_counter"]
    try:
        from jittor.torch_shim.cpp_extension.torch_utils import install_cpp_extension
        install_cpp_extension(g.utils)
    except Exception:
        # Keep `import jittor as torch` usable in minimal/non-setuptools envs; an
        # actual extension build will fail with the concrete import/toolchain error.
        pass
    # torch.func (functorch): functional transforms used by LoRA / meta-learning /
    # model ensembling (functorch). Jittor's autograd is graph-based, so these are
    # thin wrappers over jt.grad + temporary parameter rebinding.
    def _func_resolve(module, name):
        # navigate module.<a>.<b>.<2>... -> (owner, leaf_attr); supports int (Sequential)
        owner = module
        parts = name.split(".")
        for p in parts[:-1]:
            if p.isdigit() and hasattr(owner, "__getitem__"):
                owner = owner[int(p)]
            else:
                owner = getattr(owner, p)
        return owner, parts[-1]

    def _functional_call(module, parameters_and_buffers, args=None, kwargs=None,
                         *, tie_weights=True, strict=False, **_):
        # torch.func.functional_call: run module.forward with the given params/buffers
        # swapped in (then restored), without mutating the module. Accepts a dict or a
        # sequence of dicts (merged), matching torch.
        if args is None:
            args = ()
        elif isinstance(args, jt.Var) or not isinstance(args, (tuple, list)):
            args = (args,)
        else:
            args = tuple(args)
        if kwargs is None:
            kwargs = {}
        if isinstance(parameters_and_buffers, (list, tuple)):
            merged = {}
            for d in parameters_and_buffers:
                merged.update(d)
            parameters_and_buffers = merged
        saved = []
        try:
            for name, val in parameters_and_buffers.items():
                owner, attr = _func_resolve(module, name)
                saved.append((owner, attr, getattr(owner, attr, None)))
                setattr(owner, attr, val)
            return module(*args, **kwargs)
        finally:
            for owner, attr, orig in reversed(saved):
                setattr(owner, attr, orig)

    def _func_grad_core(f, argnums, has_aux, want_value):
        def wrapped(*args, **kwargs):
            single = isinstance(argnums, int)
            nums = (argnums,) if single else tuple(argnums)
            inputs = [args[i] for i in nums]
            out = f(*args, **kwargs)
            aux = None
            if has_aux:
                out, aux = out
            grads = jt.grad(out, inputs)            # list, aligned with inputs
            g0 = grads[0] if single else tuple(grads)
            if want_value:
                val = (out, aux) if has_aux else out
                return (g0, val)
            return (g0, aux) if has_aux else g0
        return wrapped

    def _func_grad(f, argnums=0, has_aux=False):
        return _func_grad_core(f, argnums, has_aux, want_value=False)

    def _func_grad_and_value(f, argnums=0, has_aux=False):
        return _func_grad_core(f, argnums, has_aux, want_value=True)

    def _jacrev(f, argnums=0):
        # reverse-mode Jacobian: one backward pass per scalar output component.
        def wrapped(*args, **kwargs):
            x = args[argnums]
            out = f(*args, **kwargs)
            flat = out.reshape(-1)
            rows = [jt.grad(flat[i], [x])[0].reshape(-1) for i in range(int(flat.shape[0]))]
            J = jt.stack(rows, dim=0)
            return J.reshape(list(out.shape) + list(x.shape))
        return wrapped

    def _stack_module_state(models):
        from collections import OrderedDict
        models = list(models)
        ps = [dict(m.named_parameters()) for m in models]
        bs = [dict(m.named_buffers()) for m in models]
        params = OrderedDict((k, jt.stack([d[k] for d in ps], dim=0)) for k in ps[0])
        buffers = OrderedDict((k, jt.stack([d[k] for d in bs], dim=0))
                              for k in (bs[0] if bs and bs[0] else {}))
        return params, buffers

    _func_ns = _types2.SimpleNamespace()
    _func_ns.functional_call = _functional_call
    _func_ns.grad = _func_grad
    _func_ns.grad_and_value = _func_grad_and_value
    _func_ns.vmap = lambda *a, **k: g.vmap(*a, **k)   # _vmap is defined later in this fn
    _func_ns.jacrev = _jacrev
    _func_ns.jacfwd = _jacrev          # same numerics; forward-mode falls back to reverse
    _func_ns.stack_module_state = _stack_module_state
    _func_ns.functionalize = lambda fn, **k: fn
    _alias("func", _func_ns)
    # torch.nn.utils also exposes stateless.functional_call (older API path).
    if not hasattr(g, "functional_call"):
        g.functional_call = _functional_call
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
    def _is_cplx(x):
        return isinstance(x, _CN) or (isinstance(x, Var) and "complex" in str(x.dtype))
    _alias("complex", lambda real, imag, **k: jt.nn.view_as_complex(jt.stack([real, imag], dim=-1)))  # native complex64
    _alias("view_as_complex", lambda x: jt.nn.view_as_complex(x))   # -> native complex64
    _alias("view_as_real", lambda x: jt.nn.view_as_real(x))         # polymorphic
    g.is_complex = lambda x: _is_cplx(x)
    g.real = lambda x: x.real if isinstance(x, (_CN, Var)) else x
    g.imag = lambda x: x.imag if isinstance(x, (_CN, Var)) else jt.zeros_like(x)
    g.polar = lambda abs, angle, **k: jt.nn.polar(abs, angle)       # -> native complex64
    g.conj = lambda x: x.conj() if isinstance(x, (_CN, Var)) else x
    g.angle = lambda x: x.angle() if isinstance(x, (_CN, Var)) else jt.zeros_like(x)
    # torch.abs of a complex tensor is its magnitude; jittor's abs only takes real Vars.
    _jt_abs = jt.abs
    def _abs(x):
        return x.abs() if isinstance(x, _CN) else _jt_abs(x)
    g.abs = _abs
    Var.abs = lambda self: _jt_abs(self)

    # torch.fft.* (#3): jittor only has a CUDA-only cufft fft2, so provide 1-D fft/ifft/
    # rfft/irfft via DFT matrices (out = x @ W^T, matmul-based -> dual-card, autograd-
    # able, correct). O(N^2) but fine for the moderate N these are used at.
    import types as _types
    import numpy as _np
    def _dft_mats(N, inverse):
        idx = _np.arange(N)
        ang = (2.0 * _np.pi / N) * _np.outer(idx, idx) * (1.0 if inverse else -1.0)
        return jt.array(_np.cos(ang).astype("float32")), jt.array(_np.sin(ang).astype("float32"))
    def _to_last(x, dim):
        nd = (x.real.ndim if isinstance(x, _CN) else x.ndim)
        d = dim if dim >= 0 else dim + nd
        if d == nd - 1:
            return x, None
        perm = [k for k in range(nd) if k != d] + [d]
        inv = [0] * nd
        for newp, oldp in enumerate(perm):
            inv[oldp] = newp
        return (x.permute(*perm) if hasattr(x, "permute") else x.transpose(perm)), inv
    def _resize_last(x, n):
        if n is None:
            return x
        L = x.shape[-1]
        if L == n:
            return x
        if L > n:
            return x[..., :n]
        pad = jt.zeros(list(x.shape[:-1]) + [n - L], x.dtype)
        return jt.concat([x, pad], dim=-1)
    def _mk_cplx(re, im):
        # build a NATIVE complex64 var from real/imag float vars (Phase 6: fft emits native)
        return jt.nn.view_as_complex(jt.stack([re, im], dim=-1))
    def _re_im(x):
        # (real, imag) float vars from a real Var / ComplexNumber / native complex64. A real
        # Var returns imag=None (skips the imag matmuls); .real/.imag handle CN and native.
        if isinstance(x, _CN):
            return x.real, x.imag
        if isinstance(x, Var) and "complex" in str(x.dtype):
            return x.real, x.imag
        return x, None
    def _fft_core(x, n, dim, inverse, norm=None):
        # x: real Var / ComplexNumber / native complex64 -> NATIVE complex64 DFT along `dim`
        x, inv = _to_last(x, dim)
        re0, im0 = _re_im(x)
        re = _resize_last(re0, n)
        im = _resize_last(im0, n) if im0 is not None else None
        N = re.shape[-1]
        Wc, Ws = _dft_mats(N, inverse)              # cos, sin matrices (N,N)
        # out = (re + i*im) @ (Wc + i*Ws)^T ; matmul over last dim == x @ W^T
        out_re = jt.matmul(re, Wc.transpose(1, 0))
        out_im = jt.matmul(re, Ws.transpose(1, 0))
        if im is not None:
            out_re = out_re - jt.matmul(im, Ws.transpose(1, 0))
            out_im = out_im + jt.matmul(im, Wc.transpose(1, 0))
        # norm: backward (default) -> ifft*1/N; forward -> fft*1/N; ortho -> 1/sqrt(N)
        if norm == "ortho":
            scale = 1.0 / (N ** 0.5)
        elif norm == "forward":
            scale = (1.0 / N) if not inverse else 1.0
        else:
            scale = (1.0 / N) if inverse else 1.0
        if scale != 1.0:
            out_re = out_re * scale
            out_im = out_im * scale
        out = _mk_cplx(out_re, out_im)              # NATIVE complex64
        if inv is not None:
            out = out.permute(*inv)
        return out
    _fft_ns = _types.ModuleType("torch.fft")
    _fft_ns.fft = lambda input, n=None, dim=-1, norm=None: _fft_core(input, n, dim, False, norm)
    _fft_ns.ifft = lambda input, n=None, dim=-1, norm=None: _fft_core(input, n, dim, True, norm)
    def _fftn(input, s=None, dim=(-2, -1), norm=None, inverse=False):
        out = input
        dims = list(dim)
        ss = list(s) if s is not None else [None] * len(dims)
        for d, n in zip(dims, ss):                  # apply 1-D fft along each dim
            out = _fft_core(out, n, d, inverse, norm)
        return out
    _fft_ns.fft2 = lambda input, s=None, dim=(-2, -1), norm=None: _fftn(input, s, dim, norm, False)
    _fft_ns.ifft2 = lambda input, s=None, dim=(-2, -1), norm=None: _fftn(input, s, dim, norm, True)
    _fft_ns.fftn = lambda input, s=None, dim=(-2, -1), norm=None: _fftn(input, s, dim, norm, False)
    _fft_ns.ifftn = lambda input, s=None, dim=(-2, -1), norm=None: _fftn(input, s, dim, norm, True)
    def _rfft(input, n=None, dim=-1, norm=None):
        full = _fft_core(input, n, dim, False, norm)  # real input -> hermitian; keep N//2+1
        N = (input.shape[dim] if n is None else n)
        keep = N // 2 + 1
        fr, fi = full.real, full.imag               # full is native complex64
        sl = [slice(None)] * fr.ndim
        sl[dim if dim >= 0 else dim + fr.ndim] = slice(0, keep)
        return _mk_cplx(fr[tuple(sl)], fi[tuple(sl)])
    _fft_ns.rfft = _rfft
    def _irfft(input, n=None, dim=-1, norm=None):
        # reconstruct the hermitian-symmetric full spectrum, inverse, take real part
        d = dim if dim >= 0 else dim + input.real.ndim
        half = input.real.shape[d]
        N = (2 * (half - 1)) if n is None else n
        full = _fft_core(input, None, dim, True)     # approx: ifft of the given half
        # exact irfft needs the mirrored conjugate; rebuild via real DFT for correctness
        re = input.real; im = input.imag
        # mirror: X[N-k] = conj(X[k]) for k=1..N/2-1
        idx_mirror = list(range(half - 2, 0, -1))
        if idx_mirror:
            sl = [slice(None)] * re.ndim
            sl[d] = idx_mirror
            re_full = jt.concat([re, re[tuple(sl)]], dim=d)
            im_full = jt.concat([im, -im[tuple(sl)]], dim=d)
        else:
            re_full, im_full = re, im
        out = _fft_core(_mk_cplx(re_full, im_full), None, dim, True, norm)
        return out.real
    _fft_ns.irfft = _irfft
    # fftshift/ifftshift: roll the zero-frequency component to/from the centre. The old
    # `lambda x: x` no-op was silent-wrong. fftshift rolls each dim by n//2; ifftshift by
    # -(n//2). Works on real Vars and on ComplexNumber (rolls real+imag).
    def _shift_dims(v, dim, inv):
        dims = list(range(v.ndim)) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
        sh = [(-(int(v.shape[d]) // 2) if inv else int(v.shape[d]) // 2) for d in dims]
        return jt.roll(v, sh, dims)
    def _fftshift(x, dim=None):
        if isinstance(x, jt.nn.ComplexNumber):
            return jt.nn.ComplexNumber(_shift_dims(x.real, dim, False), _shift_dims(x.imag, dim, False))
        return _shift_dims(x, dim, False)
    def _ifftshift(x, dim=None):
        if isinstance(x, jt.nn.ComplexNumber):
            return jt.nn.ComplexNumber(_shift_dims(x.real, dim, True), _shift_dims(x.imag, dim, True))
        return _shift_dims(x, dim, True)
    _fft_ns.fftshift = _fftshift
    _fft_ns.ifftshift = _ifftshift
    import numpy as _np_fft
    _fft_ns.fftfreq = lambda n, d=1.0, **k: jt.array(_np_fft.fft.fftfreq(n, d).astype("float32"))
    _fft_ns.rfftfreq = lambda n, d=1.0, **k: jt.array(_np_fft.fft.rfftfreq(n, d).astype("float32"))
    _alias("fft", _fft_ns)
    import sys as _sys_fft
    _sys_fft.modules["torch.fft"] = _fft_ns
    # torch.softmax / log_softmax / relu top-level function forms (convbert calls
    # torch.softmax(x, dim=...)). jittor exposes these via nn, not the top level.
    _alias("softmax", lambda input, dim=None, **k: jt.nn.softmax(input, dim=dim))
    _alias("log_softmax", lambda input, dim=None, **k: jt.nn.log_softmax(input, dim=dim))
    _alias("relu", lambda input, **k: jt.nn.relu(input))
    # elementwise / functional top-level forms missing from jittor's top level
    _alias("log1p", lambda x: jt.log(1.0 + x))
    _alias("reciprocal", lambda x: 1.0 / x)
    _alias("lerp", lambda input, end, weight: input + weight * (end - input))
    def _isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **k):
        out = jt.abs(a - b) <= (atol + rtol * jt.abs(b))
        if equal_nan:
            out = out | (jt.isnan(a) & jt.isnan(b))
        return out
    _alias("isclose", _isclose)
    def _allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **k):
        return bool(_isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan).all().item())
    _alias("allclose", _allclose)
    _alias("cosine_similarity", lambda x1, x2, dim=1, eps=1e-8: nn.cosine_similarity(x1, x2, dim=dim, eps=eps))
    _alias("pairwise_distance", lambda x1, x2, p=2.0, eps=1e-6, keepdim=False:
           nn.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim))
    # torch.take_along_dim(input, indices, dim): like gather, but torch BROADCASTS
    # indices against input on every dim except `dim` first. transformers' beam search
    # _gather_beams passes indices of shape (batch, k, 1) to gather full sequences of
    # shape (batch, beams, seq_len) along dim=1 -> expects (batch, k, seq_len). A plain
    # jt.gather returns the index's shape (batch, k, 1), collapsing seq_len -> beam
    # search crashed on the next `seq[:, :, cur_len] = ...` setitem. Broadcast first.
    def _take_along_dim(input, indices, dim=None):
        if dim is None:
            return jt.gather(input.reshape(-1), 0, indices.reshape(-1))
        nd = input.ndim
        d = dim % nd
        target = list(input.shape)
        target[d] = indices.shape[d]            # keep index extent along the gather dim
        if list(indices.shape) != target:
            indices = jt.broadcast(indices, target)   # broadcast size-1 dims to input
        return jt.gather(input, d, indices)
    _alias("take_along_dim", _take_along_dim)
    # torch.all/any accept numpy-style axis=/keepdims= aliases (transformers' beam
    # search _update_finished_beams: torch.all(x, axis=-1, keepdims=True)). jittor's
    # native all/any take only `dim` and have no keepdims. Wrap to accept both spellings
    # (dim/axis, keepdim/keepdims) while staying backward-compatible with all(x)/all(x,d).
    def _reduce_alias(orig):
        def f(input, dim=None, keepdim=False, *, axis=None, keepdims=None, out=None):
            d = axis if axis is not None else dim
            kd = keepdims if keepdims is not None else keepdim
            if d is None or d == ():
                return orig(input)
            r = orig(input, d)
            if kd:
                dims = (d,) if isinstance(d, int) else tuple(d)
                nd = input.ndim
                for dd in sorted(x % nd for x in dims):
                    r = r.unsqueeze(dd)
            return r
        return f
    _orig_all = getattr(g, "all", None)
    _orig_any = getattr(g, "any", None)
    if callable(_orig_all):
        g.all = _reduce_alias(_orig_all)
    if callable(_orig_any):
        g.any = _reduce_alias(_orig_any)
    def _movedim(x, source, destination):
        nd = x.ndim
        src = [s % nd for s in (source if isinstance(source, (list, tuple)) else [source])]
        dst = [d % nd for d in (destination if isinstance(destination, (list, tuple)) else [destination])]
        order = [d for d in range(nd) if d not in src]
        for d, s in sorted(zip(dst, src)):
            order.insert(d, s)
        return x.permute(order)
    _alias("movedim", _movedim)
    _alias("moveaxis", _movedim)
    # Var.movedim/moveaxis (the functions exist but weren't bound as methods), plus
    # index_put_/index_put (scatter-style assignment), tensor_split (uneven split), take.
    Var.movedim = lambda self, source, destination: _movedim(self, source, destination)
    Var.moveaxis = lambda self, source, destination: _movedim(self, source, destination)
    def _index_put_(self, indices, values, accumulate=False):
        idx = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
        if not accumulate:
            self[idx if len(idx) > 1 else idx[0]] = values
            return self
        # accumulate=True must add ALL contributions at duplicate indices (a plain
        # read-add-write keeps only the last). Route through index_add (dup-correct).
        vals = values if isinstance(values, Var) else jt.array(values)
        if len(idx) == self.ndim:                          # full advanced index -> linearize
            shape = self.shape
            strides = [1] * self.ndim
            for k in range(self.ndim - 2, -1, -1):
                strides[k] = strides[k + 1] * int(shape[k + 1])
            lin = None
            for k, ind in enumerate(idx):
                term = (ind if isinstance(ind, Var) else jt.array(ind)).int64().reshape((-1,)) * strides[k]
                lin = term if lin is None else lin + term
            vflat = vals.reshape((-1,))
            if int(vflat.shape[0]) == 1 and int(lin.shape[0]) > 1:
                vflat = vflat.broadcast(lin.shape)
            self.assign(self.reshape((-1,)).index_add(0, lin, vflat).reshape(shape))
            return self
        if len(idx) == 1:                                  # index along dim 0
            i0 = (idx[0] if isinstance(idx[0], Var) else jt.array(idx[0])).int64().reshape((-1,))
            self.assign(self.index_add(0, i0, vals))
            return self
        raise NotImplementedError("index_put_(accumulate=True) with a partial multi-dim index")
    Var.index_put_ = _index_put_
    Var.index_put = lambda self, indices, values, accumulate=False: _index_put_(self.clone(), indices, values, accumulate)
    # index_copy_(dim, index, source): self[..,index[i],..] = source[i,..] along dim
    # (overwrite, NOT accumulate -- cf. index_add).
    def _index_copy_(self, dim, index, source):
        d = dim % self.ndim
        idx = index if isinstance(index, Var) else jt.array(index)
        if d == 0:
            self[idx] = source
        else:
            sl = [slice(None)] * self.ndim; sl[d] = idx
            self[tuple(sl)] = source
        return self
    Var.index_copy_ = _index_copy_
    Var.index_copy = lambda self, dim, index, source: _index_copy_(self.clone(), dim, index, source)
    g.index_copy = lambda input, dim, index, source: _index_copy_(input.clone(), dim, index, source)
    g.index_put = lambda input, indices, values, accumulate=False: _index_put_(input.clone(), indices, values, accumulate)
    def _tensor_split(self, indices_or_sections, dim=0):
        d = dim % self.ndim
        L = int(self.shape[d])
        def _slice(a, b):
            ix = [slice(None)] * self.ndim; ix[d] = slice(a, b)
            return self[tuple(ix)]
        if isinstance(indices_or_sections, int):
            n = indices_or_sections
            base, rem = L // n, L % n
            sizes = [base + 1] * rem + [base] * (n - rem)
            out, start = [], 0
            for s in sizes:
                out.append(_slice(start, start + s)); start += s
            return out
        pts, out, prev = list(indices_or_sections), [], 0
        for p in pts + [L]:
            out.append(_slice(prev, p)); prev = p
        return out
    Var.tensor_split = _tensor_split
    g.tensor_split = lambda input, indices_or_sections, dim=0: _tensor_split(input, indices_or_sections, dim)
    Var.take = lambda self, index: self.reshape((-1,))[index]
    g.take = lambda input, index: input.reshape((-1,))[index]
    # torch.eye(n, m=None, *, dtype=, ...): identity / rectangular-identity
    # matrix. jittor has no top-level eye (only jt.init.eye), so add one.
    def _eye(n, m=None, dtype=None, **k):
        # torch.eye(n) is the n x n identity; torch.eye(n, m) is n x m.
        # jittor's init.eye requires a 2-element shape (a bare (n,) asserts),
        # so always pass (n, n) / (n, m).
        shape = (int(n), int(n)) if m is None else (int(n), int(m))
        import jittor.init as _init
        return _init.eye(shape, _dtype_to_str(dtype) or "float32")
    _alias("eye", _eye)
    # torch.narrow(input, dim, start, length) / torch.tile(input, dims) --
    # function forms mirroring the Var methods (added in _install_tensor_methods).
    _alias("narrow", lambda input, dim, start, length: input.narrow(dim, start, length))
    _alias("tile", lambda input, *dims: input.tile(*dims))
    # torch.equal returns a Python bool (True iff same shape & all elements
    # equal). jittor's native `equal` is elementwise, so force-override.
    def _torch_equal(a, b):
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
        except Exception:
            return False
    g.equal = _torch_equal
    Var.equal = lambda self, other: _torch_equal(self, other)
    _alias("diff", lambda x, n=1, dim=-1, prepend=None, append=None:
           _diff(x, n=n, dim=dim, prepend=prepend, append=append))
    _alias("trapz", _trapz)
    _alias("trapezoid", _trapz)
    _alias("repeat_interleave", _repeat_interleave)
    _alias("autocast", lambda *a, **k: _AutocastContext())
    # Real loop-based torch.vmap. The old no-op stub (`lambda fn,*a,**k: fn`)
    # ignored in_dims/out_dims, so transformers' vmap-based causal-mask builder
    # (taken when a model passes and_mask/or_mask -- e.g. falcon) collapsed to a
    # single direct call and produced a wrong all-True (seq,) mask instead of the
    # (b,1,q,kv) causal triangle -> bidirectional attention -> ~79% forward error.
    # Map over in_dims and stack along out_dims. jittor has no 0-d tensors, so a
    # scalar leaf is (1,) where torch has (); collapse that spurious trailing
    # singleton so the stacked rank matches torch.vmap.
    def _vmap(func, in_dims=0, out_dims=0, *_a, **_k):
        def wrapped(*args):
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
            if all(o.ndim >= 1 and o.shape[-1] == 1 for o in outs) and all(o.ndim == outs[0].ndim for o in outs):
                outs = [o.reshape(o.shape[:-1]) if o.ndim > 1 else o for o in outs]
            od = out_dims if isinstance(out_dims, int) else (out_dims[0] if out_dims else 0)
            return jt.stack(outs, dim=od)
        return wrapped
    _alias("vmap", _vmap)
    _alias("outer", lambda a, b: jt.matmul(a.reshape(-1, 1), b.reshape(1, -1)))
    _alias("isin", _isin)
    # torch.cdist(x1,x2,p): pairwise p-distances (...,P,M),(...,R,M)->(...,P,R). Used by
    # contrastive/clustering/retrieval. torch.bucketize: indices to insert into sorted
    # boundaries (samplers / piecewise schedules).
    def _cdist(x1, x2, p=2.0, compute_mode=None, **k):
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)          # (...,P,R,M)
        if p == 2:
            return jt.sqrt((diff * diff).sum(-1))
        if p == 1:
            return jt.abs(diff).sum(-1)
        return (jt.abs(diff) ** p).sum(-1) ** (1.0 / p)
    _alias("cdist", _cdist)
    def _bucketize(input, boundaries, out_int32=False, right=False, **k):
        b = boundaries.reshape((-1,))
        cmp = (input.unsqueeze(-1) >= b) if right else (input.unsqueeze(-1) > b)
        r = cmp.int32().sum(-1)
        return r if out_int32 else r.int64()
    _alias("bucketize", _bucketize)
    # trace / diag_embed / diagflat / kron / logcumsumexp / tensordot / pdist.
    def _trace(input):
        k = min(int(input.shape[0]), int(input.shape[1]))
        ar = jt.arange(k)
        return input[ar, ar].sum()
    _alias("trace", _trace); Var.trace = _trace
    def _diag_embed(input, offset=0, dim1=-2, dim2=-1):
        N = int(input.shape[-1])
        return input.unsqueeze(-1) * jt.init.eye(N)
    _alias("diag_embed", _diag_embed); Var.diag_embed = lambda self, offset=0, dim1=-2, dim2=-1: _diag_embed(self)
    _alias("diagflat", lambda input, offset=0: _diag_embed(input.reshape((-1,))))
    def _kron(a, b):
        nd = max(a.ndim, b.ndim)
        a2 = a.reshape((1,) * (nd - a.ndim) + tuple(a.shape))
        b2 = b.reshape((1,) * (nd - b.ndim) + tuple(b.shape))
        aex, bex, fin = [], [], []
        for i in range(nd):
            aex += [int(a2.shape[i]), 1]; bex += [1, int(b2.shape[i])]
            fin.append(int(a2.shape[i]) * int(b2.shape[i]))
        return (a2.reshape(aex) * b2.reshape(bex)).reshape(fin)
    _alias("kron", _kron); Var.kron = _kron
    def _logcumsumexp(input, dim):
        m = input.max(dim, keepdims=True)
        return m + jt.log(jt.cumsum(jt.exp(input - m), dim))
    _alias("logcumsumexp", _logcumsumexp); Var.logcumsumexp = _logcumsumexp
    def _tensordot(a, b, dims=2):
        if isinstance(dims, int):
            adims, bdims = list(range(a.ndim - dims, a.ndim)), list(range(dims))
        else:
            adims, bdims = list(dims[0]), list(dims[1])
        a_free = [i for i in range(a.ndim) if i not in adims]
        b_free = [i for i in range(b.ndim) if i not in bdims]
        import numpy as _np_td
        af = int(_np_td.prod([int(a.shape[i]) for i in a_free])) if a_free else 1
        cs = int(_np_td.prod([int(a.shape[i]) for i in adims])) if adims else 1
        bf = int(_np_td.prod([int(b.shape[i]) for i in b_free])) if b_free else 1
        out = jt.matmul(a.permute(a_free + adims).reshape((af, cs)), b.permute(bdims + b_free).reshape((cs, bf)))
        fin = [int(a.shape[i]) for i in a_free] + [int(b.shape[i]) for i in b_free]
        return out.reshape(fin) if fin else out.reshape((1,))   # full contraction -> scalar (jittor (1,))
    _alias("tensordot", _tensordot)
    def _pdist(input, p=2.0):
        N = int(input.shape[0])
        diff = input.unsqueeze(1) - input.unsqueeze(0)
        d = ((jt.abs(diff) ** p).sum(-1)) ** (1.0 / p)
        ii = [i for i in range(N) for j in range(i + 1, N)]
        jj = [j for i in range(N) for j in range(i + 1, N)]
        return d[jt.array(ii), jt.array(jj)]
    _alias("pdist", _pdist)
    # shape ops: unflatten / swapaxes / swapdims / ravel + numpy-style stacking helpers.
    def _unflatten(input, dim, sizes):
        d = dim % input.ndim
        return input.reshape(list(input.shape[:d]) + list(sizes) + list(input.shape[d + 1:]))
    _alias("unflatten", _unflatten); Var.unflatten = _unflatten
    def _swapaxes(input, axis0, axis1):
        perm = list(range(input.ndim))
        a, b = axis0 % input.ndim, axis1 % input.ndim
        perm[a], perm[b] = perm[b], perm[a]
        return input.permute(perm)
    _alias("swapaxes", _swapaxes); _alias("swapdims", _swapaxes)
    Var.swapaxes = _swapaxes; Var.swapdims = _swapaxes
    _alias("ravel", lambda input: input.reshape((-1,))); Var.ravel = lambda self: self.reshape((-1,))
    def _vstack(tensors):
        return jt.concat([t if t.ndim >= 2 else t.reshape((1, -1)) for t in tensors], dim=0)
    _alias("vstack", _vstack); _alias("row_stack", _vstack)
    _alias("hstack", lambda tensors: jt.concat(list(tensors), dim=0) if all(t.ndim == 1 for t in tensors)
           else jt.concat(list(tensors), dim=1))
    def _dstack(tensors):
        out = []
        for t in tensors:
            out.append(t.reshape((1, -1, 1)) if t.ndim == 1 else (t.unsqueeze(-1) if t.ndim == 2 else t))
        return jt.concat(out, dim=2)
    _alias("dstack", _dstack)
    _alias("column_stack", lambda tensors: jt.concat([t.reshape((-1, 1)) if t.ndim == 1 else t for t in tensors], dim=1))
    # element-wise ops: copysign / xlogy / heaviside / float_power / signbit.
    def _copysign(input, other):
        s = (other >= 0).float32() * 2 - 1                 # +1 where other>=0 (incl +0), -1 else
        return jt.abs(input) * s
    _alias("copysign", _copysign); Var.copysign = _copysign
    def _xlogy(input, other):
        return jt.ternary(input == 0, jt.zeros_like(input), input * jt.log(other))  # xlogy(0,y)=0
    _alias("xlogy", _xlogy); Var.xlogy = _xlogy
    def _heaviside(input, values):
        return (input > 0).float32() + (input == 0).float32() * values
    _alias("heaviside", _heaviside); Var.heaviside = _heaviside
    def _float_power(input, exponent):
        b = exponent.float64() if isinstance(exponent, Var) else exponent
        return (input.float64() ** b)
    _alias("float_power", _float_power); Var.float_power = _float_power
    _alias("signbit", lambda input: input < 0); Var.signbit = lambda self: self < 0
    # reductions: logsumexp (attention/MoE/loss/beam), nansum/nanmean, std_mean/var_mean,
    # aminmax, quantile. NaN handling uses nan_to_num + (x==x) mask to avoid jittor's
    # isnan+ternary JIT segfault (see jittor-jit-inf-nan-segfault).
    def _logsumexp(input, dim, keepdim=False):
        m = input.max(dim, keepdims=True)
        out = m + jt.log(jt.exp(input - m).sum(dim, keepdims=True))
        if keepdim:
            return out
        # torch removes the reduced dim(s) entirely (1D -> 0-dim scalar). jittor's
        # squeeze keeps a trailing (1,) for the last remaining dim, so reshape to
        # the explicit reduced shape instead.
        dims = [dim] if isinstance(dim, int) else list(dim)
        nd = input.ndim
        dims = [d % nd for d in dims]
        target = [s for i, s in enumerate(input.shape) if i not in dims]
        # jittor has no 0-dim tensors; a full reduction stays (1,).
        return out.reshape(target) if target else out.reshape(-1)
    _alias("logsumexp", _logsumexp); Var.logsumexp = _logsumexp
    def _nansum(input, dim=None, keepdim=False, **k):
        z = jt.nan_to_num(input, nan=0.0)
        return z.sum() if dim is None else z.sum(dim, keepdims=keepdim)
    _alias("nansum", _nansum); Var.nansum = _nansum
    def _nanmean(input, dim=None, keepdim=False, **k):
        # count of non-NaN. NB: `input == input` (a var vs ITSELF) gets optimized to
        # all-True by jittor, so it does NOT detect NaN -- use isnan instead.
        cnt = 1.0 - jt.isnan(input).float32()
        z = jt.nan_to_num(input, nan=0.0)
        if dim is None:
            return z.sum() / cnt.sum()
        return z.sum(dim, keepdims=keepdim) / cnt.sum(dim, keepdims=keepdim)
    _alias("nanmean", _nanmean); Var.nanmean = _nanmean
    def _std_mean(input, dim=None, unbiased=True, keepdim=False, correction=None, **k):
        mean = input.mean() if dim is None else input.mean(dim, keepdims=keepdim)
        std = input.std() if dim is None else input.std(dim)  # jittor std is unbiased
        return (std, mean)
    _alias("std_mean", _std_mean)
    def _var_mean(input, dim=None, unbiased=True, keepdim=False, correction=None, **k):
        s, m = _std_mean(input, dim, unbiased, keepdim)
        return (s * s, m)
    _alias("var_mean", _var_mean)
    _AMinMax = _collections.namedtuple("aminmax", ["min", "max"])
    def _aminmax(input, dim=None, keepdim=False):
        if dim is None:
            return _AMinMax(input.min(), input.max())
        return _AMinMax(input.min(dim, keepdims=keepdim), input.max(dim, keepdims=keepdim))
    _alias("aminmax", _aminmax); Var.aminmax = _aminmax
    def _quantile(input, q, dim=None, keepdim=False, interpolation="linear", **k):
        import numpy as _np_q
        arr = input.numpy()
        qn = q.numpy() if isinstance(q, Var) else q
        r = _np_q.quantile(arr, qn, axis=dim, keepdims=keepdim)
        return jt.array(r.astype("float32"))
    _alias("quantile", _quantile)
    def _nanquantile(input, q, dim=None, keepdim=False, interpolation="linear", **k):
        import numpy as _np_q
        arr = input.numpy()
        qn = q.numpy() if isinstance(q, Var) else q
        r = _np_q.nanquantile(arr, qn, axis=dim, keepdims=keepdim)
        return jt.array(r.astype("float32"))
    _alias("nanquantile", _nanquantile)
    _alias("square", lambda x: x * x)   # torch.square (jittor only had jt.sqr); persimmon
    # torch.addmm(input, mat1, mat2, *, beta=1, alpha=1):
    #   out = beta * input + alpha * (mat1 @ mat2)   (gpt2 uses this for its
    #   Conv1D linear). jittor has no top-level addmm, so add one.
    def _addmm(input, mat1, mat2, *, beta=1, alpha=1):
        res = jt.matmul(mat1, mat2)
        if alpha != 1:
            res = res * alpha
        if beta == 0:
            return res
        return beta * input + res
    _alias("addmm", _addmm)

    # ---- torch.* ops used by mmdetection (additive aliases) ----
    _alias("mm", lambda input, mat2, out=None: jt.matmul(input, mat2))   # 2-D matmul
    def _mv(input, vec, out=None):
        if input.ndim != 2 or vec.ndim != 1:
            raise RuntimeError(
                f"mv: expected a 2-D matrix and a 1-D vector, got "
                f"{input.ndim}-D and {vec.ndim}-D tensors")
        if input.shape[1] != vec.shape[0]:
            raise RuntimeError(
                f"mv: size mismatch, matrix has {input.shape[1]} columns but "
                f"vector has {vec.shape[0]} elements")
        result = jt.matmul(input, vec)
        if out is not None:
            out.assign(result)
            return out
        return result
    _alias("mv", _mv)
    _alias("masked_select", lambda input, mask, out=None: input[mask])   # -> 1-D selected
    _alias("split_with_sizes",
           lambda input, split_sizes, dim=0: input.split(split_sizes, dim))
    _alias("_shape_as_tensor",
           lambda input: jt.array(np.asarray(input.shape, dtype=np.int64)))
    def _nan_to_num_inplace(input, nan=0.0, posinf=None, neginf=None):
        r = g.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)
        try:
            input.assign(r); return input          # honour in-place semantics
        except Exception:
            return r
    _alias("nan_to_num_", _nan_to_num_inplace)
    # torch.randint_like(input, low, high=None, *, dtype=...): jittor's native lacks
    # the dtype kwarg (DINO's denoising uses it). Force-override with torch semantics.
    def _randint_like(input, low, high=None, dtype=None, device=None,
                      requires_grad=False, **kw):
        if high is None:
            low, high = 0, low
        r = jt.randint(int(low), int(high), tuple(int(s) for s in input.shape))
        return r.cast(_dtype_to_str(dtype)) if dtype is not None else r
    g.randint_like = _randint_like

    # torch.sparse_coo_tensor + torch.sparse.sum: mmdet's free_anchor head builds a
    # (hybrid) COO tensor then immediately densifies it. Back it with a dense Var
    # materialised eagerly via index_add_ (COO accumulates duplicate coordinates).
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
    def _sparse_coo_tensor(indices, values, size=None, dtype=None, device=None,
                           requires_grad=False, **kw):
        if not isinstance(indices, jt.Var): indices = jt.array(indices)
        if not isinstance(values, jt.Var): values = jt.array(values)
        S = int(indices.shape[0])
        nnz = int(indices.shape[1]) if indices.ndim == 2 else int(indices.shape[0])
        tail = [int(d) for d in values.shape[1:]]
        idx_np = indices.numpy().astype("int64").reshape(S, -1)
        if size is not None:
            full = [int(s) for s in size]
        else:
            full = [int(idx_np[s].max()) + 1 if nnz > 0 else 0 for s in range(S)] + tail
        sparse_shape = full[:S]; tail2 = full[S:]
        prod = 1
        for d in sparse_shape: prod *= int(d)
        lin = np.zeros(nnz, dtype="int64"); stride = 1     # row-major linear index
        for s in range(S - 1, -1, -1):
            lin = lin + idx_np[s] * stride
            stride *= int(sparse_shape[s])
        flat = jt.zeros([prod] + tail2, dtype=str(values.dtype))
        if nnz > 0:
            flat.index_add_(0, jt.array(lin), values.reshape([nnz] + tail2))  # in-place
        return _SparseCOO(flat.reshape(sparse_shape + tail2))
    _alias("sparse_coo_tensor", _sparse_coo_tensor)
    import jittor.sparse as _jt_sparse
    if not hasattr(_jt_sparse, "sum"):
        def _sparse_sum(x, dim=None):
            d = x._dense if isinstance(x, _SparseCOO) else x
            return _SparseCOO(d.sum(dim) if dim is not None else d.sum())
        _jt_sparse.sum = _sparse_sum

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
        if len(shape) == 1 and isinstance(shape[0], dtype):
            return _bitcast(self, shape[0])
        if len(shape) == 1 and isinstance(shape[0], tuple) and type(shape[0]) is not tuple:
            shape = (tuple(int(s) for s in shape[0]),)
        return _orig_reshape(self, *shape)
    Var.reshape = _torch_reshape
    Var.view = _torch_reshape

    # jittor's CUDA codegen can't emit atomicAdd for uint8/int8 reductions
    # (yolox/rtmdet SimOTA assigners do mask.sum() on a uint8 match matrix);
    # torch promotes integer sums to int64 anyway. Cast narrow ints to int32.
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
    if not hasattr(Var, "mv"):          Var.mv = lambda self, vec: _mv(self, vec)
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
    # det/inverse on (batched) square matrices (mmrotate GWD/KLD/KFIoU Gaussian losses)
    def _vdet(self):
        import jittor.linalg as _la; return _la.det(self)
    def _vinv(self):
        import jittor.linalg as _la; return _la.inv(self)
    if not hasattr(Var, "det"):       Var.det = _vdet
    if not hasattr(Var, "inverse"):   Var.inverse = _vinv
    g.det = lambda x: _vdet(x)
    g.inverse = lambda x: _vinv(x)

    # ---- linalg (peft / lora init need svd_lowrank, svd) ----
    def _svd(x, some=True, compute_uv=True, **kw):
        import jittor.linalg as _la
        u, s, v = _la.svd(x)
        return _MinMax(u, s) if False else (u, s, v)
    def _svd_lowrank(A, q=6, niter=2, M=None):
        # torch.svd_lowrank returns (U, S, V) of a rank-q approximation.
        import jittor.linalg as _la
        if M is not None:
            A = A - M
        u, s, v = _la.svd(A)
        q = min(q, s.shape[0])
        return u[:, :q], s[:q], v[:, :q]
    _alias("svd", _svd)
    _alias("svd_lowrank", _svd_lowrank)
    _alias("pca_lowrank", lambda A, q=6, center=True, niter=2: _svd_lowrank(
        A - (A.mean(0, keepdims=True) if center else 0), q, niter))
