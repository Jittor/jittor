"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt

from ..functional import (
    _torch_norm_impl,
    _torch_where_select,
)
from ..grad import (
    _GradScaler,
)
from ..nested import (
    _torch_make_parameter, _torch_prune_leaf_registry,
)
from ..types import (
    _dtype_to_str,
    _make_dtypes, device, dtype,
)

def install(ctx):
    g = ctx.jittor_module
    ctx.registry.publish("torch", g)
    g.torch = g
    ctx.registry.publish("torch.torch", g)
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
    ctx.state["dtypes"] = _DTYPE_OBJS
    ctx.state["Var"] = jt.Var


def install_misc(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    import types as _types_misc
    _types2 = _types_misc

    if "torch.storage" not in _modules:
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
        _modules["torch.storage"] = _storage_mod
    else:
        _storage_mod = _modules["torch.storage"]

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
    _modules["torch.random"] = _random_mod
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
            native_nonzero = getattr(jt, "_vj_native_nonzero", None)
            if native_where is not None:
                idx = native_where(condition)
            elif native_nonzero is not None:
                idx = native_nonzero(condition)
            else:
                idx = condition.nonzero()
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
