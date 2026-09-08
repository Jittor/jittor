"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt
import types as _types_misc
import numpy as _np

from ..functional import (
    _torch_norm_impl,
    _torch_where_select,
)
from ..grad import (
    _GradScaler,
    autocast_is_enabled as _autocast_is_enabled,
    autocast_dtype as _autocast_dtype,
)
from ..nested import (
    _torch_make_parameter, _torch_prune_leaf_registry,
)
from ..types import (
    _dtype_to_str,
    _make_dtypes, device, dtype, make_torch_types_module,
    SymBool, SymFloat, SymInt,
)
from ..core_install_api import bind_core_install_api
from ..fidelity import Fidelity, register_fidelity
from ...diagnostics import EXPECTED, swallowed
from ...transaction import set_flag, set_attr


_LN2 = 0.6931471805599453
_INV_LN10 = 0.4342944819032518
_UNARY_MATH_FIDELITY_DETAIL = (
    "matches Torch elementwise values and preserves the input dtype for "
    "supported real tensors, including sign(NaN) == 0 as Torch 2.x returns; "
    "device, layout, dtype-keyword, and out semantics are not implemented"
)


def sign(input):
    """Return -1/0/+1 elementwise, in the input's own dtype.

    Casting each comparison back to ``input.dtype`` is what keeps
    ``torch.sign(int_tensor)`` integral, the way Torch does it. The float-valued
    ``(x > 0) * 1.0 - (x < 0) * 1.0`` form this replaced returned float32 for
    every input, so the module-level function and ``Tensor.sign`` -- which came
    from a second copy in installers/tensor.py -- disagreed on integer tensors.
    """
    return (input > 0).cast(input.dtype) - (input < 0).cast(input.dtype)


def trunc(input):
    """Round toward zero elementwise."""
    return jt.ternary(input >= 0, jt.floor(input), jt.ceil(input))


def frac(input):
    """Return the fractional part, carrying the sign of ``input``."""
    return input - trunc(input)


def exp2(input):
    """Return ``2 ** input`` elementwise."""
    return jt.exp(input * _LN2)


def log10(input):
    """Return the base-10 logarithm elementwise."""
    return jt.log(input) * _INV_LN10


for _unary_api, _unary_impl in (
    ("torch.sign", sign),
    ("torch.trunc", trunc),
    ("torch.Tensor.frac", frac),
    ("torch.exp2", exp2),
    ("torch.log10", log10),
):
    register_fidelity(
        _unary_api, _unary_impl, Fidelity.APPROXIMATE,
        _UNARY_MATH_FIDELITY_DETAIL)
del _unary_api, _unary_impl


def _set_install_flag(ctx, name, value):
    """Record install-time flag mutations when a transaction is active."""
    set_flag(ctx.native_backend.flags, name, value, context=ctx)


def install(ctx):
    g = ctx.jittor_module
    ctx.registry.publish("torch", g)
    g.torch = g
    ctx.registry.publish("torch.torch", g)
    ctx.registry.publish("torch.types", make_torch_types_module())

    # Escape hatch for the APIs this layer refuses to fake.  See
    # jittor/compat/stub_policy.py; JITTOR_TORCH_ALLOW_STUB=1 does the same.
    from ...stub_policy import (
        registry as _unimplemented_registry,
        approximate_registry as _approximate_registry,
    )

    bind_core_install_api(ctx)
    g.compat_unimplemented_apis = _unimplemented_registry
    g.compat_approximate_apis = _approximate_registry

    # Every failure this layer decided to continue past, on the record. The
    # handlers used to be `except Exception: pass`, so a marker that failed to
    # propagate or a dtype that failed to restore left nothing behind at all.
    # See jittor/compat/diagnostics.py; JITTOR_COMPAT_DEBUG=1 also prints them.
    from ...diagnostics import (
        records as _swallowed_records,
        counts as _swallowed_counts,
        set_debug as _set_compat_debug,
    )
    g.compat_swallowed = _swallowed_records
    g.compat_swallowed_counts = _swallowed_counts
    g.compat_debug = _set_compat_debug

    # Who owns what `torch.__version__` reports.
    #
    # `torch` IS this module, so `torch.__version__ = x` is `jittor.__version__
    # = x`: it changes the framework's own version number for every user in the
    # process. An adapter used to do exactly that (the vLLM one), which is why
    # the boundary rules now forbid a staged adapter from assigning to any
    # attribute of torch/jittor at all. The decision belongs here, in the layer
    # that knows both numbers, and it is explicit, idempotent and reversible.
    #
    # By default `torch.__version__` keeps Jittor's version (that is what this
    # module is) and the torch API level lives at `torch.__torch_version__` /
    # `torch.version.__version__`. A caller that must present the API level --
    # a library that gates features on `torch.__version__` and cannot be told
    # otherwise -- asks for it here.
    if "core_native_api" not in ctx.state:
        from types import MappingProxyType
        ctx.state["core_native_api"] = MappingProxyType({
            name: getattr(g, name, None)
            for name in ("load", "save", "where", "nonzero", "seed")
        })

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
            _set_install_flag(ctx, "use_cuda", 1)
    except EXPECTED as exc:
        swallowed("torch/installers/core.py install: import os as _os", exc)
    _DTYPE_OBJS = _make_dtypes(g)
    g.dtype = dtype
    g.device = device
    g.SymInt = SymInt
    g.SymFloat = SymFloat
    g.SymBool = SymBool
    g.GradScaler = _GradScaler        # picked up by torch.amp/torch.cuda.amp in the shim
    try:
        import jittor.nn as _jt_nn_top
        for _conv_name in ("conv1d", "conv2d", "conv3d",
                           "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
            if not hasattr(g, _conv_name) and hasattr(_jt_nn_top, _conv_name):
                setattr(g, _conv_name, getattr(_jt_nn_top, _conv_name))
    except (AttributeError, TypeError) as exc:
        swallowed("torch/installers/core.py install: import jittor.nn as _jt_nn_top", exc)
    ctx.state["dtypes"] = _DTYPE_OBJS
    from ..frontend import make_tensor_type
    tensor_type = make_tensor_type(ctx.native_backend)
    ctx.state["Var"] = tensor_type
    g.Var = g.Tensor = tensor_type
    g.clone = tensor_type.clone


# Public misc objects have one module owner. Mutable policy and seed
# state belong to the active installation, never a captured installer closure.
_types_random = _types_misc
_seed_sentinel = object()


def _misc_context():
    from ..context import get_install_context

    return get_install_context(jt)


_FINFO_SPECIAL = {
    "bfloat16": (
        -3.3895313892515355e38,
        3.3895313892515355e38,
        0.0078125,
        1.1754943508222875e-38,
        16,
    ),
    "float8_e4m3fn": (-448.0, 448.0, 0.125, 0.015625, 8),
    "float8_e4m3fnuz": (-240.0, 240.0, 0.125, 0.0078125, 8),
    "float8_e5m2": (-57344.0, 57344.0, 0.25, 6.103515625e-05, 8),
    "float8_e5m2fnuz": (-57344.0, 57344.0, 0.25, 6.103515625e-05, 8),
    "float8_e8m0fnu": (-3.4e38, 3.4e38, 1.0, 1e-38, 8),
    "float4_e2m1fn_x2": (-6.0, 6.0, 0.5, 0.5, 4),
}

_PROMO_ORDER = [
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]

_PROMO_ROWS = {
    "bool": [
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
    "uint8": [
        "uint8",
        "uint8",
        "int16",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
    "int8": [
        "int8",
        "int16",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
    "int16": [
        "int16",
        "int16",
        "int16",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
    "int32": [
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
    "int64": [
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
    "float16": [
        "float16",
        "float16",
        "float16",
        "float16",
        "float16",
        "float16",
        "float16",
        "float32",
        "float32",
        "float64",
    ],
    "bfloat16": [
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "float32",
        "bfloat16",
        "float32",
        "float64",
    ],
    "float32": [
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float64",
    ],
    "float64": [
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ],
}

_PROMO_IDX = {n: i for i, n in enumerate(_PROMO_ORDER)}


class UntypedStorage:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def _typed_storage(self):
        return TypedStorage(wrap_storage=self)


class TypedStorage:
    def __init__(
        self, *args, wrap_storage=None, dtype=None, device=None, _internal=False, **kwargs
    ):
        ctx = _misc_context()
        g = ctx.jittor_module
        self._untyped_storage = wrap_storage
        self.dtype = dtype if dtype is not None else getattr(g, "float32", "float32")
        self.device = device
        self.args = args
        self.kwargs = kwargs

    def untyped(self):
        return self._untyped_storage


class _RandomModule(_types_random.ModuleType):
    def __call__(self, *args, **kwargs):
        ctx = _misc_context()
        _native_random_fn = ctx.state["core_misc_native_random"]
        if callable(_native_random_fn):
            return _native_random_fn(*args, **kwargs)
        raise TypeError("torch.random is not callable")


def _manual_seed(s):
    ctx = _misc_context()
    g = ctx.jittor_module
    s = int(s)
    ctx.state["core_misc"]["seed"] = s
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(s)
    return g


def _torch_seed():
    import secrets

    value = secrets.randbits(31)
    _manual_seed(value)
    return value


def _seed(value=_seed_sentinel):
    ctx = _misc_context()
    g = ctx.jittor_module
    if value is _seed_sentinel:
        return _torch_seed()
    value = int(value)
    ctx.state["core_misc"]["seed"] = value
    native_seed = ctx.state["core_native_api"]["seed"]
    if callable(native_seed):
        return native_seed(value)
    return jt.set_seed(value)


def _get_rng_state():
    return jt.array([initial_seed()], dtype="int64")


def _set_rng_state(state):
    ctx = _misc_context()
    Var = ctx.state["Var"]
    try:
        if isinstance(state, Var):
            state = int(state.reshape(-1)[0].item())
        elif hasattr(state, "__len__"):
            state = int(list(state)[0])
        else:
            state = int(state)
    except EXPECTED as exc:
        swallowed("torch/installers/core.py _set_rng_state: if isinstance(state, Var):", exc)
        state = initial_seed()
    _manual_seed(state)


class PyTorchFileReader:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "torch.PyTorchFileReader is not implemented by the jittor torch shim; use torch.load instead"
        )


def norm(input, p="fro", dim=None, keepdim=False, dtype=None, out=None, **kw):
    return _torch_norm_impl(input, p=p, dim=dim, keepdim=keepdim, dtype=dtype)


def _is_autocast_enabled(device_type=None, *a, **k):
    return _autocast_is_enabled(device_type)


def _get_autocast_dtype(device_type=None, *a, **k):
    ctx = _misc_context()
    g = ctx.jittor_module
    name = _autocast_dtype(device_type)
    if name is None:
        return getattr(g, "float32", "float32")
    return getattr(g, name, name)


def where(condition, input=None, other=None, *, out=None):
    if input is None and other is None:
        native = _misc_context().state["core_native_api"]
        native_where = native["where"]
        native_nonzero = native["nonzero"]
        if native_where is not None:
            idx = native_where(condition)
        elif native_nonzero is not None:
            idx = native_nonzero(condition)
        else:
            idx = condition.nonzero()
        if isinstance(idx, (tuple, list)):
            return tuple(idx)
        if getattr(idx, "ndim", 0) == 2:
            return tuple((idx[:, d] for d in range(idx.shape[1])))
        return (idx.reshape(-1),)
    if input is None or other is None:
        raise TypeError("torch.where expected either 1 or 3 arguments")
    return _torch_where_select(condition, input, other)


def bincount(input, weights=None, minlength=0):
    x = input.reshape(-1).int64()
    ml = max(int(minlength), 0)
    if x.numel() == 0:
        wdtype = weights.dtype if weights is not None else jt.int64
        return jt.zeros((ml,), dtype=wdtype)
    n = max(int(x.max().item()) + 1, ml)
    if weights is not None:
        out = jt.zeros((n,), dtype=weights.dtype)
        src = weights.reshape(-1).cast(_jittor_dtype_name(weights.dtype))
    else:
        out = jt.zeros((n,), dtype=jt.int64)
        src = jt.ones((x.shape[0],), dtype=jt.int64)
    return out.scatter_add(0, x, src)


def segment_reduce(data, reduce="sum", *, lengths=None, **kw):
    assert lengths is not None, "torch_compat segment_reduce requires lengths="
    lengths_list = [int(l) for l in lengths]
    tail = list(data.shape[1:])
    segs = []
    start = 0
    for l in lengths_list:
        chunk = data[start : start + l]
        start += l
        if reduce == "sum":
            r = chunk.sum(dim=0)
        elif reduce == "mean":
            r = chunk.mean(dim=0)
        elif reduce == "prod":
            r = chunk.prod(dim=0)
        elif reduce in ("max", "amax"):
            r = chunk.amax(dim=0)
        elif reduce in ("min", "amin"):
            r = chunk.amin(dim=0)
        else:
            raise ValueError(f"Unsupported segment_reduce op: {reduce}")
        segs.append(r.reshape([1] + tail))
    return jt.concat(segs, dim=0)


class finfo:
    def __init__(self, dt):
        ds = _dtype_to_str(dt) or "float32"
        if ds in _FINFO_SPECIAL:
            mn, mx, eps, tiny, bits = _FINFO_SPECIAL[ds]
            self.min, self.max, self.eps, self.tiny, self.smallest_normal = (
                mn,
                mx,
                eps,
                tiny,
                tiny,
            )
            self.bits, self.dtype = (bits, ds)
            self.resolution = eps
            return
        info = _np.finfo(_np.dtype(ds))
        self.min = float(info.min)
        self.max = float(info.max)
        self.eps = float(info.eps)
        self.tiny = float(info.tiny)
        self.smallest_normal = float(info.tiny)
        self.resolution = float(info.resolution)
        self.bits = info.bits
        self.dtype = ds


class iinfo:
    def __init__(self, dt):
        ds = _dtype_to_str(dt) or "int64"
        info = _np.iinfo(_np.dtype(ds))
        self.min = int(info.min)
        self.max = int(info.max)
        self.bits = info.bits


def _promote_pair(a, b):
    if a == b:
        return a
    ia, ib = (_PROMO_IDX.get(a), _PROMO_IDX.get(b))
    if ia is not None and ib is not None:
        return _PROMO_ROWS[a][ib]
    if a.startswith("complex") or b.startswith("complex"):
        wide = "complex128" if "128" in a or "128" in b or "float64" in (a, b) else "complex64"
        return wide
    return a if ib is None else b


def promote_types(t1, t2):
    ctx = _misc_context()
    _DTYPE_OBJS = ctx.state["dtypes"]
    return _DTYPE_OBJS.get(
        _promote_pair(_dtype_to_str(t1), _dtype_to_str(t2)),
        _promote_pair(_dtype_to_str(t1), _dtype_to_str(t2)),
    )


def _category(name):
    if name == "bool":
        return 0
    if name.startswith(("int", "uint")):
        return 1
    if name.startswith("complex"):
        return 3
    return 2


def result_type(a, b):
    ctx = _misc_context()
    _DTYPE_OBJS = ctx.state["dtypes"]
    (na, sa), (nb, sb) = (_result_type_info(a), _result_type_info(b))
    if sa and (not sb):
        res = _promote_pair(na, nb) if _category(na) > _category(nb) else nb
    elif sb and (not sa):
        res = _promote_pair(na, nb) if _category(nb) > _category(na) else na
    else:
        res = _promote_pair(na, nb)
    return _DTYPE_OBJS.get(res, res)


def can_cast(from_dtype, to_dtype):
    f, t = (_dtype_to_str(from_dtype), _dtype_to_str(to_dtype))
    return _promote_pair(f, t) == t


def set_default_dtype(d):
    ctx = _misc_context()
    _state = ctx.state["core_misc"]
    if not isinstance(d, dtype) or _dtype_to_str(d) not in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ):
        raise TypeError("only floating-point types are supported as the default type")
    _state["dtype"] = d


def get_default_device():
    ctx = _misc_context()
    g = ctx.jittor_module
    if not jt.flags.use_cuda:
        return g.device("cpu")
    try:
        index = int(jt.current_device())
    except EXPECTED as exc:
        swallowed(
            "torch/installers/core.py get_default_device: index = int(jt.current_device())",
            exc,
            "reporting cuda:0, which is wrong on any other device",
        )
        index = 0
    return g.device("cuda", index if index >= 0 else 0)


def set_default_device(device=None):
    """torch.set_default_device -- now actually moves the default.

    Was `lambda *a, **k: None` while get_default_device() reported the real
    residency, so set/get openly contradicted each other: a script that set
    the default to "cuda" allocated on the CPU and was told it had not.
    Jittor's default residency is the global use_cuda flag, so this sets it.
    """
    ctx = _misc_context()
    if device is None:
        _set_install_flag(ctx, "use_cuda", 0)
        return None
    if isinstance(device, str):
        name, _, raw_index = device.partition(":")
        index = int(raw_index) if raw_index.isdigit() else None
    else:
        name = getattr(device, "type", None) or str(device)
        index = getattr(device, "index", None)
        if not isinstance(index, int) or isinstance(index, bool):
            index = None
        if index is None and ":" in str(name):
            name, _, raw_index = str(name).partition(":")
            index = int(raw_index) if raw_index.isdigit() else None
    name = str(name).split(":")[0]
    if name == "cpu":
        _set_install_flag(ctx, "use_cuda", 0)
        return None
    if name in ("cuda", "gpu", "npu"):
        if not jt.has_cuda:
            raise RuntimeError(
                "torch.set_default_device(%r): this build has no CUDA/NPU device available."
                % (device,)
            )
        _set_install_flag(ctx, "use_cuda", 1)
        if index is not None:
            try:
                jt.set_device(int(index))
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise RuntimeError("torch.set_default_device(%r): %s" % (device, error))
        return None
    from ...stub_policy import unimplemented

    return unimplemented(
        "torch.set_default_device(%r)" % (name,),
        "silently keep the previous default device",
        "Only 'cpu' and 'cuda' defaults are supported.",
    )


def _result_type_info(x):
    ctx = _misc_context()
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    if isinstance(x, Var):
        return (_dtype_to_str(x.dtype), False)
    if isinstance(x, dtype) or (
        isinstance(x, str) and _dtype_to_str(x) in _jittor_dtype_name(_DTYPE_OBJS)
    ):
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


def initial_seed():
    return int(_misc_context().state["core_misc"].get("seed", 0))


def is_tensor(value):
    return isinstance(value, _misc_context().state["Var"])


def numel(value):
    return value.numel()


def set_autocast_enabled(*args, **kwargs):
    return None


def is_grad_enabled():
    return not bool(getattr(jt.flags, "no_grad", 0))


def set_grad_enabled(mode):
    owner = _misc_context().jittor_module
    return owner.enable_grad() if mode else owner.no_grad()


def get_autocast_gpu_dtype(*args, **kwargs):
    owner = _misc_context().jittor_module
    return (
        _get_autocast_dtype("cuda")
        if _autocast_is_enabled("cuda")
        else getattr(owner, "float16", "float16")
    )


def is_autocast_available(*args, **kwargs):
    return True


def are_deterministic_algorithms_enabled():
    return False


def use_deterministic_algorithms(*args, **kwargs):
    return None


def is_floating_point(value):
    return "float" in _jittor_dtype_name(value.dtype)


def get_default_dtype():
    return _misc_context().state["core_misc"]["dtype"]


manual_seed = _manual_seed
seed = _seed
get_rng_state = _get_rng_state
set_rng_state = _set_rng_state
is_autocast_enabled = _is_autocast_enabled
get_autocast_dtype = _get_autocast_dtype


class DoubleStorage(TypedStorage):
    pass


class FloatStorage(TypedStorage):
    pass


class HalfStorage(TypedStorage):
    pass


class BFloat16Storage(TypedStorage):
    pass


class LongStorage(TypedStorage):
    pass


class IntStorage(TypedStorage):
    pass


class ShortStorage(TypedStorage):
    pass


class CharStorage(TypedStorage):
    pass


class ByteStorage(TypedStorage):
    pass


class BoolStorage(TypedStorage):
    pass


_STORAGE_TYPES = (
    UntypedStorage,
    TypedStorage,
    DoubleStorage,
    FloatStorage,
    HalfStorage,
    BFloat16Storage,
    LongStorage,
    IntStorage,
    ShortStorage,
    CharStorage,
    ByteStorage,
    BoolStorage,
)
_MISC_BINDINGS = {
    "manual_seed": manual_seed,
    "initial_seed": initial_seed,
    "seed": seed,
    "get_rng_state": get_rng_state,
    "set_rng_state": set_rng_state,
    "is_tensor": is_tensor,
    "numel": numel,
    "PyTorchFileReader": PyTorchFileReader,
    "norm": norm,
    "where": where,
    "bincount": bincount,
    "segment_reduce": segment_reduce,
    "is_autocast_enabled": is_autocast_enabled,
    "set_autocast_enabled": set_autocast_enabled,
    "is_grad_enabled": is_grad_enabled,
    "set_grad_enabled": set_grad_enabled,
    "get_autocast_dtype": get_autocast_dtype,
    "get_autocast_gpu_dtype": get_autocast_gpu_dtype,
    "is_autocast_available": is_autocast_available,
    "are_deterministic_algorithms_enabled": are_deterministic_algorithms_enabled,
    "use_deterministic_algorithms": use_deterministic_algorithms,
    "is_floating_point": is_floating_point,
    "finfo": finfo,
    "iinfo": iinfo,
    "promote_types": promote_types,
    "result_type": result_type,
    "can_cast": can_cast,
    "get_default_dtype": get_default_dtype,
    "set_default_dtype": set_default_dtype,
    "get_default_device": get_default_device,
    "set_default_device": set_default_device,
}
_MISC_DETAILS = {
    "PyTorchFileReader": "raises NotImplementedError; use torch.load instead",
    "set_autocast_enabled": "no-op setter; use the supported autocast scope",
    "use_deterministic_algorithms": "no-op setter; deterministic algorithm policy is not implemented",
    "get_rng_state": "seed-only state, not a full generator snapshot or exact stream restoration",
    "set_rng_state": "restores the recorded seed, not an exact generator stream snapshot",
    "norm": "existing Torch norm adapter; out and extra keyword semantics are not implemented",
    "where": "existing one- or three-argument selection; out is not implemented",
    "bincount": "native scatter-add implementation; existing flatten/minlength behavior retained",
    "segment_reduce": "lengths-based dim-0 reduction only; additional keyword semantics are not implemented",
    "finfo": "NumPy limits plus declared metadata-only low-precision specs; this does not enable their computation",
    "iinfo": "NumPy integer-limit metadata for supported dtype names",
    "is_autocast_available": "legacy True capability answer; does not verify a requested device",
    "are_deterministic_algorithms_enabled": "legacy False answer; deterministic algorithms are not configurable",
}
for _name, _implementation in _MISC_BINDINGS.items():
    _level = (
        Fidelity.UNIMPLEMENTED
        if _name in ("PyTorchFileReader", "set_autocast_enabled", "use_deterministic_algorithms")
        else Fidelity.APPROXIMATE
    )
    register_fidelity(
        "torch." + _name,
        _implementation,
        _level,
        _MISC_DETAILS.get(
            _name, "existing Jittor compatibility behavior; not a claim of complete Torch parity"
        ),
    )
for _storage in _STORAGE_TYPES:
    register_fidelity(
        "torch." + _storage.__name__,
        _storage,
        Fidelity.APPROXIMATE,
        "metadata carrier only; no byte-storage allocation or tensor-storage semantics",
    )
    register_fidelity(
        "torch.storage." + _storage.__name__,
        _storage,
        Fidelity.APPROXIMATE,
        "metadata carrier only; no byte-storage allocation or tensor-storage semantics",
    )
for _name, _implementation in (
    ("manual_seed", manual_seed),
    ("initial_seed", initial_seed),
    ("seed", _torch_seed),
    ("get_rng_state", get_rng_state),
    ("set_rng_state", set_rng_state),
):
    register_fidelity(
        "torch.random." + _name,
        _implementation,
        Fidelity.APPROXIMATE,
        _MISC_DETAILS.get(
            _name, "existing Jittor compatibility behavior; not a claim of complete Torch parity"
        ),
    )
register_fidelity(
    "torch.Tensor.bincount", bincount, Fidelity.APPROXIMATE, _MISC_DETAILS["bincount"]
)
del _name, _implementation, _level, _storage


def install_misc(ctx):
    """Bind stable misc objects and initialize their per-installation state."""
    modules = ctx.registry.module_map
    owner = ctx.jittor_module
    var_type = ctx.state["Var"]
    ctx.state.setdefault("core_misc", {"dtype": getattr(owner, "float32", "float32")})
    ctx.state.setdefault("core_misc_native_random", getattr(owner, "random", None))
    storage = modules.get("torch.storage")
    if storage is None:
        storage = modules["torch.storage"] = _types_misc.ModuleType("torch.storage")
    for storage_type in _STORAGE_TYPES:
        setattr(storage, storage_type.__name__, storage_type)
        setattr(owner, storage_type.__name__, storage_type)
    owner.storage = storage
    random = modules.get("torch.random")
    if not isinstance(random, _RandomModule):
        random = modules["torch.random"] = _RandomModule("torch.random")
    for name in ("manual_seed", "initial_seed", "get_rng_state", "set_rng_state"):
        setattr(random, name, _MISC_BINDINGS[name])
    random.seed = _torch_seed
    owner.random = random
    for name, implementation in _MISC_BINDINGS.items():
        setattr(owner, name, implementation)
    var_type.bincount = bincount
    owner.Tensor.bincount = bincount
    for name, implementation in (
        ("exp2", exp2),
        ("log10", log10),
        ("sign", sign),
        ("trunc", trunc),
    ):
        setattr(owner, name, implementation)
        setattr(var_type, name, implementation)
    var_type.frac = frac
