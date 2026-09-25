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
from ..grad_scaler import _GradScaler
from ..amp import (
    autocast_cache_enabled as _autocast_cache_enabled,
    autocast_configured_dtype as _autocast_configured_dtype,
    autocast_decrement_nesting,
    autocast_increment_nesting,
    autocast_is_enabled as _autocast_is_enabled,
    autocast_dtype as _autocast_dtype,
    clear_autocast_cache,
    is_autocast_available,
    set_autocast_cache_enabled_state as _set_autocast_cache_enabled,
    set_autocast_dtype_state as _set_autocast_dtype,
    set_autocast_enabled_state as _set_autocast_enabled,
)
from ..types import (
    _dtype_to_str,
    _make_dtypes, device, dtype, make_torch_types_module,
    SymBool, SymFloat, SymInt,
)
from ..core_install_api import bind_core_install_api
from ..fidelity import Fidelity, register_fidelity
from ...diagnostics import EXPECTED, swallowed
from ...transaction import set_flag


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


def _torch_dtype_object(name):
    """The torch dtype object for a jittor dtype name, e.g. "float16"."""
    return getattr(_misc_context().jittor_module, name, name)


def _get_autocast_dtype(device_type, *a, **k):
    """torch.get_autocast_dtype: the *configured* fast dtype for that device.

    It used to return the dtype only while a region was open and float32
    otherwise, which is not what torch answers -- torch keeps a per-device
    setting that defaults to float16 (bfloat16 on the CPU) and that
    ``set_autocast_dtype`` mutates, and transformers reads it *before*
    entering a region to decide what to cast weights to.
    """
    return _torch_dtype_object(_autocast_configured_dtype(device_type))


def set_autocast_dtype(device_type, dtype=None, *a, **k):
    """torch.set_autocast_dtype: set the fast dtype for one device type."""
    return _set_autocast_dtype(device_type, dtype)


def get_autocast_cpu_dtype(*a, **k):
    """Deprecated torch spelling of get_autocast_dtype("cpu")."""
    return _torch_dtype_object(_autocast_configured_dtype("cpu"))


def set_autocast_cpu_dtype(dtype, *a, **k):
    """Deprecated torch spelling of set_autocast_dtype("cpu", dtype)."""
    return _set_autocast_dtype("cpu", dtype)


def set_autocast_gpu_dtype(dtype, *a, **k):
    """Deprecated torch spelling of set_autocast_dtype("cuda", dtype)."""
    return _set_autocast_dtype("cuda", dtype)


def is_autocast_cpu_enabled(*a, **k):
    """Deprecated torch spelling of is_autocast_enabled("cpu")."""
    return _autocast_is_enabled("cpu")


def set_autocast_cpu_enabled(enabled, *a, **k):
    """Deprecated torch spelling of set_autocast_enabled("cpu", enabled)."""
    return _set_autocast_enabled("cpu", enabled)


def is_autocast_cache_enabled(*a, **k):
    """torch.is_autocast_cache_enabled."""
    return _autocast_cache_enabled()


def set_autocast_cache_enabled(enabled, *a, **k):
    """torch.set_autocast_cache_enabled."""
    return _set_autocast_cache_enabled(enabled)


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
    lengths_list = [int(length) for length in lengths]
    tail = list(data.shape[1:])
    segs = []
    start = 0
    for length in lengths_list:
        chunk = data[start : start + length]
        start += length
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
        # A range query computes nothing: resolve the name without demanding
        # compute support, or the float8/float4 entries in _FINFO_SPECIAL are
        # unreachable and `torch.finfo(torch.float8_e4m3fn)` raises.
        ds = _dtype_to_str(dt, require_compute=False) or "float32"
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
        # Same as finfo: an integer range query computes nothing.
        ds = _dtype_to_str(dt, require_compute=False) or "int64"
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
    """torch's ``result_type`` for two operands.

    torch ranks operands in three tiers -- tensors with dimensions, 0-dim
    tensors, Python scalars (c10 ``ResultTypeState``: dimResult, zeroResult,
    wrappedResult). A weaker operand joins promotion only when its category
    is higher than the stronger one's; within a tier the pair promotes as
    usual. So ``half_tensor * torch.tensor(2.0)`` is half, like
    ``half_tensor * 2.0``. The 0-dim tier used to count as a full tensor:
    diffusers' schedulers multiply float16 latents by 0-dim float32 entries of
    ``alphas_cumprod``, which turned every sampling step's latents float32.
    """
    ctx = _misc_context()
    _DTYPE_OBJS = ctx.state["dtypes"]
    (na, la), (nb, lb) = (_result_type_info(a), _result_type_info(b))
    if la > lb:
        res = _promote_pair(na, nb) if _category(na) > _category(nb) else nb
    elif lb > la:
        res = _promote_pair(na, nb) if _category(nb) > _category(na) else na
    else:
        res = _promote_pair(na, nb)
    return _DTYPE_OBJS.get(res, res)


def can_cast(from_dtype, to_dtype):
    """torch's ``canCast``: three refusals by category, not by width.

    This is not numpy's "safe cast" rule. ``c10/core/ScalarType.h`` refuses
    exactly complex -> non-complex, floating -> integral, and non-bool -> bool;
    everything else is allowed, so ``can_cast(int64, int32)`` is True in torch
    and was False here, and so was every other narrowing pair. Verified
    against torch 2.13 over the whole 10x10 table.
    """
    f, t = (_dtype_to_str(from_dtype), _dtype_to_str(to_dtype))
    source, target = _category(f), _category(t)
    if source == 3 and target != 3:
        return False
    if source == 2 and target == 1:
        return False
    return not (source != 0 and target == 0)


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
    """torch.get_default_device: CPU until `set_default_device` says otherwise.

    Used to report cuda whenever `jt.flags.use_cuda` was on, which conflates
    "the accelerator is enabled" with "the accelerator is the default device".
    torch keeps those apart: CUDA being available never moves the default off
    the CPU. See `compat/torch/frontend.py::default_device`.
    """
    ctx = _misc_context()
    g = ctx.jittor_module
    from ..frontend import default_device as _recorded_default
    spelling = _recorded_default()
    if str(spelling).split(":")[0] == "cpu":
        return g.device("cpu")
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


def get_device_module(device=None):
    """torch.get_device_module: the module that implements a device's runtime.

    No argument means "the current accelerator", as in torch; under the facade
    that is CUDA whenever jittor's ``use_cuda`` flag is on. MiniMax-H3's video
    VAE stores the result and drives its ``device()`` scope and ``manual_seed``
    through it, so a missing name aborted engine construction.
    """
    ctx = _misc_context()
    g = ctx.jittor_module
    resolved = get_default_device() if device is None else device
    name = getattr(resolved, "type", None) or str(resolved).split(":")[0]
    if name in ("cuda", "gpu"):
        return g.cuda
    if name == "npu" and hasattr(g, "npu"):
        return g.npu
    if name == "cpu":
        return g.cpu
    raise NotImplementedError(
        "torch.get_device_module(%r): unsupported device type %r" % (device, name))


def as_strided(input, size, stride, storage_offset=None):
    """torch.as_strided -- a tensor with the requested size and strides.

    vLLM-Omni's CPU offload rebuilds a possibly strided parameter from its
    packed host buffer through this entry point. ``Tensor.as_strided``
    materializes the window with a gather, so reads are exact; the result does
    not alias ``input`` the way a real strided view does.

    The default offset is 0, not ``input.storage_offset()`` as in torch: jittor
    materializes slices, so ``input``'s own data already starts at its first
    element. Torch's default would apply the parent-relative storage offset a
    second time -- the offload passes ``gpu_weight[offset:offset+numel]`` and
    then indexed past the end of it ("index 10751 is out of bounds for
    dimension 0 with size 5376").
    """
    return input.as_strided(size, stride, 0 if storage_offset is None else storage_offset)


def empty_strided(size, stride, *, dtype=None, layout=None, device=None,
                  requires_grad=False, pin_memory=False):
    """torch.empty_strided -- a tensor with the requested size and strides.

    jittor tensors are contiguous, so `stride` cannot be honored: the result
    has `size` with contiguous strides. vLLM-Omni's offload calls this only to
    get an independent buffer and then `copy_`s into it, so the values are
    exact. A non-strided `layout` is refused rather than silently ignored.
    """
    if layout is not None and "strided" not in str(layout):
        raise NotImplementedError(
            "torch.empty_strided(layout=%r): only the strided layout exists"
            % (layout,))
    g = _misc_context().target_namespace
    return g.empty(tuple(size), dtype=dtype, device=device,
                   requires_grad=requires_grad, pin_memory=pin_memory)


def _restore_default_device_index(ctx):
    """Put back the current device an indexed default device took over."""
    state = ctx.state["core_misc"]
    saved = state.pop("default_device_saved_index", None)
    if saved is None or saved < 0:
        return
    try:
        if int(jt.current_device()) != saved:
            jt.set_device(saved)
    except EXPECTED as exc:
        swallowed("torch/installers/core.py _restore_default_device_index: "
                  "jt.set_device(%r)" % (saved,), exc,
                  "the default device's index stays current after it is cleared")


def _record_default_device(spelling):
    """Tell the tensor factories where a `device=`-less tensor belongs."""
    from ..frontend import set_default_device_spelling
    set_default_device_spelling(spelling)


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
        _restore_default_device_index(ctx)
        _record_default_device(None)
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
        _restore_default_device_index(ctx)
        _record_default_device('cpu')
        return None
    if name in ("cuda", "gpu", "npu"):
        if not jt.has_cuda:
            raise RuntimeError(
                "torch.set_default_device(%r): this build has no CUDA/NPU device available."
                % (device,)
            )
        _set_install_flag(ctx, "use_cuda", 1)
        if index is not None:
            # Remember what was current *before* the first indexed default, so
            # clearing the default puts it back. Without this, the index
            # leaked: `set_default_device("cuda:4")` followed by
            # `set_default_device(None)` left jittor's current device at 4, so
            # the next tensor built after CUDA came back on -- through
            # `.cuda()`, say -- landed on cuda:4 while `get_default_device()`
            # had already said "cpu". torch's default device is a separate
            # thing from `torch.cuda.current_device()` and clearing one never
            # strands the other.
            state = ctx.state["core_misc"]
            if state.get("default_device_saved_index") is None:
                try:
                    state["default_device_saved_index"] = int(jt.current_device())
                except EXPECTED as exc:
                    swallowed("torch/installers/core.py set_default_device: "
                              "state['default_device_saved_index']", exc,
                              "clearing the default will not restore the current device")
            try:
                jt.set_device(int(index))
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise RuntimeError("torch.set_default_device(%r): %s" % (device, error))
        _record_default_device(name if index is None else "%s:%d" % (name, index))
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
    # (dtype name, tier): 0 a tensor with dimensions (or a bare dtype),
    # 1 a 0-dim tensor, 2 a Python scalar. See `result_type`.
    # Any Var, not only the frontend's Tensor type: the binary operators pass
    # native Vars through here too, and one that fell to the fallback below
    # lost its dtype.
    if isinstance(x, (Var, jt.Var)):
        return (_dtype_to_str(x.dtype), 1 if len(x.shape) == 0 else 0)
    if isinstance(x, dtype) or (
        isinstance(x, str) and _dtype_to_str(x) in _jittor_dtype_name(_DTYPE_OBJS)
    ):
        return (_dtype_to_str(x), 0)
    if isinstance(x, bool):
        return ("bool", 2)
    if isinstance(x, int):
        return ("int64", 2)
    if isinstance(x, float):
        return (_dtype_to_str(g.get_default_dtype()) or "float32", 2)
    if isinstance(x, complex):
        return ("complex64", 2)
    return (_dtype_to_str(x) or "float32", 0)


def initial_seed():
    return int(_misc_context().state["core_misc"].get("seed", 0))


def is_tensor(value):
    return isinstance(value, _misc_context().state["Var"])


def numel(value):
    return value.numel()


def set_autocast_enabled(device_type, enabled=None, *a, **k):
    """torch.set_autocast_enabled: turn autocast on or off for one device.

    This was a registered no-op, so a script that opened its mixed-precision
    region with the setter instead of the context manager trained in float32
    while ``is_autocast_enabled()`` agreed with it. It now moves the same
    per-device state ``torch.autocast`` moves. The pre-2.4 one-argument form
    (``set_autocast_enabled(True)``) still means "cuda", as it does in torch.
    """
    if enabled is None and not isinstance(device_type, str):
        device_type, enabled = "cuda", device_type
    return _set_autocast_enabled(device_type, enabled)


def is_grad_enabled():
    return not bool(getattr(jt.flags, "no_grad", 0))


def set_grad_enabled(mode):
    owner = _misc_context().jittor_module
    return owner.enable_grad() if mode else owner.no_grad()


def get_autocast_gpu_dtype(*args, **kwargs):
    """Deprecated torch spelling of get_autocast_dtype("cuda")."""
    return _torch_dtype_object(_autocast_configured_dtype("cuda"))


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
class _DefaultGenerator:
    """`torch.default_generator`: a handle on the *global* CPU generator.

    Deliberately not a `Generator` instance. That class owns a private stream
    so that two generators seeded alike agree whatever the process has already
    done -- which is exactly what the default generator must *not* do, because
    `torch.manual_seed(n)` seeds this one and `torch.get_rng_state()` is its
    state. So this delegates and holds nothing; anything else would let the two
    drift apart.

    Missing entirely before, and the shim's namespace reports a missing name by
    raising `AttributeError(name)`, so MiniMax-H3's reference path failed with
    a bare `default_generator` and nothing to say where it came from.
    """

    @property
    def device(self):
        return _torch_device_misc("cpu")

    def manual_seed(self, value):
        manual_seed(value)
        return self

    def initial_seed(self):
        return initial_seed()

    def seed(self):
        return seed()

    def get_state(self):
        return get_rng_state()

    def set_state(self, state):
        set_rng_state(state)
        return self

    def __repr__(self):
        return "<torch.Generator object (default, device=cpu)>"


def _torch_device_misc(spelling):
    """A device object, taken from the install context, not the module registry.

    `installers/` must not reach into the interpreter's module table -- that is
    the boundary `test_torch_compat_structure` defends, and spelling the lookup
    through an alias to slip past its substring check would be gaming it rather
    than honouring it.
    """
    return _misc_context().jittor_module.device(spelling)


def fork_rng(devices=None, enabled=True, _caller="fork_rng",
             _devices_kw="devices", device_type="cuda"):
    """torch.random.fork_rng: run a block, then put the RNG back.

    A context manager, not a function -- callers write
    `with torch.random.fork_rng(devices=[0]):`. MiniMax-H3's reference-to-video
    path uses it around its sampling, and without it the request died with
    `module 'torch.random' has no attribute 'fork_rng'`.

    `devices=None` means every visible device of `device_type`, which is what
    torch does; passing an explicit list is cheaper and is what callers that
    care do. `enabled=False` makes the whole thing a no-op, again as torch
    does, so a caller can keep one code path for both.
    """
    return _ForkRng(devices, enabled, device_type)


class _ForkRng:
    """The context manager behind :func:`fork_rng`.

    Written as a class rather than `@contextlib.contextmanager` so that the
    state is captured on `__enter__`, not when the generator object is made.
    `with fork_rng():` and `cm = fork_rng(); with cm:` then behave the same,
    which a generator-based one would not.
    """

    def __init__(self, devices, enabled, device_type):
        self._devices = devices
        self._enabled = bool(enabled)
        self._device_type = device_type
        self._cpu_state = None
        self._device_states = ()
        self._targets = ()

    def _accelerator(self):
        if self._device_type != "cuda":
            return None
        cuda = getattr(_misc_context().jittor_module, "cuda", None)
        if cuda is None or not getattr(cuda, "is_available", lambda: False)():
            return None
        return cuda

    def __enter__(self):
        if not self._enabled:
            return self
        self._cpu_state = get_rng_state()
        cuda = self._accelerator()
        if cuda is not None:
            devices = self._devices
            if devices is None:
                devices = range(int(cuda.device_count()))
            # Passed through as given, not coerced with `int()`. Callers hand
            # this whatever torch accepts -- MiniMax-H3's VAE passes
            # `[torch.device('cuda:0')]` -- and `int()` on a device object
            # raises "int() argument must be ... not 'device'". The accessors
            # below already take an index, a device or a string.
            self._targets = tuple(devices)
            self._device_states = tuple(
                cuda.get_rng_state(device) for device in self._targets)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self._enabled:
            return False
        # Restore on the way out of a failure too: a block that raised has
        # still consumed randomness, and leaving the stream advanced would make
        # the next draw depend on whether an unrelated error happened.
        set_rng_state(self._cpu_state)
        cuda = self._accelerator()
        if cuda is not None:
            for device, state in zip(self._targets, self._device_states):
                cuda.set_rng_state(state, device)
        return False


_MISC_BINDINGS = {
    "manual_seed": manual_seed,
    "initial_seed": initial_seed,
    "seed": seed,
    "get_rng_state": get_rng_state,
    "set_rng_state": set_rng_state,
    "fork_rng": fork_rng,
    "default_generator": _DefaultGenerator(),
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
    "set_autocast_dtype": set_autocast_dtype,
    "get_autocast_gpu_dtype": get_autocast_gpu_dtype,
    "set_autocast_gpu_dtype": set_autocast_gpu_dtype,
    "get_autocast_cpu_dtype": get_autocast_cpu_dtype,
    "set_autocast_cpu_dtype": set_autocast_cpu_dtype,
    "is_autocast_cpu_enabled": is_autocast_cpu_enabled,
    "set_autocast_cpu_enabled": set_autocast_cpu_enabled,
    "is_autocast_cache_enabled": is_autocast_cache_enabled,
    "set_autocast_cache_enabled": set_autocast_cache_enabled,
    "clear_autocast_cache": clear_autocast_cache,
    "autocast_increment_nesting": autocast_increment_nesting,
    "autocast_decrement_nesting": autocast_decrement_nesting,
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
    "get_device_module": get_device_module,
    "as_strided": as_strided,
    "empty_strided": empty_strided,
}
_MISC_DETAILS = {
    "PyTorchFileReader": "raises NotImplementedError; use torch.load instead",
    "is_autocast_enabled": "per-device-type autocast flag; the no-argument form answers for cuda as torch's does",
    "set_autocast_enabled": "moves the same per-device autocast state torch.autocast moves; one amp register serves every device type",
    "get_autocast_dtype": "the configured per-device fast dtype, answered whether or not a region is open",
    "set_autocast_dtype": "records the per-device fast dtype; a dtype jittor cannot express is refused rather than ignored",
    "get_autocast_gpu_dtype": "deprecated spelling of get_autocast_dtype('cuda')",
    "set_autocast_gpu_dtype": "deprecated spelling of set_autocast_dtype('cuda', dtype)",
    "get_autocast_cpu_dtype": "deprecated spelling of get_autocast_dtype('cpu')",
    "set_autocast_cpu_dtype": "deprecated spelling of set_autocast_dtype('cpu', dtype)",
    "is_autocast_cpu_enabled": "deprecated spelling of is_autocast_enabled('cpu')",
    "set_autocast_cpu_enabled": "deprecated spelling of set_autocast_enabled('cpu', enabled)",
    "is_autocast_cache_enabled": "the recorded preference; jittor casts an operand per operator and has no weight cast cache",
    "set_autocast_cache_enabled": "records the preference; there is no weight cast cache to enable, so disabling it is exact and enabling it asks for an absent optimisation",
    "clear_autocast_cache": "jittor keeps no cached weight casts, so the postcondition already holds",
    "autocast_increment_nesting": "real thread-local nesting depth, as torch's counter",
    "autocast_decrement_nesting": "real thread-local nesting depth, as torch's counter",
    "use_deterministic_algorithms": "no-op setter; deterministic algorithm policy is not implemented",
    "get_rng_state": "seed-only state, not a full generator snapshot or exact stream restoration",
    "set_rng_state": "restores the recorded seed, not an exact generator stream snapshot",
    "norm": "existing Torch norm adapter; out and extra keyword semantics are not implemented",
    "where": "existing one- or three-argument selection; out is not implemented",
    "bincount": "native scatter-add implementation; existing flatten/minlength behavior retained",
    "segment_reduce": "lengths-based dim-0 reduction only; additional keyword semantics are not implemented",
    "finfo": "NumPy limits plus declared metadata-only low-precision specs; this does not enable their computation",
    "iinfo": "NumPy integer-limit metadata for supported dtype names",
    "is_autocast_available": "answers for the backends this build can run -- cpu and cuda always, npu when ACL is built -- so it is False for the xpu/mps/xla/ipu/mtia device types torch answers True for",
    "are_deterministic_algorithms_enabled": "legacy False answer; deterministic algorithms are not configurable",
    "as_strided": "gather-based view; reads are exact but the result does not alias the input storage",
    "empty_strided": "contiguous allocation; the requested strides are not honored",
}
for _name, _implementation in _MISC_BINDINGS.items():
    _level = (
        Fidelity.UNIMPLEMENTED
        if _name in ("PyTorchFileReader", "use_deterministic_algorithms")
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
    ("fork_rng", fork_rng),
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
    for name in ("manual_seed", "initial_seed", "get_rng_state",
                 "set_rng_state", "fork_rng"):
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
