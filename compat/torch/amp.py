"""Automatic mixed precision: ``torch.amp`` and its legacy device namespaces.

The whole family -- ``autocast``, the per-device autocast state torch keeps
behind ``is_autocast_enabled``/``set_autocast_dtype``, ``GradScaler``, and the
``custom_fwd``/``custom_bwd`` decorators for custom autograd Functions -- has
one owner, so ``torch.autocast``, ``torch.amp.autocast``,
``torch.amp.autocast_mode.autocast`` and ``torch.cuda.amp.autocast`` all resolve
to objects defined here rather than to four independent stubs.  It was split out
of :mod:`jittor.compat.torch.grad` when the real implementations landed; that
module still re-exports every name.
"""

from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import collections.abc as _collections_abc
import functools as _functools

import numpy as np
import jittor as jt

from .fidelity import Fidelity, register_fidelity
from .grad_scaler import (          # noqa: F401
    GradScaler, OptState, _CpuGradScaler, _CudaGradScaler, _GradScaler,
)


# Jittor's auto-mixed-precision control registers (src/type/nano_string.h).
# amp_reg drives dtype inference for every op created while it is set.
_AMP_PREFER32 = 1
_AMP_PREFER16 = 2
_AMP_KEEP_REDUCE = 4
_AMP_KEEP_WHITE = 8
_AMP_ARRAY_PREFER = 16

# ---------------------------------------------------------------------------
# torch.amp state.
#
# torch keeps a thread-local, *per device type* record -- an enabled flag and a
# fast dtype for each device, one weight-cache flag and one nesting counter --
# which ``torch.set_autocast_enabled``/``torch.set_autocast_dtype`` mutate
# directly and which ``torch.autocast`` only saves and restores.  This layer
# used to keep a stack of entered regions instead, and the difference was
# visible: ``torch.get_autocast_dtype("cuda")`` answered float32 outside a
# region where torch answers float16 (transformers reads it to pick a cast
# dtype *before* entering one), and ``torch.set_autocast_enabled`` was a
# registered no-op.  The record below is torch's shape.
# ---------------------------------------------------------------------------
import threading as _threading


class _AutocastState(_threading.local):
    def __init__(self):
        self.enabled = {}        # device type -> bool
        self.dtype = {}          # device type -> jittor dtype name
        self.order = []          # device types in the order they were enabled
        self.cache_enabled = True
        self.nesting = 0
        # jt.flags.amp_reg as it was before autocast first took it over, so the
        # last region out puts the process register back exactly.
        self.baseline_reg = None


_autocast_state = _AutocastState()

# The device types torch's own device parser accepts, in its order: its error
# message quotes this list, and a string outside it is not a device at all.
_TORCH_DEVICE_TYPES = (
    "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip",
    "ve", "fpga", "maia", "xla", "lazy", "vulkan", "mps", "meta", "hpu",
    "mtia", "privateuseone",
)
# Ascend is spelled "npu" by jittor and by torch_npu; upstream torch routes it
# through "privateuseone", so it is not in the list above but must be accepted.
_JITTOR_DEVICE_TYPES = ("npu",)

# torch's per-device default fast dtype: bfloat16 on the CPU, float16 elsewhere.
_AUTOCAST_DEFAULT_DTYPE = {"cpu": "bfloat16"}

# The dtypes this layer can express as an amp register setting.
_AUTOCAST_DTYPES = ("float16", "half", "bfloat16",
                    "float32", "float", "double", "float64")


def _autocast_device_type(device_type):
    """Normalise and validate a torch ``device_type`` argument, as torch does."""
    if not isinstance(device_type, str):
        raise ValueError("Expected `device_type` of type `str`, got: `%s`"
                         % type(device_type))
    if not device_type:
        raise RuntimeError("Device string must not be empty")
    base = device_type.split(":")[0]
    if base not in _TORCH_DEVICE_TYPES and base not in _JITTOR_DEVICE_TYPES:
        raise RuntimeError(
            "Expected one of %s device type at start of device string: %s"
            % (", ".join(_TORCH_DEVICE_TYPES), device_type))
    return base


def _accelerator_device_type():
    return "npu" if getattr(jt.compiler, "has_acl", 0) else "cuda"


def is_autocast_available(device_type):
    """True when this build can actually run an autocast region on that device.

    torch answers from the dispatch keys it was *compiled* with, so it also says
    True for xpu/mps/xla/ipu/mtia/maia.  Jittor has no backend for any of them,
    and answering True there would let ``torch.autocast("xla")`` set jittor's
    amp register and quietly lower the precision of whatever backend happened to
    be current -- the silent-no-op shape this layer refuses.  ``cpu`` and
    ``cuda`` are always True: the register is a dtype-inference bias rather than
    a device feature, so a CUDA autocast region is expressible on a CPU-only
    build exactly as torch's is.
    """
    base = _autocast_device_type(device_type)
    if base in ("cpu", "cuda"):
        return True
    if base == "npu":
        return bool(getattr(jt.compiler, "has_acl", 0))
    return False


def _autocast_dtype_name(value):
    """A torch dtype object, a jittor dtype or a name -> the jittor dtype name."""
    if value is None:
        return None
    name = getattr(value, "__name__", None) or _jittor_dtype_name(value)
    return str(name).split(".")[-1]


def _default_autocast_dtype(device_type):
    return _AUTOCAST_DEFAULT_DTYPE.get(device_type, "float16")


def autocast_configured_dtype(device_type=None):
    """The dtype an autocast region on that device computes in.

    ``torch.get_autocast_dtype``: it answers whether or not a region is open.
    """
    base = "cuda" if device_type is None else _autocast_device_type(device_type)
    return _autocast_state.dtype.get(base) or _default_autocast_dtype(base)


def autocast_is_enabled(device_type=None):
    """True inside an *enabled* torch.autocast region (torch.is_autocast_enabled).

    ``device_type=None`` means "cuda", which is what torch's no-argument form
    answers.
    """
    base = "cuda" if device_type is None else _autocast_device_type(device_type)
    return bool(_autocast_state.enabled.get(base, False))


def autocast_dtype(device_type=None):
    """dtype of the innermost enabled autocast region for that device, else None.

    The dispatcher in :mod:`jittor.compat.torch.library` reads this to decide
    whether to push an ``AutocastCPU``/``AutocastCUDA`` key, so a device that is
    *not* in an enabled region has to answer None rather than its default dtype.
    """
    if device_type is None:
        for base in reversed(_autocast_state.order):
            if _autocast_state.enabled.get(base):
                return autocast_configured_dtype(base)
        return None
    base = _autocast_device_type(device_type)
    if not _autocast_state.enabled.get(base):
        return None
    return autocast_configured_dtype(base)


def _amp_register_for(dtype_name):
    if dtype_name in ("float32", "float", "double", "float64"):
        return _AMP_PREFER32
    if dtype_name == "bfloat16":
        from ..stub_policy import degraded
        degraded(
            "torch.autocast(dtype=torch.bfloat16)",
            "jittor's amp register keeps bfloat16 only when an operand "
            "already is bfloat16; an all-float32 region computes in "
            "float16 instead",
            "Cast the module with .to(torch.bfloat16) to stay in bfloat16.")
    return _AMP_PREFER16


def _running_device_type():
    """The device type jittor is currently executing on."""
    code = int(getattr(jt.flags, "use_cuda", 0) or 0)
    if code == 0:
        return "cpu"
    return _accelerator_device_type()


def _effective_autocast_device():
    """Which device's autocast setting jittor's single amp register follows.

    torch has one dispatch key per device, so a CPU region and a CUDA region are
    independent.  Jittor has one process-wide register, so a choice has to be
    made: the backend currently executing wins, and otherwise the most recently
    enabled device does -- which is what keeps ``torch.autocast("cuda")`` on a
    CPU-only build doing what the script asked for.
    """
    running = _running_device_type()
    if _autocast_state.enabled.get(running):
        return running
    for base in reversed(_autocast_state.order):
        if _autocast_state.enabled.get(base):
            return base
    return None


def _refresh_amp_register():
    """Drive ``jt.flags.amp_reg`` from the autocast state, and restore it after."""
    base = _effective_autocast_device()
    if base is None:
        if _autocast_state.baseline_reg is not None:
            jt.flags.amp_reg = _autocast_state.baseline_reg
            _autocast_state.baseline_reg = None
        return
    if _autocast_state.baseline_reg is None:
        _autocast_state.baseline_reg = int(getattr(jt.flags, "amp_reg", 0))
    jt.flags.amp_reg = _amp_register_for(autocast_configured_dtype(base))


def set_autocast_enabled_state(device_type, enabled):
    """Turn autocast on or off for one device type (torch.set_autocast_enabled)."""
    base = _autocast_device_type(device_type)
    enabled = bool(enabled)
    _autocast_state.enabled[base] = enabled
    order = _autocast_state.order
    if base in order:
        order.remove(base)
    if enabled:
        order.append(base)
    _refresh_amp_register()
    return None


def set_autocast_dtype_state(device_type, dtype):
    """Set the fast dtype for one device type (torch.set_autocast_dtype)."""
    base = _autocast_device_type(device_type)
    name = _autocast_dtype_name(dtype)
    if name is None:
        raise ValueError("set_autocast_dtype() requires a dtype, got None")
    if name not in _AUTOCAST_DTYPES:
        from ..stub_policy import unimplemented
        unimplemented(
            "torch.set_autocast_dtype(%r, %s)" % (base, name),
            "record an autocast dtype jittor cannot express and silently "
            "keep computing in the tensors' original dtype",
            "Use torch.float16, torch.bfloat16 or torch.float32.")
        return None
    _autocast_state.dtype[base] = name
    _refresh_amp_register()
    return None


def autocast_cache_enabled():
    """torch.is_autocast_cache_enabled."""
    return bool(_autocast_state.cache_enabled)


def set_autocast_cache_enabled_state(enabled):
    """torch.set_autocast_cache_enabled.

    The flag round-trips, which is the whole observable contract, but it governs
    a cache jittor does not have: torch caches the low-precision cast of a
    weight for the duration of a region, while jittor casts an operand per
    operator.  Turning it off is therefore exactly honoured; turning it on asks
    for an optimisation that is absent, and the numbers are the same either way.
    """
    _autocast_state.cache_enabled = bool(enabled)
    return None


def clear_autocast_cache():
    """torch.clear_autocast_cache.

    Torch drops the cached low-precision weight casts here.  Jittor keeps no
    such cache, so the postcondition -- no cached cast survives this call --
    already holds.  That is a complete implementation of an empty cache, not a
    stub standing in for a missing one.
    """
    return None


def autocast_increment_nesting():
    """torch.autocast_increment_nesting: the new depth."""
    _autocast_state.nesting += 1
    return _autocast_state.nesting


def autocast_decrement_nesting():
    """torch.autocast_decrement_nesting: the new depth."""
    _autocast_state.nesting = max(0, _autocast_state.nesting - 1)
    return _autocast_state.nesting


def autocast_nesting():
    return _autocast_state.nesting


class autocast:
    """torch.autocast, implemented on jittor's amp registers.

    It used to be a total no-op: every argument was accepted, nothing changed,
    and a script that asked for mixed precision silently trained in float32 with
    none of the promised memory or speed -- while ``is_autocast_enabled()``
    agreed it was off, so nothing in the program could notice.

    The region now sets ``jt.flags.amp_reg`` so op dtype inference actually
    prefers the low-precision type (matmul/conv in fp16, ``exp``/``pow`` and
    reductions kept in fp32, mirroring torch's autocast lists), and restores the
    previous register on exit.  It is still BOTH a context manager and a
    decorator -- accelerate does ``new_forward = autocast(model_forward)``.

    Known differences from torch, each on the record in the fidelity registry:

    * jittor's register selects *bfloat16* only when an operand already is
      bfloat16, so an all-float32 model under ``autocast(dtype=torch.bfloat16)``
      computes in float16 -- same mantissa or better, narrower exponent range.
      A one-time warning says so.
    * torch's autocast lists are per operator; jittor's register is a global
      dtype-inference bias, so the fall-through category (``add`` on two float32
      operands, which torch leaves in float32) also lowers here.
    * one process-wide register serves every device type, so a CPU region and a
      CUDA region are not independent the way torch's dispatch keys are.
    """

    def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        base = _autocast_device_type(device_type)
        if not is_autocast_available(base):
            raise RuntimeError(
                "User specified an unsupported autocast device_type '%s'"
                % device_type)
        self.device = base
        self._enabled = bool(enabled)
        name = (autocast_configured_dtype(base) if dtype is None
                else _autocast_dtype_name(dtype))
        self.fast_dtype = name
        self._cache_enabled = (autocast_cache_enabled() if cache_enabled is None
                               else bool(cache_enabled))
        # A list, not a single slot: the same instance is legitimately re-entered
        # (``with scaler_ctx:`` inside a recursive module), and one slot would
        # make the inner exit restore the outer entry's state.
        self._saved = []
        if self._enabled and name not in _AUTOCAST_DTYPES:
            from ..stub_policy import unimplemented
            unimplemented(
                "torch.autocast(dtype=%s)" % name,
                "accept an autocast dtype jittor cannot express and silently "
                "keep computing in the tensors' original dtype",
                "Use torch.float16, torch.bfloat16 or torch.float32.")

    # Names the pre-2.0 context object exposed; kept so nothing that reached
    # into the instance breaks on the rename to torch's own attribute names.
    @property
    def device_type(self):
        return self.device

    @property
    def enabled(self):
        return self._enabled

    @property
    def cache_enabled(self):
        return self._cache_enabled

    def __enter__(self):
        self._saved.append((autocast_is_enabled(self.device),
                            autocast_configured_dtype(self.device),
                            autocast_cache_enabled()))
        set_autocast_dtype_state(self.device, self.fast_dtype)
        set_autocast_enabled_state(self.device, self._enabled)
        autocast_increment_nesting()
        set_autocast_cache_enabled_state(self._cache_enabled)
        return self

    def __exit__(self, *exc):
        if not self._saved:
            return False
        prev_enabled, prev_dtype, prev_cache = self._saved.pop()
        if autocast_decrement_nesting() == 0:
            clear_autocast_cache()
        set_autocast_dtype_state(self.device, prev_dtype)
        set_autocast_enabled_state(self.device, prev_enabled)
        set_autocast_cache_enabled_state(prev_cache)
        return False

    def _fresh(self):
        """An equivalent region that shares no entry state with this one."""
        return autocast(self.device, dtype=self.fast_dtype,
                        enabled=self._enabled,
                        cache_enabled=self._cache_enabled)

    def __call__(self, func):
        return autocast_decorator(self, func)


def autocast_decorator(autocast_instance, func):
    """torch.amp.autocast_mode.autocast_decorator.

    Torch re-enters the *same* instance on every call; this enters a fresh copy,
    which is the same region and is safe when the decorated function is called
    from more than one thread.
    """
    @_functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance._fresh():
            return func(*args, **kwargs)
    decorate_autocast.__script_unsupported = (
        "@autocast() decorator is not supported in script mode")
    return decorate_autocast


class _DeviceAutocast(autocast):
    """The legacy per-device spelling: ``torch.cuda.amp.autocast(enabled=...)``.

    torch keeps these as subclasses of :class:`autocast` with the device baked
    in and the old argument order, and ``isinstance(ctx, torch.amp.autocast)``
    in library code depends on the subclass relationship.
    """

    _device = "cuda"
    _device_default_dtype = "float16"

    def __init__(self, enabled=True, dtype=None, cache_enabled=True):
        super().__init__(
            self._device,
            dtype=self._device_default_dtype if dtype is None else dtype,
            enabled=enabled, cache_enabled=cache_enabled)

    def _fresh(self):
        return type(self)(enabled=self._enabled, dtype=self.fast_dtype,
                          cache_enabled=self._cache_enabled)


class _CudaAutocast(_DeviceAutocast):
    _device = "cuda"
    _device_default_dtype = "float16"


class _CpuAutocast(_DeviceAutocast):
    _device = "cpu"
    _device_default_dtype = "bfloat16"


def _var_device_type(value):
    from .types import _var_is_cpu_resident
    return "cpu" if _var_is_cpu_resident(value) else _accelerator_device_type()


def _amp_cast(value, device_type, dtype):
    """torch.amp.autocast_mode._cast: cast eligible floating tensors, recursively.

    Only floating tensors that live on ``device_type`` are cast, and float64 is
    left alone -- the same eligibility rule torch applies, so a custom op that
    asks for float32 inputs does not quietly downcast a double.
    """
    name = _autocast_dtype_name(dtype)
    if isinstance(value, jt.Var):
        current = _jittor_dtype_name(value.dtype)
        eligible = (current.startswith("float")
                    and current != "float64"
                    and _var_device_type(value) == device_type)
        return value.cast(name) if eligible and current != name else value
    if isinstance(value, (str, bytes, np.ndarray)):
        return value
    if isinstance(value, _collections_abc.Mapping):
        return {_amp_cast(k, device_type, dtype): _amp_cast(v, device_type, dtype)
                for k, v in value.items()}
    if isinstance(value, _collections_abc.Iterable):
        items = (_amp_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(items)
        return items
    return value


def custom_fwd(fwd=None, *, device_type, cast_inputs=None):
    """torch.amp.custom_fwd for a custom autograd Function's ``forward``.

    This was ``lambda f: f`` -- a decorator that accepted ``cast_inputs=`` and
    then did not cast, so an op written to require float32 inputs received
    whatever the surrounding autocast region produced and computed its
    reduction in float16.  It now does what torch does: record the region's
    dtype on the context, and when ``cast_inputs`` is given inside an enabled
    region, cast the incoming floating tensors and run ``forward`` with autocast
    off.
    """
    if not isinstance(device_type, str):
        raise ValueError("Expected `device_type` of type `str`, got: `%s`"
                         % type(device_type))
    if fwd is None:
        return _functools.partial(custom_fwd, device_type=device_type,
                                  cast_inputs=cast_inputs)

    @_functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = autocast_configured_dtype(device_type)
        if cast_inputs is None:
            args[0]._fwd_used_autocast = autocast_is_enabled(device_type)
            return fwd(*args, **kwargs)
        in_autocast = autocast_is_enabled(device_type)
        args[0]._fwd_used_autocast = False
        if not in_autocast:
            return fwd(*args, **kwargs)
        with autocast(device_type, enabled=False):
            return fwd(*_amp_cast(args, device_type, cast_inputs),
                       **_amp_cast(kwargs, device_type, cast_inputs))

    return decorate_fwd


def custom_bwd(bwd=None, *, device_type):
    """torch.amp.custom_bwd: run ``backward`` in the region ``forward`` ran in."""
    if not isinstance(device_type, str):
        raise ValueError("Expected `device_type` of type `str`, got: `%s`"
                         % type(device_type))
    if bwd is None:
        return _functools.partial(custom_bwd, device_type=device_type)

    @_functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(device_type,
                      enabled=getattr(args[0], "_fwd_used_autocast", False),
                      dtype=getattr(args[0], "_dtype", None)):
            return bwd(*args, **kwargs)

    return decorate_bwd


def cuda_custom_fwd(fwd=None, *, cast_inputs=None):
    """torch.cuda.amp.custom_fwd: custom_fwd with device_type pinned to cuda."""
    return custom_fwd(fwd, device_type="cuda", cast_inputs=cast_inputs)


def cuda_custom_bwd(bwd=None):
    """torch.cuda.amp.custom_bwd: custom_bwd with device_type pinned to cuda."""
    return custom_bwd(bwd, device_type="cuda")


def amp_definitely_not_available():
    """torch.cuda.amp.common.amp_definitely_not_available."""
    return not bool(getattr(jt, "has_cuda", False))


#: Historical private name: ``jittor.torch_compat._AutocastContext`` is pickled
#: by checked-in payloads and re-exported from :mod:`jittor.compat.torch`, so
#: the rename keeps both spellings alive.
_AutocastContext = autocast


_AMP_FIDELITY = (
    (("torch.autocast", "torch.amp.autocast", "torch.amp.autocast_mode.autocast"),
     autocast, Fidelity.APPROXIMATE,
     "Sets jittor's global amp register instead of torch's per-operator "
     "autocast lists, so which operators lower is the register's rule and not "
     "torch's: matmul, linear and convolution lower and reductions stay in "
     "float32 as torch does, but the fall-through category torch leaves alone "
     "(float32 add) lowers here too. One register serves every device type, so "
     "a CPU region and a CUDA region are not independent the way torch's "
     "dispatch keys are, and bfloat16 is selected only when an operand already "
     "is bfloat16 (an all-float32 region computes in float16, warned once). "
     "cache_enabled is recorded; there is no weight cast cache to enable"),
    (("torch.cuda.amp.autocast",), _CudaAutocast, Fidelity.APPROXIMATE,
     "Legacy cuda-pinned subclass of torch.autocast with the same register "
     "behaviour and limitations"),
    (("torch.cpu.amp.autocast",), _CpuAutocast, Fidelity.APPROXIMATE,
     "Legacy cpu-pinned subclass of torch.autocast; jittor's register keeps "
     "bfloat16 only when an operand already is bfloat16, so the CPU default "
     "dtype computes in float16"),
    (("torch.amp.custom_fwd", "torch.amp.autocast_mode.custom_fwd"),
     custom_fwd, Fidelity.APPROXIMATE,
     "Records the region dtype on the context and, with cast_inputs inside an "
     "enabled region, casts eligible floating inputs and runs forward with "
     "autocast off; eligibility uses the Var's jittor residency rather than a "
     "torch device object"),
    (("torch.amp.custom_bwd", "torch.amp.autocast_mode.custom_bwd"),
     custom_bwd, Fidelity.APPROXIMATE,
     "Re-enters the region forward ran in, from the dtype and enabled flag "
     "custom_fwd recorded on the context"),
    (("torch.cuda.amp.custom_fwd",), cuda_custom_fwd, Fidelity.APPROXIMATE,
     "custom_fwd with device_type pinned to cuda"),
    (("torch.cuda.amp.custom_bwd",), cuda_custom_bwd, Fidelity.APPROXIMATE,
     "custom_bwd with device_type pinned to cuda"),
    (("torch.amp.is_autocast_available",),
     is_autocast_available, Fidelity.APPROXIMATE,
     "Answers for the backends this build can execute on -- cpu and cuda "
     "always, npu when ACL is built -- so it is False for the xpu/mps/xla/ipu/"
     "mtia/maia device types torch answers True for, and torch.autocast then "
     "refuses them with torch's own message instead of silently biasing the "
     "register of whatever backend is current"),
    (("torch.cuda.amp.common.amp_definitely_not_available",),
     amp_definitely_not_available, Fidelity.APPROXIMATE,
     "Reports whether this build has a CUDA accelerator; torch also counts a "
     "torch_xla installation, which does not apply here"),
    (("torch.amp.autocast_mode.autocast_decorator",),
     autocast_decorator, Fidelity.APPROXIMATE,
     "Enters a fresh copy of the region per call instead of re-entering the "
     "same instance, so a decorated function stays correct across threads"),
)

for _apis, _implementation, _level, _detail in _AMP_FIDELITY:
    for _api in _apis:
        register_fidelity(_api, _implementation, _level, _detail)
del _apis, _implementation, _level, _detail, _api
