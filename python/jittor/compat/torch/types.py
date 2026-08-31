"""Torch-compatible dtype, device, and residency primitives."""

import os
import types as _python_types
import typing

import jittor as jt

_NATIVE_DTYPE_CONVERTERS = {}


class dtype(str):
    """A torch-like dtype that IS the jittor dtype string.

    Subclasses str so it passes jittor's C++ type-dispatched constructors
    (which require a str/NanoString) unchanged, while printing torch-style and
    carrying is_floating_point like torch.dtype.
    """
    _registry = {}

    def __new__(cls, name, is_floating_point=False):
        obj = super().__new__(cls, name)   # the str value is the bare jittor name
        obj.name = name
        obj._is_fp = is_floating_point
        cls._registry[name] = obj
        return obj

    @property
    def is_floating_point(self):
        return self._is_fp

    @property
    def itemsize(self):
        # bytes per element (torch.dtype.itemsize); used by vLLM weight transfer.
        _sz = {"bool": 1, "uint8": 1, "uint1": 1, "uint2": 1, "uint3": 1, "uint4": 1,
               "uint5": 1, "uint6": 1, "uint7": 1,
               "int8": 1, "float8_e4m3fn": 1,
               "float8_e5m2": 1, "float8_e4m3fnuz": 1, "float8_e5m2fnuz": 1,
               "float8_e8m0fnu": 1, "qint8": 1, "quint8": 1,
               "int16": 2, "uint16": 2, "float16": 2, "bfloat16": 2,
               "int32": 4, "uint32": 4, "float32": 4, "complex32": 4, "qint32": 4,
               "int64": 8, "uint64": 8, "float64": 8, "complex64": 8,
               "complex128": 16}
        return _sz.get(self.name, 4)
    element_size = itemsize

    # NanoString-compatible predicates: jittor internals call x.dtype.is_float()
    # /is_int()/is_bool(). Since we now return this object from Var.dtype, it
    # must answer them too.
    def is_float(self):
        return self._is_fp
    def is_bool(self):
        return self.name == "bool"
    def is_int(self):
        return self.name.startswith(("int", "uint"))
    def is_unsigned(self):
        return self.name.startswith("uint")

    def __repr__(self):
        return "torch." + self.name

    # transformers' save_pretrained recovers the bare dtype name via
    # `str(model.dtype).split(".")[1]`, relying on torch's
    # `str(torch.float32) == "torch.float32"`. We cannot make the underlying str
    # *value* torch-prefixed: jittor's own Python code (contrib.concat,
    # linalg, nn) does `str(var.dtype)` and feeds the result straight back into
    # jittor's C++ dtype dispatch, which only knows the bare names. So instead
    # `__str__` returns the dtype object itself (a str whose value stays bare,
    # which jittor accepts), and only a literal `.split(".")` is special-cased
    # to surface the torch-style ["torch", name]. No jittor dtype name contains
    # a dot, so every other split is the normal str split.
    def __str__(self):
        return self

    def split(self, sep=None, maxsplit=-1):
        if sep == ".":
            return ["torch", self.name]
        return str.split(self, sep, maxsplit)

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other or ("torch." + self.name) == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __call__(self, *args, **kwargs):
        """Preserve Jittor's historical ``jt.float32(value)`` constructors."""
        converter = _NATIVE_DTYPE_CONVERTERS.get(self.name)
        if converter is None:
            raise TypeError("dtype %s has no Jittor tensor constructor" % self.name)
        return converter(*args, **kwargs)


def _make_dtypes(ns):
    specs = [
        ("float32", True), ("float64", True), ("float16", True),
        ("bfloat16", True),
        ("int8", False), ("int16", False), ("int32", False), ("int64", False),
        ("uint8", False), ("uint16", False), ("uint32", False), ("uint64", False),
        ("bool", False),
        # complex types -- jittor has no native complex, but the dtype objects
        # must exist (libraries index size tables by them). Best-effort names.
        ("complex64", False), ("complex128", False), ("complex32", False),
        # quantized dtypes -- no compute support, but tensordict/torch index
        # dtype tables by them so the objects must exist + be distinct.
        ("qint8", False), ("quint8", False), ("qint32", False),
        ("quint4x2", False), ("quint2x4", False),
        # low-precision float8 / float4 -- unsupported for compute, but the
        # dtype objects must exist (transformers/safetensors reference them).
        ("float8_e4m3fn", True), ("float8_e4m3fnuz", True),
        ("float8_e5m2", True), ("float8_e5m2fnuz", True),
        ("float8_e8m0fnu", True), ("float4_e2m1fn_x2", True),
        # sub-byte unsigned dtypes used by torchao/diffusers import-time tables.
        # They are placeholders only; Jittor kernels do not implement them.
        ("uint1", False), ("uint2", False), ("uint3", False), ("uint4", False),
        ("uint5", False), ("uint6", False), ("uint7", False),
    ]
    objs = {}
    for name, is_fp in specs:
        converter = getattr(ns, name, None)
        if callable(converter) and not isinstance(converter, dtype):
            _NATIVE_DTYPE_CONVERTERS.setdefault(name, converter)
        objs[name] = dtype(name, is_fp)
    objs["float"] = objs["float32"]
    objs["double"] = objs["float64"]
    objs["half"] = objs["float16"]
    objs["short"] = objs["int16"]
    objs["int"] = objs["int32"]
    objs["long"] = objs["int64"]
    objs["cfloat"] = objs["complex64"]
    objs["cdouble"] = objs["complex128"]
    for k, v in objs.items():
        setattr(ns, k, v)
    return objs


def _dtype_to_str(d):
    if d is None:
        return None
    if isinstance(d, dtype):
        return d.name
    if isinstance(d, str):
        return d.replace("torch.", "")
    if callable(d) and hasattr(d, "__name__"):
        return d.__name__
    return str(d)


class device:
    def __init__(self, type="cpu", index=None):
        if isinstance(type, device):
            self.type, self.index = type.type, type.index
            return
        if isinstance(type, str):
            if ":" in type:
                t, i = type.split(":")
                self.type, self.index = t, int(i)
            else:
                self.type, self.index = type, index
        else:
            self.type, self.index = "cpu", None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self):
        return f"device(type='{self.type}'" + (f", index={self.index})" if self.index is not None else ")")

    def __eq__(self, other):
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other or self.type == other
        return NotImplemented

    def __hash__(self):
        return hash((self.type, self.index))

    # torch allows `with torch.device(...):` as a device context manager.
    # jittor has a single global backend, so for real devices this is a no-op.
    #
    # transformers' from_pretrained builds the model under `with
    # torch.device("meta")` and uses that context to SKIP weight inits (and the
    # `_is_hf_initialized` marking) -- see modeling_utils.get_torch_context_
    # manager_or_global_device(), which probes `torch.tensor([]).device`. If the
    # inits run anyway, modules end up flagged initialized, and the later
    # `_initialize_missing_keys()` step never recomputes non-persistent buffers
    # (e.g. RoPE inv_freq), leaving them as the `torch.empty_like` garbage that
    # `_move_missing_keys_from_meta_to_device` wrote. We can't allocate real
    # meta tensors in jittor, but we can make the *meta* context observable: push
    # it on a thread-local stack so Var.device reports "meta" inside it. Tensors
    # are still really allocated (harmless -- real weights get loaded over them),
    # but transformers correctly skips the eager init.
    def __enter__(self):
        if self.type == "meta":
            _DEVICE_CTX_STACK.append(self)
        return self

    def __exit__(self, *exc):
        if self.type == "meta" and _DEVICE_CTX_STACK and _DEVICE_CTX_STACK[-1] is self:
            _DEVICE_CTX_STACK.pop()
        return False


# Stack of active `torch.device("meta")` contexts (see device.__enter__).
# Only meta contexts are tracked; real-device `with` blocks stay no-ops.
# Model construction in from_pretrained is single-threaded, so a plain list
# is sufficient.
_DEVICE_CTX_STACK = []


Number = typing.Union[int, float, bool]
Device = typing.Union[device, str, int, type(None)]
FileLike = typing.Union[str, os.PathLike, typing.IO[bytes]]


class Storage:
    """Typing-level storage protocol exposed by ``torch.types``."""

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def _new_shared(self, size):
        raise NotImplementedError

    def _write_file(self, file, is_real_file, save_size, element_size):
        raise NotImplementedError

    def element_size(self):
        raise NotImplementedError


def make_torch_types_module():
    module = _python_types.ModuleType("torch.types")
    module.Number = Number
    module.Device = Device
    module.FileLike = FileLike
    module.Storage = Storage
    module._Number = (int, float, bool)
    module.__all__ = ["Number", "Device", "FileLike", "Storage"]
    return module


# ---- CPU-residency support for the torch device= API -----------------------
# jittor places a Var on CUDA or host according to the GLOBAL jt.flags.use_cuda
# flag, with no per-tensor device. But native C++ extensions linked against the
# jtorch ABI (nvdiffrast's `ranges`, cumesh's xatlas) call TORCH_CHECK on
# tensor.is_cpu(), and jtorch's is_cpu()/device() read the Var's ACTUAL memory
# residency (var->allocator->is_cuda(), surfaced to Python as Var.location() ==
# "cpu" vs "device"). A Var built/computed under `flag_scope(use_cuda=0)` is
# genuinely host-resident, so torch code that asks for device='cpu' can be
# honored by routing the allocation through use_cuda=0 -- the C++ shim then
# correctly sees it as CPU. These helpers implement that bounded device='cpu'
# support (creation + .cpu()/.cuda()/.to() migration + residency reporting).

def _device_is_cpu(dev):
    """True if a torch device= argument designates the CPU.

    Accepts the torch_compat `device` class, a torch.device-like object with a
    `.type`, a "cpu"/"cpu:0" string, or None (None means 'use the global default
    placement', i.e. NOT an explicit CPU request)."""
    if dev is None:
        return False
    t = getattr(dev, "type", None)
    if t is not None:
        return t == "cpu"
    if isinstance(dev, str):
        return dev == "cpu" or dev.split(":")[0] == "cpu"
    return False


def _device_is_cuda(dev):
    """True if a torch device= argument explicitly designates CUDA/GPU."""
    if dev is None:
        return False
    t = getattr(dev, "type", None)
    if t is not None:
        return t in ("cuda", "npu")
    if isinstance(dev, str):
        return dev.split(":")[0] in ("cuda", "npu")
    return False


def _var_is_cpu_resident(v):
    """True if a Var's data actually lives in host memory.

    Uses Var.location() (var->allocator->is_cuda()), the same residency that
    jtorch's C++ is_cpu()/device() report -- NOT the global use_cuda flag. A
    not-yet-materialized Var reports 'none'; treat that as following the global
    flag (it will land per use_cuda when realized)."""
    try:
        if getattr(v, "_jittor_torch_force_cpu", False):
            return True
        if getattr(v, "_jittor_torch_force_cuda", False):
            return False
    except Exception:
        pass
    try:
        loc = v.location()
    except Exception:
        return False
    if loc == "cpu":
        return True
    if loc == "device":
        return False
    # 'none' (unmaterialized) / 'disk': fall back to the global placement flag.
    return not bool(jt.flags.use_cuda)


def _var_has_cpu_residency_hint(v):
    try:
        return bool(getattr(v, "_jittor_torch_force_cpu", False))
    except Exception:
        return False


def _make_cpu_resident(v, inplace=False):
    """Return a host-resident Var.

    Tensor.cpu() asks for a copy, while Module.cpu()/to("cpu") is in-place. Use
    the native Var storage migration when available and keep the old NumPy
    rebuild only as a fallback for older cores.
    """
    if not isinstance(v, jt.Var):
        return v
    if _var_is_cpu_resident(v):
        return v
    if v.numel() == 0:
        out = v if inplace else v.clone()
        out._jittor_torch_force_cpu = True
        out._jittor_torch_force_cuda = False
        return out
    if hasattr(v, "migrate_to_cpu"):
        try:
            out = v if inplace else v.clone()
            out.migrate_to_cpu()
            try:
                out._jittor_torch_force_cpu = True
            except Exception:
                pass
            return out
        except Exception:
            pass
    try:
        arr = v.clone().numpy()
    except Exception:
        arr = v.numpy()
    with jt.flag_scope(use_cuda=0):
        out = jt.array(arr)
        out.sync()
    try:
        out._jittor_torch_force_cpu = True
    except Exception:
        pass
    return out


def _make_cuda_resident(v, force=False, inplace=False):
    """Return a CUDA-resident Var.

    Prefer native storage migration over a NumPy round-trip. The latter remains
    as a compatibility fallback for unmaterialized or older-core Vars.
    """
    if not isinstance(v, jt.Var):
        return v
    if not jt.flags.use_cuda:
        return v
    loc = None
    try:
        loc = v.location()
    except Exception:
        loc = None
    if loc == "device":
        return v
    if v.numel() == 0:
        out = v if inplace or loc != "cpu" else v.clone()
        out._jittor_torch_force_cpu = False
        out._jittor_torch_force_cuda = True
        return out
    try:
        if not getattr(v, "_jittor_torch_force_cpu", False) and loc not in ("cpu", "disk"):
            with jt.flag_scope(use_cuda=1):
                v.sync()
            if v.location() == "device":
                try:
                    v._jittor_torch_force_cpu = False
                except Exception:
                    pass
                return v
    except Exception:
        pass
    try:
        if not force and v.location() == "device":
            return v
    except Exception:
        pass
    if hasattr(v, "migrate_to_gpu"):
        try:
            # A lazy clone of a CPU Var may migrate the source when global
            # use_cuda=1. Keep tensor.cuda() copy semantics by reserving native
            # CPU->GPU migration for in-place Module.to/cuda paths.
            if (not inplace) and loc == "cpu":
                raise RuntimeError("preserve source CPU tensor")
            out = v if inplace else v.clone()
            out.migrate_to_gpu()
            try:
                out._jittor_torch_force_cpu = False
            except Exception:
                pass
            return out
        except Exception:
            pass
    arr = v.numpy()
    with jt.flag_scope(use_cuda=1):
        out = jt.array(arr)
        out.sync()
    try:
        out._jittor_torch_force_cpu = False
    except Exception:
        pass
    return out


def _mark_cpu_like(out, *inputs):
    try:
        if not isinstance(out, jt.Var):
            return out
        for x in inputs:
            if not isinstance(x, jt.Var):
                continue
            try:
                if getattr(x, "_jittor_torch_force_cpu", False):
                    out._jittor_torch_force_cpu = True
                    break
            except Exception:
                pass
    except Exception:
        pass
    return out
