"""Torch-compatible dtype, device, and residency primitives."""

import os
import types as _python_types
import typing
from typing import cast

import jittor as jt
from ..diagnostics import EXPECTED, swallowed

class dtype:
    """Immutable Torch dtype identity, independent of Python strings."""
    __slots__ = ("name", "_is_fp")
    if typing.TYPE_CHECKING:
        name: str
        _is_fp: bool
    _registry: typing.ClassVar[typing.Dict[str, "dtype"]] = {}
    _supported = frozenset({
        "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16",
        "int32", "int64", "float16", "bfloat16", "float32", "float64", "complex64",
    })

    def __new__(cls, name, is_floating_point=False):
        if not isinstance(name, str):
            raise TypeError("dtype name must be a string")
        if name in cls._registry:
            return cls._registry[name]
        obj = super().__new__(cls)
        object.__setattr__(obj, "name", name)
        object.__setattr__(obj, "_is_fp", name.startswith(("float", "bfloat")))
        cls._registry[name] = obj
        return obj

    def __setattr__(self, name, value):
        raise AttributeError("torch.dtype objects are immutable")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("torch.dtype cannot be subclassed")

    @property
    def is_floating_point(self):
        return self._is_fp

    @property
    def is_complex(self):
        return self.name.startswith("complex")

    @property
    def itemsize(self):
        sizes = {
            "bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2,
            "float16": 2, "bfloat16": 2, "uint32": 4, "int32": 4,
            "float32": 4, "uint64": 8, "int64": 8, "float64": 8,
            "complex32": 4, "complex64": 8, "complex128": 16,
            "qint8": 1, "quint8": 1, "qint32": 4, "quint4x2": 1, "quint2x4": 1,
        }
        if self.name.startswith(("float8_", "float4_", "uint")):
            return sizes.get(self.name, 1)
        return sizes[self.name]
    element_size = itemsize

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

    __str__ = __repr__

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash((type(self), self.name))

    def __reduce__(self):
        return _restore_dtype, (self.name, self._is_fp)

    def __setstate__(self, state):
        # Old dtype(str) pickles stored these immutable fields in a dict.
        # Validate that state rather than mutating the canonical singleton.
        if (not isinstance(state, dict) or set(state) - {"name", "_is_fp"}
                or state.get("name", self.name) != self.name
                or state.get("_is_fp", self._is_fp) != self._is_fp):
            raise ValueError("invalid serialized torch.dtype state")

    @property
    def _jittor_compute_name(self):
        if self.name not in self._supported:
            raise NotImplementedError(
                "torch.%s is a metadata-only dtype; Jittor has no computation or allocation support"
                % self.name)
        return self.name

def _restore_dtype(name, is_floating_point):
    return dtype(name, is_floating_point)


def _make_dtypes(ns):
    specs = [
        ("float32", True), ("float64", True), ("float16", True),
        ("bfloat16", True),
        ("int8", False), ("int16", False), ("int32", False), ("int64", False),
        ("uint8", False), ("uint16", False), ("uint32", False), ("uint64", False),
        ("bool", False),
        # complex64 is native; complex32/complex128 remain metadata-only.
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
    from jittor._core.dtypes import register_dtype_type
    register_dtype_type(dtype)
    return objs


def _dtype_to_str(d):
    if d is None:
        return None
    if isinstance(d, dtype):
        return d._jittor_compute_name
    if isinstance(d, str):
        name = d.replace("torch.", "")
        registered = dtype._registry.get(name)
        return registered._jittor_compute_name if registered is not None else name
    if callable(d) and hasattr(d, "__name__"):
        return d.__name__
    return str(d)


class device:
    type: str
    index: typing.Optional[int]
    _prev_index: typing.Optional[int]

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
    # An *indexed* CUDA device context is not a no-op any more: torch's
    # `with torch.device("cuda:1"):` makes device 1 the default new tensors
    # are built on, and jittor now has a current device that means exactly
    # that. A bare "cuda" still names the current device, so it stays a no-op.
    def __enter__(self):
        if self.type == "meta":
            _DEVICE_CTX_STACK.append(self)
        elif self.type in ("cuda", "npu") and self.index is not None:
            try:
                self._prev_index = int(jt.current_device())
                if self._prev_index != int(self.index):
                    jt.set_device(int(self.index))
            except EXPECTED as exc:
                swallowed("types.py device.__enter__: switch to %s" % (self,), exc,
                          "the block runs on the current device instead")
                self._prev_index = None
        return self

    def __exit__(self, *exc):
        if self.type == "meta" and _DEVICE_CTX_STACK and _DEVICE_CTX_STACK[-1] is self:
            _DEVICE_CTX_STACK.pop()
        prev = getattr(self, "_prev_index", None)
        if prev is not None and prev >= 0:
            try:
                if int(jt.current_device()) != prev:
                    jt.set_device(prev)
            except EXPECTED as exc:
                swallowed("types.py device.__exit__: restore device %r" % (prev,),
                          exc, "the block's device stays current after the block")
            self._prev_index = None
        return False


# Stack of active `torch.device("meta")` contexts (see device.__enter__).
# Only meta contexts are tracked; real-device `with` blocks stay no-ops.
# Model construction in from_pretrained is single-threaded, so a plain list
# is sufficient.
_DEVICE_CTX_STACK: typing.List[device] = []


Number = typing.Union[int, float, bool]
Device = typing.Union[device, str, int, type(None)]
FileLike = typing.Union[str, os.PathLike, typing.IO[bytes]]
SymInt = int
SymFloat = float
SymBool = bool
py_sym_types = (SymInt, SymFloat, SymBool)


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
    module = cast(typing.Any, _python_types.ModuleType("torch.types"))
    setattr(module, "Number", Number)
    setattr(module, "Device", Device)
    setattr(module, "FileLike", FileLike)
    setattr(module, "Storage", Storage)
    setattr(module, "_Number", (int, float, bool))
    setattr(module, "py_sym_types", py_sym_types)
    setattr(module, "PySymType", Number)
    setattr(module, "__all__", ["Number", "Device", "FileLike", "Storage"])
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


def _cuda_index_of(dev):
    """The CUDA device index a torch ``device=`` argument names, or None.

    None means "no particular device": a bare "cuda", ``torch.device("cuda")``
    or ``None`` itself, all of which mean the current device in torch.
    Accepts an int, a "cuda:N"/"npu:N" string, and any object with
    ``type``/``index``.

    Strings are handled before the attribute lookup on purpose: ``str.index``
    is a method, so ``getattr(dev, "index", None)`` on a string returns a
    bound method rather than None.
    """
    if dev is None or isinstance(dev, bool):
        return None
    if isinstance(dev, int):
        return dev if dev >= 0 else None
    if isinstance(dev, str):
        head, _, tail = dev.partition(":")
        if head not in ("cuda", "npu") or not tail:
            return None
        try:
            return int(tail)
        except ValueError:
            return None
    if getattr(dev, "type", None) in ("cuda", "npu"):
        idx = getattr(dev, "index", None)
        return int(idx) if isinstance(idx, int) and not isinstance(idx, bool) else None
    return None


def _move_to_cuda_index(v, dev, default_index=None):
    """Return ``v`` on the CUDA device ``dev`` names, copying when it is
    somewhere else.

    ``dev`` without an index -- a bare "cuda" -- means "wherever it already
    is", which is what torch's ``.to("cuda")`` does for an already-CUDA
    tensor. Pass the *original* Var's device as ``default_index`` for that
    case: the residency helpers rebuild a host-resident Var from scratch, and
    a rebuilt Var takes the current device, so without this a ``cuda:1``
    tensor comes back on ``cuda:0``."""
    idx = _cuda_index_of(dev)
    if idx is None:
        idx = default_index
    if idx is None or idx < 0 or not isinstance(v, jt.Var):
        return v
    try:
        current = int(v.device_id)
    except EXPECTED as exc:
        swallowed("types.py _move_to_cuda_index: current = int(v.device_id)", exc,
                  "the Var is left where it is instead of being moved")
        return v
    if current < 0 or current == idx:
        return v
    return v.to_device(idx)


def _var_is_cpu_resident(v):
    """Explicit tensor backend, or legacy storage residency when unplaced.

    Uses Var.location() (var->allocator->is_cuda()), the same residency that
    jtorch's C++ is_cpu()/device() report -- NOT the global use_cuda flag. A
    not-yet-materialized Var reports 'none'; treat that as following the global
    flag (it will land per use_cuda when realized)."""
    if isinstance(v, jt.Var) and v.placement_backend >= 0:
        return v.placement_backend == 0
    try:
        if getattr(v, "_jittor_torch_force_cpu", False):
            return True
        if getattr(v, "_jittor_torch_force_cuda", False):
            return False
    except (AttributeError, TypeError) as exc:
        swallowed("torch/types.py _var_is_cpu_resident: if getattr(v, '_jittor_torch_force_cpu', False):", exc)
    try:
        loc = v.location()
    except EXPECTED as exc:
        swallowed("torch/types.py _var_is_cpu_resident: loc = v.location()", exc)
        return False
    if loc == "cpu":
        return True
    if loc == "device":
        return False
    # 'none' (unmaterialized) / 'disk': fall back to the global placement flag.
    return not bool(jt.flags.use_cuda)


def _var_has_cpu_residency_hint(v):
    if isinstance(v, jt.Var) and v.placement_backend >= 0:
        return v.placement_backend == 0
    try:
        return bool(getattr(v, "_jittor_torch_force_cpu", False))
    except EXPECTED as exc:
        swallowed("torch/types.py _var_has_cpu_residency_hint: return bool(getattr(v, '_jittor_torch_force_cpu', False))", exc)
        return False


def _make_cpu_resident(v, inplace=False):
    """Return a host-resident Var.

    Tensor.cpu() asks for a copy, while Module.cpu()/to("cpu") is in-place. Use
    the native Var storage migration when available and keep the old NumPy
    rebuild only as a fallback for older cores.
    """
    if not isinstance(v, jt.Var):
        return v
    if v.placement_backend >= 0:
        if v.placement_backend == 0:
            return v
        moved = v._copy_to_cpu()
        if inplace:
            trainable = bool(v.requires_grad)
            v.assign(moved.detach())
            v.requires_grad = trainable
            return v
        return moved
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
            except (AttributeError, TypeError) as exc:
                swallowed("torch/types.py _make_cpu_resident: out._jittor_torch_force_cpu = True", exc,
                          "the Var will report residency from the global use_cuda flag "
                          "instead of from where it actually lives")
            return out
        except EXPECTED as exc:
            swallowed("torch/types.py _make_cpu_resident: out = v if inplace else v.clone()", exc)
    try:
        arr = v.clone().numpy()
    except EXPECTED as exc:
        swallowed("torch/types.py _make_cpu_resident: arr = v.clone().numpy()", exc)
        arr = v.numpy()
    with jt.flag_scope(use_cuda=0):
        out = jt.array(arr)
        out.sync()
    try:
        out._jittor_torch_force_cpu = True
    except (AttributeError, TypeError) as exc:
        swallowed("torch/types.py _make_cpu_resident: out._jittor_torch_force_cpu = True", exc,
                  "the Var will report residency from the global use_cuda flag "
                  "instead of from where it actually lives")
    return out


def _make_cuda_resident(v, force=False, inplace=False, device=None):
    """Return a CUDA-resident Var.

    Prefer native storage migration over a NumPy round-trip. The latter remains
    as a compatibility fallback for unmaterialized or older-core Vars.
    """
    if not isinstance(v, jt.Var):
        return v
    if v.placement_backend >= 0:
        from .frontend import _placement_request
        request = device
        if request is None:
            request = "cuda:%d" % v.device_id if v.placement_backend else "cuda"
        backend, index = _placement_request(jt, request)
        if v.placement_backend == backend and v.device_id == index:
            return v
        moved = v.to_device(index)
        if inplace:
            trainable = bool(v.requires_grad)
            v.assign(moved.detach())
            v.requires_grad = trainable
            return v
        return moved
    if not jt.flags.use_cuda:
        return v
    loc = None
    try:
        loc = v.location()
    except EXPECTED as exc:
        swallowed("torch/types.py _make_cuda_resident: loc = v.location()", exc)
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
                except (AttributeError, TypeError) as exc:
                    swallowed("torch/types.py _make_cuda_resident: v._jittor_torch_force_cpu = False", exc,
                              "the Var will keep reporting CPU residency after being moved")
                return v
    except EXPECTED as exc:
        swallowed("torch/types.py _make_cuda_resident: if not getattr(v, '_jittor_torch_force_cpu', False) and...", exc)
    try:
        if not force and v.location() == "device":
            return v
    except EXPECTED as exc:
        swallowed("torch/types.py _make_cuda_resident: if not force and v.location() == 'device':", exc)
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
            except (AttributeError, TypeError) as exc:
                swallowed("torch/types.py _make_cuda_resident: out._jittor_torch_force_cpu = False", exc,
                          "the Var will keep reporting CPU residency after being moved")
            return out
        except EXPECTED as exc:
            swallowed("torch/types.py _make_cuda_resident: if (not inplace) and loc == 'cpu':", exc)
    arr = v.numpy()
    with jt.flag_scope(use_cuda=1):
        out = jt.array(arr)
        out.sync()
    try:
        out._jittor_torch_force_cpu = False
    except (AttributeError, TypeError) as exc:
        swallowed("torch/types.py _make_cuda_resident: out._jittor_torch_force_cpu = False", exc,
                  "the Var will keep reporting CPU residency after being moved")
    return out


def _mark_cpu_like(out, *inputs):
    if isinstance(out, jt.Var) and out.placement_backend >= 0:
        return out
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
            except (AttributeError, TypeError) as exc:
                swallowed("torch/types.py _mark_cpu_like: if getattr(x, '_jittor_torch_force_cpu', False):", exc)
    except EXPECTED as exc:
        swallowed("torch/types.py _mark_cpu_like: if not isinstance(out, jt.Var):", exc)
    return out
