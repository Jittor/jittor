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
import numpy as np


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

    def __repr__(self):
        return "torch." + self.name

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other or ("torch." + self.name) == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


def _make_dtypes(ns):
    specs = [
        ("float32", True), ("float64", True), ("float16", True),
        ("bfloat16", True),
        ("int8", False), ("int16", False), ("int32", False), ("int64", False),
        ("uint8", False), ("bool", False),
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


def install(torch):
    g = torch
    _make_dtypes(g)
    g.dtype = dtype
    g.device = device

    Var = jt.Var
    g.Tensor = Var
    g.FloatTensor = Var
    g.LongTensor = Var

    def tensor(data, dtype=None, device=None, requires_grad=False, **kw):
        if isinstance(data, Var):
            v = data.clone()
        else:
            v = jt.array(data)
        ds = _dtype_to_str(dtype)
        if ds is not None:
            v = v.cast(ds)
        return v
    g.tensor = tensor

    def as_tensor(data, dtype=None, device=None):
        if isinstance(data, Var):
            return data if dtype is None else data.cast(_dtype_to_str(dtype))
        return tensor(data, dtype=dtype)
    g.as_tensor = as_tensor

    def from_numpy(arr):
        return jt.array(arr)
    g.from_numpy = from_numpy

    class Size(tuple):
        def numel(self):
            n = 1
            for d in self:
                n *= d
            return n
    g.Size = Size

    if not hasattr(g, "cat"):
        g.cat = jt.concat
    g.concat = jt.concat

    if not hasattr(nn, "functional"):
        import types as _types
        F = _types.ModuleType("jittor.nn.functional")
        for fname in dir(nn):
            fobj = getattr(nn, fname)
            if callable(fobj) and not isinstance(fobj, type):
                setattr(F, fname, fobj)
        if hasattr(nn, "relu"): F.relu = nn.relu
        if hasattr(nn, "gelu"): F.gelu = nn.gelu
        if hasattr(nn, "softmax"): F.softmax = nn.softmax
        if hasattr(nn, "linear"): F.linear = nn.linear
        if hasattr(nn, "cross_entropy_loss"): F.cross_entropy = nn.cross_entropy_loss
        if hasattr(nn, "layer_norm"): F.layer_norm = nn.layer_norm
        if hasattr(nn, "embedding"): F.embedding = nn.embedding
        nn.functional = F
    g.nn.functional = nn.functional

    _install_cuda(g)
    _install_tensor_methods(g, Var)
    _install_misc(g, Var)


def _install_cuda(g):
    import types as _types, contextlib
    cuda = _types.ModuleType("torch.cuda")
    def is_available():
        try:
            return bool(jt.flags.use_cuda) or bool(getattr(jt.compiler, "has_acl", 0))
        except Exception:
            return False
    cuda.is_available = is_available
    cuda.device_count = lambda: 1
    cuda.current_device = lambda: 0
    cuda.set_device = lambda *a, **k: None
    cuda.empty_cache = lambda: None
    cuda.synchronize = lambda *a, **k: jt.sync_all(True)
    cuda.manual_seed = lambda s: jt.set_global_seed(int(s))
    cuda.manual_seed_all = lambda s: jt.set_global_seed(int(s))
    cuda.is_bf16_supported = lambda: True
    cuda.get_device_capability = lambda *a, **k: (8, 0)
    cuda.get_device_name = lambda *a, **k: "Ascend910B/NPU"
    cuda.get_device_properties = lambda *a, **k: type("P", (), {"total_memory": 64*1024**3, "name": "Ascend910B"})()
    class _amp:
        @staticmethod
        def autocast(*a, **k):
            return contextlib.nullcontext()
    cuda.amp = _amp
    g.cuda = cuda


def _install_tensor_methods(g, Var):
    if not hasattr(Var, "device"):
        def _device(self):
            return device("cuda", 0) if (jt.flags.use_cuda or getattr(jt.compiler, "has_acl", 0)) else device("cpu")
        Var.device = property(_device)

    if not hasattr(Var, "requires_grad"):
        def _rg_get(self):
            try:
                return not self.is_stop_grad()
            except Exception:
                return False
        def _rg_set(self, v):
            if v: self.start_grad()
            else: self.stop_grad()
        Var.requires_grad = property(_rg_get, _rg_set)

    def requires_grad_(self, v=True):
        self.requires_grad = v
        return self
    Var.requires_grad_ = requires_grad_

    def _to(self, *args, **kwargs):
        ds = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, dtype):
                ds = a.name
            elif isinstance(a, str) and a.replace("torch.", "") in dtype._registry:
                ds = a.replace("torch.", "")
        if ds is not None:
            return self.cast(ds)
        return self
    Var.to = _to

    if not hasattr(Var, "contiguous"):
        Var.contiguous = lambda self: self
    if not hasattr(Var, "is_cuda"):
        Var.is_cuda = property(lambda self: bool(jt.flags.use_cuda) or bool(getattr(jt.compiler, "has_acl", 0)))


def _install_misc(g, Var):
    if hasattr(jt, "set_global_seed"):
        g.manual_seed = lambda s: jt.set_global_seed(int(s))
    g.is_tensor = lambda x: isinstance(x, Var)
    if not hasattr(g, "numel"):
        g.numel = lambda x: x.numel()
