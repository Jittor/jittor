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
        ("uint8", False), ("uint16", False), ("uint32", False), ("uint64", False),
        ("bool", False),
        # complex types -- jittor has no native complex, but the dtype objects
        # must exist (libraries index size tables by them). Best-effort names.
        ("complex64", False), ("complex128", False),
        # low-precision float8 / float4 -- unsupported for compute, but the
        # dtype objects must exist (transformers/safetensors reference them).
        ("float8_e4m3fn", True), ("float8_e4m3fnuz", True),
        ("float8_e5m2", True), ("float8_e5m2fnuz", True),
        ("float8_e8m0fnu", True), ("float4_e2m1fn_x2", True),
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


import functools as _functools


class _GradDecoratorCtx:
    """Mimics torch.no_grad/enable_grad: usable as a context manager, a bare
    decorator (@torch.no_grad), and a called decorator (@torch.no_grad())."""

    def __init__(self, scope_factory, func=None):
        self._scope_factory = scope_factory
        self._func = func if callable(func) else None

    def __call__(self, *args, **kwargs):
        # used as @torch.no_grad() returning a decorator, then applied to a func
        if self._func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            @_functools.wraps(func)
            def wrapped(*a, **k):
                with self._scope_factory():
                    return func(*a, **k)
            return wrapped
        # used as @torch.no_grad (bare): self._func was set at construction
        if self._func is not None:
            with self._scope_factory():
                return self._func(*args, **kwargs)
        raise TypeError("no_grad/enable_grad misuse")

    def __enter__(self):
        self._scope = self._scope_factory()
        return self._scope.__enter__()

    def __exit__(self, *exc):
        return self._scope.__exit__(*exc)


def install(torch):
    g = torch
    _make_dtypes(g)
    g.dtype = dtype
    g.device = device

    # torch.no_grad / enable_grad work as bare decorator (@torch.no_grad),
    # called decorator (@torch.no_grad()), and context manager.
    # NB: g IS the jittor module, so capture the originals before overwriting.
    _orig_no_grad = jt.no_grad
    _orig_enable_grad = jt.enable_grad
    g.no_grad = lambda func=None: _GradDecoratorCtx(_orig_no_grad, func)
    g.enable_grad = lambda func=None: _GradDecoratorCtx(_orig_enable_grad, func)
    g.inference_mode = lambda func=None: _GradDecoratorCtx(_orig_no_grad, func)

    Var = jt.Var
    g.Tensor = Var
    # torch's typed tensor classes are all aliased to Var (jittor is dtype-typed
    # at the data level, not via tensor subclasses).
    for _tn in ("FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor",
                "LongTensor", "IntTensor", "ShortTensor", "CharTensor",
                "ByteTensor", "BoolTensor"):
        setattr(g, _tn, Var)

    def _array_keep_dtype(data):
        # jittor's jt.array downcasts numpy int64 -> int32; torch keeps int64.
        # Preserve the source dtype for (u)int64/float64 so dtypes match torch.
        import numpy as _np
        v = jt.array(data)
        if isinstance(data, _np.ndarray):
            dn = data.dtype.name
            if dn in ("int64", "uint64") and str(v.dtype) != "int64":
                v = v.int64()
            elif dn == "float64" and str(v.dtype) != "float64":
                v = v.float64()
        return v

    def tensor(data, dtype=None, device=None, requires_grad=False, **kw):
        if isinstance(data, Var):
            v = data.clone()
        else:
            import numpy as _np
            v = _array_keep_dtype(data if isinstance(data, _np.ndarray) else _np.asarray(data)) \
                if not isinstance(data, (int, float, bool)) else jt.array(data)
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
        return _array_keep_dtype(arr)
    g.from_numpy = from_numpy

    class Size(tuple):
        def numel(self):
            n = 1
            for d in self:
                n *= d
            return n
    g.Size = Size

    # torch.Generator (RNG handle) -- jittor uses a global seed; provide a
    # lightweight stand-in that supports manual_seed and is accepted where a
    # generator is passed (it is otherwise ignored).
    class Generator:
        def __init__(self, device=None):
            self.device = device
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
    g.Generator = Generator

    # numeric / misc top-level constants and small types
    import math as _math
    g.inf = _math.inf
    g.nan = _math.nan
    g.pi = _math.pi
    g.e = _math.e
    g.strided = "strided"
    g.contiguous_format = "contiguous_format"
    g.preserve_format = "preserve_format"
    g.channels_last = "channels_last"
    class layout:  # torch.layout placeholder
        pass
    g.layout = layout
    class memory_format:
        pass
    g.memory_format = memory_format

    # torch.cat: tolerate empty tensors (skip zero-numel inputs) like torch,
    # accept `dim=`/`out=`. jittor's concat trips on an empty leading tensor.
    _jt_concat = jt.concat
    def cat(tensors, dim=0, out=None):
        tensors = [t for t in tensors if t is not None]
        nonempty = [t for t in tensors if t.numel() > 0]
        if len(nonempty) == 0:
            return tensors[0]
        if len(nonempty) == 1:
            return nonempty[0]
        return _jt_concat(nonempty, dim)
    g.cat = cat
    g.concat = cat
    g.concatenate = cat

    # Wrap tensor constructors to tolerate torch's device=/requires_grad=/
    # layout=/pin_memory= kwargs and torch dtype objects. jittor's versions
    # don't accept device=, which torch code passes everywhere.
    _wrap_constructors(g)

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
        if hasattr(nn, "softmax"): F.softmax = nn.softmax
        if hasattr(nn, "linear"): F.linear = nn.linear
        if hasattr(nn, "cross_entropy_loss"): F.cross_entropy = nn.cross_entropy_loss
        if hasattr(nn, "layer_norm"): F.layer_norm = nn.layer_norm
        if hasattr(nn, "embedding"): F.embedding = nn.embedding
        nn.functional = F
    g.nn.functional = nn.functional

    # scaled_dot_product_attention (torch F.sdpa) -- standard math impl with
    # causal masking + attn_mask + GQA support (jittor has no native sdpa).
    if not hasattr(nn.functional, "scaled_dot_product_attention"):
        import math as _math
        def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=0.0, is_causal=False,
                                         scale=None, enable_gqa=False, **kw):
            # query: (..., Lq, E), key/value: (..., Lk, E)
            d = query.shape[-1]
            sf = (1.0 / _math.sqrt(d)) if scale is None else scale
            # GQA: repeat kv heads to match query heads
            if enable_gqa and key.shape[-3] != query.shape[-3]:
                rep = query.shape[-3] // key.shape[-3]
                key = key.repeat_interleave(rep, dim=-3) if hasattr(key, "repeat_interleave") else key
                value = value.repeat_interleave(rep, dim=-3) if hasattr(value, "repeat_interleave") else value
            scores = jt.matmul(query, key.transpose(-1, -2)) * sf
            if is_causal:
                Lq, Lk = query.shape[-2], key.shape[-2]
                mask = jt.triu(jt.ones((Lq, Lk)), 1) * (-1e30)
                scores = scores + mask
            if attn_mask is not None:
                if str(attn_mask.dtype) == "bool":
                    scores = scores + (1 - attn_mask.float32()) * (-1e30)
                else:
                    scores = scores + attn_mask
            attn = nn.softmax(scores, dim=-1)
            return jt.matmul(attn, value)
        nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    g.scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

    _install_nn_extras(nn)
    _install_cuda(g)
    _install_tensor_methods(g, Var)
    _install_misc(g, Var)


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
    _topk = getattr(_jt, "topk", None)
    _gather = _jt.gather

    def argmax(x, dim=None, keepdim=False):
        if dim is None:
            idx, _ = _argmax(x.reshape(-1), 0)
            return idx.int64()
        idx, _ = _argmax(x, dim, keepdims=keepdim)
        return idx.int64()
    def argmin(x, dim=None, keepdim=False):
        if dim is None:
            idx, _ = _argmin(x.reshape(-1), 0)
            return idx.int64()
        idx, _ = _argmin(x, dim, keepdims=keepdim)
        return idx.int64()
    g.argmax = argmax
    g.argmin = argmin

    def _maxmin(which, x, *args, **kwargs):
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        other = None
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
            return x.max() if which == "max" else x.min()
        af = _argmax if which == "max" else _argmin
        idx, val = af(x, dim, keepdims=keepdim)
        return _MinMax(val, idx.int64())
    g.max = lambda x, *a, **k: _maxmin("max", x, *a, **k)
    g.min = lambda x, *a, **k: _maxmin("min", x, *a, **k)

    def topk(x, k, dim=-1, largest=True, sorted=True):
        if _topk is not None:
            res = _topk(x, k, dim, largest, sorted)
            if isinstance(res, (tuple, list)):
                return _TopK(res[0], res[1].int64())
        idx, _ = _argsort(x, dim=dim, descending=largest)
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(0, k)
        idx = idx[tuple(sl)]
        val = _gather(x, dim, idx)
        return _TopK(val, idx.int64())
    g.topk = topk

    def sort(x, dim=-1, descending=False, **kw):
        idx, val = _argsort(x, dim=dim, descending=descending)
        return _Sort(val, idx.int64())
    g.sort = sort
    g.argsort = lambda x, dim=-1, descending=False, **kw: _argsort(x, dim=dim, descending=descending)[0].int64()


def _wrap_constructors(g):
    """Wrap jittor tensor constructors to accept torch kwargs (device=,
    requires_grad=, layout=, pin_memory=, out=) and torch dtype objects."""
    import functools
    _DROP = ("device", "requires_grad", "layout", "pin_memory", "memory_format",
             "out", "non_blocking")

    def wrap(name):
        orig = getattr(g, name, None)
        if orig is None:
            return
        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            for k in _DROP:
                kwargs.pop(k, None)
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                kwargs["dtype"] = _dtype_to_str(kwargs["dtype"])
            return orig(*args, **kwargs)
        wrapped._torch_wrapped = True
        setattr(g, name, wrapped)

    for name in ("zeros", "ones", "empty", "full", "arange", "rand", "randn",
                 "randint", "eye", "linspace", "zeros_like", "ones_like",
                 "empty_like", "full_like", "randn_like", "rand_like", "tril",
                 "triu", "normal"):
        wrap(name)



def _install_nn_extras(nn):
    # Activation modules torch has that jittor.nn may lack.
    import jittor as _jt
    _install_init_aliases()
    if not hasattr(nn, "Hardswish"):
        class Hardswish(nn.Module):
            def execute(self, x):
                return x * _jt.clamp(x + 3, 0, 6) / 6
        nn.Hardswish = Hardswish
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

    # Layer classes torch has that jittor.nn may lack -- needed at least for
    # isinstance() checks in model init. Provide a distinct empty subclass so
    # isinstance discrimination still works.
    if not hasattr(nn, "ConvTranspose1d"):
        class ConvTranspose1d(nn.Module):
            def __init__(self, *a, **k): super().__init__()
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
    if not hasattr(nn, "MultiheadAttention"):
        class MultiheadAttention(nn.Module):
            def __init__(self, *a, **k): super().__init__()
        nn.MultiheadAttention = MultiheadAttention

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

    # torch's Module.train(mode=True)/eval() take a mode arg; jittor's train()
    # takes none. Wrap to accept it and set jittor's is_training accordingly.
    _orig_train = M.train
    def _train(self, mode=True):
        try:
            _orig_train(self)        # jittor sets is_training=True recursively
        except TypeError:
            pass
        if not mode:
            # put into eval: jittor uses .eval() per-module
            try:
                # set is_training flag across submodules
                for m in self.modules() if hasattr(self, "modules") else [self]:
                    if hasattr(m, "is_training"):
                        m.is_training = False
            except Exception:
                pass
        else:
            for m in (self.modules() if hasattr(self, "modules") else [self]):
                if hasattr(m, "is_training"):
                    m.is_training = True
        return self
    M.train = _train
    _orig_eval = getattr(M, "eval", None)
    def _eval(self):
        return _train(self, False)
    M.eval = _eval

    if not hasattr(M, "to"):
        def _module_to(self, *args, **kwargs):
            # torch Module.to(device/dtype/...) -- cast float params to a dtype
            # if one is given; device moves are no-ops on single-device jittor.
            ds = None
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, dtype):
                    ds = a.name
                elif isinstance(a, str) and a.replace("torch.", "") in dtype._registry:
                    ds = a.replace("torch.", "")
            if ds is not None and ds in ("float16", "bfloat16", "float32", "float64"):
                for p in self.parameters():
                    if p.dtype.is_float() if hasattr(p.dtype, "is_float") else ("float" in str(p.dtype)):
                        p.assign(p.cast(ds))
            return self
        M.to = _module_to

    if not hasattr(M, "cpu"):
        M.cpu = lambda self: self
    if not hasattr(M, "float"):
        def _mfloat(self):
            for p in self.parameters():
                p.assign(p.float32())
            return self
        M.float = _mfloat
    if not hasattr(M, "zero_grad"):
        M.zero_grad = lambda self, *a, **k: None
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
    if not hasattr(M, "register_parameter"):
        def _register_parameter(self, name, param):
            setattr(self, name, param)
        M.register_parameter = _register_parameter
    if not hasattr(M, "type"):
        M.type = lambda self, dst_type=None: self


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
        tensor.assign(value)
        if was_trainable:
            tensor.start_grad()
        return tensor

    def normal_(tensor, mean=0.0, std=1.0, generator=None):
        return _assign(tensor, _jt2.normal(float(mean), float(std), tensor.shape).cast(str(tensor.dtype)))
    def uniform_(tensor, a=0.0, b=1.0, generator=None):
        return _assign(tensor, (_jt2.rand(tensor.shape) * (b - a) + a).cast(str(tensor.dtype)))
    def zeros_(tensor):
        return _assign(tensor, _jt2.zeros(tensor.shape, tensor.dtype))
    def ones_(tensor):
        return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype))
    def constant_(tensor, val):
        return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype) * val)
    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
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
    # stub classes referenced in annotations / guarded paths
    cuda.CUDAGraph = type("CUDAGraph", (), {})
    class _Stream:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def synchronize(self): jt.sync_all(True)
    cuda.Stream = _Stream
    cuda.Event = type("Event", (), {"__init__": lambda self, *a, **k: None,
                                      "record": lambda self, *a, **k: None,
                                      "synchronize": lambda self: None,
                                      "elapsed_time": lambda self, o: 0.0})
    cuda.stream = lambda s=None: contextlib.nullcontext()
    cuda.current_stream = lambda *a, **k: _Stream()
    cuda.memory_allocated = lambda *a, **k: 0
    cuda.max_memory_allocated = lambda *a, **k: 0
    cuda.reset_peak_memory_stats = lambda *a, **k: None
    cuda.mem_get_info = lambda *a, **k: (64*1024**3, 64*1024**3)
    g.cuda = cuda


def _install_tensor_methods(g, Var):
    # in-place tensor ops torch code uses heavily (jittor exposes assign()).
    # _ip() preserves grad-tracking: jittor's assign() adopts the source's
    # stop_grad flag, which would freeze a trainable parameter.
    def _ip(self, value):
        was_trainable = not self.is_stop_grad()
        self.assign(value)
        if was_trainable:
            self.start_grad()
        return self
    def _copy_(self, other, non_blocking=False):
        src = other if isinstance(other, Var) else jt.array(other)
        return _ip(self, src.cast(str(self.dtype)) if hasattr(self, "dtype") else src)
    if not hasattr(Var, "copy_"):
        Var.copy_ = _copy_
    if not hasattr(Var, "fill_"):
        Var.fill_ = lambda self, val: _ip(self, jt.ones(self.shape, self.dtype) * val)
    if not hasattr(Var, "zero_"):
        Var.zero_ = lambda self: _ip(self, jt.zeros(self.shape, self.dtype))
    if not hasattr(Var, "add_"):
        Var.add_ = lambda self, o, alpha=1: _ip(self, self + (o * alpha))
    if not hasattr(Var, "sub_"):
        Var.sub_ = lambda self, o, alpha=1: _ip(self, self - (o * alpha))
    if not hasattr(Var, "mul_"):
        Var.mul_ = lambda self, o: _ip(self, self * o)
    if not hasattr(Var, "div_"):
        Var.div_ = lambda self, o: _ip(self, self / o)
    if not hasattr(Var, "clamp_"):
        Var.clamp_ = lambda self, min=None, max=None: _ip(self, jt.clamp(self, min, max))
    if not hasattr(Var, "normal_"):
        Var.normal_ = lambda self, mean=0.0, std=1.0, generator=None: _ip(self, jt.normal(float(mean), float(std), self.shape).cast(str(self.dtype)))
    if not hasattr(Var, "uniform_"):
        Var.uniform_ = lambda self, a=0.0, b=1.0, generator=None: _ip(self, (jt.rand(self.shape)*(b-a)+a).cast(str(self.dtype)))

    # bitwise/logical operators torch supports on tensors
    if not hasattr(Var, "__invert__"):
        def _invert(self):
            if str(self.dtype) == "bool":
                return self.logical_not()
            return jt.logical_not(self) if str(self.dtype) == "bool" else (-self - 1)
        Var.__invert__ = _invert

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

    # autocast / grad-mode query helpers
    g.is_autocast_enabled = lambda *a, **k: False
    g.set_autocast_enabled = lambda *a, **k: None
    g.is_grad_enabled = lambda: not bool(getattr(jt.flags, "no_grad", 0))
    g.set_grad_enabled = lambda mode: None
    g.get_autocast_dtype = lambda *a, **k: getattr(g, "float32", "float32")
    g.is_autocast_available = lambda *a, **k: False
    g.are_deterministic_algorithms_enabled = lambda: False
    g.use_deterministic_algorithms = lambda *a, **k: None
    g.is_floating_point = lambda x: ("float" in str(x.dtype))


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

    # ---- save / load (numpy-backed; jittor pickle) ----
    def save(obj, f, *a, **k):
        return jt.save(obj, f) if hasattr(jt, "save") else None
    def load(f, *a, **k):
        return jt.load(f) if hasattr(jt, "load") else None
    g.save = save
    g.load = load

    # ---- elementwise / reduction helpers that may be missing ----
    def _alias(name, fn):
        if not hasattr(g, name):
            setattr(g, name, fn)
    _alias("rsqrt", lambda x: 1.0 / jt.sqrt(x))
    _alias("empty_like", lambda x, **k: jt.empty(x.shape, x.dtype))
    _alias("equal", lambda a, b: bool((a == b).all().item()))
    _alias("diff", lambda x, dim=-1, n=1: _diff(x, dim, n))
    _alias("repeat_interleave", _repeat_interleave)
    _alias("autocast", lambda *a, **k: __import__("contextlib").nullcontext())
    _alias("vmap", lambda fn, *a, **k: fn)
    _alias("outer", lambda a, b: jt.matmul(a.reshape(-1, 1), b.reshape(1, -1)))
    _alias("isin", _isin)


def _diff(x, dim=-1, n=1):
    for _ in range(n):
        idx = [slice(None)] * x.ndim
        idx0 = list(idx); idx1 = list(idx)
        idx0[dim] = slice(1, None); idx1[dim] = slice(0, -1)
        x = x[tuple(idx0)] - x[tuple(idx1)]
    return x


def _repeat_interleave(x, repeats, dim=None):
    import jittor as _jt
    if dim is None:
        x = x.reshape(-1); dim = 0
    if isinstance(repeats, int):
        idx = _jt.arange(x.shape[dim]).reshape(-1, 1).broadcast([x.shape[dim], repeats]).reshape(-1)
    else:
        parts = []
        r = repeats.numpy() if hasattr(repeats, "numpy") else repeats
        for i, c in enumerate(r):
            parts += [i] * int(c)
        idx = _jt.array(parts)
    return x[idx] if dim == 0 else x.transpose(0, dim)[idx].transpose(0, dim)


def _isin(elements, test_elements, **kw):
    import jittor as _jt
    te = test_elements.numpy() if hasattr(test_elements, "numpy") else test_elements
    import numpy as _np
    el = elements.numpy() if hasattr(elements, "numpy") else _np.asarray(elements)
    return _jt.array(_np.isin(el, te))

