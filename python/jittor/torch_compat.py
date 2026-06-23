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

    # torch allows `with torch.device(...):` as a device context manager.
    # jittor has a single global backend, so this is a no-op scope.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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


class _AutocastContext:
    """torch.autocast is BOTH a context manager and a decorator -- accelerate does
    `new_forward = autocast(model_forward)`. On jittor, bf16/fp16 is determined by
    the actual tensor dtypes (no global autocast state), so this is a no-op that
    supports `with autocast(...):`, `@autocast(...)`, and `autocast(...)(fn)`."""
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


class _GradScaler:
    """Functional fp16 dynamic loss scaler (matches torch.cuda.amp.GradScaler).
    Works with the jittor optimizer bridge: scale(loss).backward() routes scaled
    grads into the optimizer; step() unscales, SKIPS the step on inf/nan, and
    update() grows/backs off the scale. bf16 doesn't need scaling but this is
    correct (and required) for fp16 mixed-precision training."""
    def __init__(self, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        self._enabled = enabled
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False
        self._unscaled = False

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale if self._enabled else 1.0

    def scale(self, outputs):
        return outputs * self._scale if self._enabled else outputs

    def _grads(self, opt):
        gs = []
        for pg in getattr(opt, "param_groups", []):
            for g in (pg.get("grads", []) or []):
                if g is not None:
                    gs.append(g)
        return gs

    def unscale_(self, opt):
        if not self._enabled:
            return
        import math as _m
        inv = 1.0 / self._scale
        found = False
        for g in self._grads(opt):
            g.update(g * inv)
            m = float(g.abs().max().item()) if g.numel() else 0.0
            if _m.isinf(m) or _m.isnan(m):
                found = True
        self._found_inf = found
        self._unscaled = True

    def step(self, opt, *a, **k):
        if not self._enabled:
            return opt.step(*a, **k)
        if not self._unscaled:
            self.unscale_(opt)
        self._unscaled = False
        if self._found_inf:
            return None  # skip optimizer step on overflow
        return opt.step(*a, **k)

    def update(self, new_scale=None):
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale); return
        if self._found_inf:
            self._scale = max(1.0, self._scale * self._backoff_factor)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def state_dict(self):
        return {"scale": self._scale, "growth_tracker": self._growth_tracker}

    def load_state_dict(self, sd):
        self._scale = sd.get("scale", self._scale)
        self._growth_tracker = sd.get("growth_tracker", 0)


def install(torch):
    g = torch
    # Critical: jittor dispatches every op to CPU unless flags.use_cuda is set.
    # The accelerator (Ascend NPU via jt.compiler.has_acl, or NVIDIA GPU via
    # jt.has_cuda) is present, but use_cuda defaults to 0 -- so `import torch` +
    # model.to("cuda") (a no-op here) would silently run the ENTIRE model on CPU,
    # ~10000x slower (a 2048^3 matmul: 20s CPU vs 2ms NPU). Enable device dispatch
    # globally whenever an accelerator exists, so tensors/ops land on it by default,
    # matching what torch users expect from .cuda()/.to(device).
    try:
        if getattr(jt.compiler, "has_acl", 0) or getattr(jt, "has_cuda", 0):
            jt.flags.use_cuda = 1
    except Exception:
        pass
    _DTYPE_OBJS = _make_dtypes(g)
    g.dtype = dtype
    g.device = device
    g.GradScaler = _GradScaler        # picked up by torch.amp/torch.cuda.amp in the shim

    # jt.grad's C-binding only accepts a *plain* list of targets, so passing the
    # torch-style parameters() iterator/_ParamList (a list subclass) or a single
    # Var raises a cryptic "Wrong inputs arguments". Coerce to a plain list (and
    # accept a lone Var, like torch.autograd.grad). Internal jittor callers pass a
    # plain list -> passthrough, so this never changes their behavior.
    _native_grad = g.grad
    def _grad_compat(loss, targets, *a, **k):
        if type(targets) is not list:
            if isinstance(targets, jt.Var):
                targets = [targets]
            else:
                try:
                    targets = list(targets)
                except Exception:
                    pass
        return _native_grad(loss, targets, *a, **k)
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
        import numpy as _np
        if isinstance(data, Var):
            v = data.clone()
        elif isinstance(data, _np.ndarray):
            v = _array_keep_dtype(data)          # explicit numpy: preserve dtype (torch does too)
        else:
            # Python scalar/list/tuple: numpy infers float64 from Python floats, but
            # torch's default float dtype is float32. Match torch (and avoid float64,
            # which Ascend/ACL does not support) by downcasting inferred float64.
            arr = _np.asarray(data)
            if arr.dtype == _np.float64:
                arr = arr.astype(_np.float32)
            v = _array_keep_dtype(arr)
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
    _install_tensor_methods(g, Var, _DTYPE_OBJS)
    _install_misc(g, Var)
    _install_optimizers(g)
    _install_autograd_function(g)


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


def _install_optimizers(g):
    """Register every jittor optimizer instance as g._current_optimizer on
    construction, and mirror lr into each param_group. This makes the
    `loss.backward()` bridge (Var.backward) and torch-style LR schedulers work
    even when using `import jittor as torch` directly (no torch_shim wrapper)."""
    import jittor as _jt
    try:
        from jittor import optim as _optim
    except Exception:
        return
    Base = getattr(_optim, "Optimizer", None)
    if Base is None or getattr(Base, "_torch_compat_wrapped", False):
        return
    _orig_init = Base.__init__
    def _init(self, *a, **k):
        _orig_init(self, *a, **k)
        _jt._current_optimizer = self
        try:
            for pg in self.param_groups:
                pg.setdefault("lr", self.lr)
        except Exception:
            pass
    Base.__init__ = _init
    Base._torch_compat_wrapped = True

    # jittor's load_state_dict runs a dfs that calls .stop_grad() on every Var
    # it meets -- including params nested under param_groups -- freezing all
    # trainable params (accelerate round-trips state_dict on wrap). Guard it.
    _orig_lsd = getattr(Base, "load_state_dict", None)
    if _orig_lsd is not None:
        def _lsd(self, state):
            trainable = []
            try:
                for pg in self.param_groups:
                    for p in pg.get("params", []):
                        if not p.is_stop_grad():
                            trainable.append(p)
            except Exception:
                pass
            r = _orig_lsd(self, state)
            for p in trainable:
                try: p.start_grad()
                except Exception: pass
            return r
        Base.load_state_dict = _lsd


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
            return x.max() if which == "max" else x.min()
        af = _argmax if which == "max" else _argmin
        idx, val = af(x, dim, keepdims=keepdim)
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
        def clip_grad_norm_(parameters, max_norm, norm_type=2.0, **k):
            if isinstance(parameters, _jt.Var):
                parameters = [parameters]
            grads = _grads_of(parameters)
            if not grads:
                return _jt.array(0.0)
            if norm_type == float("inf"):
                total = _jt.concat([g.abs().reshape(-1) for g in grads]).max()
            else:
                total = _jt.sqrt(_jt.concat([g.cast("float32").sqr().reshape(-1) for g in grads]).sum())
            mn = float(max_norm)
            if mn != float("inf"):
                coef = mn / (float(total.item()) + 1e-6)
                if coef < 1.0:
                    for g in grads:
                        g.update(g * coef)
            return total
        def clip_grad_value_(parameters, clip_value, **k):
            if isinstance(parameters, _jt.Var):
                parameters = [parameters]
            for g in _grads_of(parameters):
                g.update(g.clamp(-clip_value, clip_value))
        _u.clip_grad_norm_ = clip_grad_norm_
        _u.clip_grad_value_ = clip_grad_value_
        nn.utils = _u

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
        seen = set()
        for name, v in _orig_named_parameters(self, recurse=recurse):
            if remove_duplicate and id(v) in seen:
                continue
            seen.add(id(v))
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
    def _load_state_dict(self, state_dict, strict=True, assign=False):
        # preserve trainable flags: jittor assign can flip stop_grad
        trainable = set()
        try:
            for n, p in self.named_parameters():
                if not p.is_stop_grad():
                    trainable.add(n)
        except Exception:
            pass
        _orig_load_state_dict(self, state_dict)
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
    _orig_parameters = M.parameters
    def _parameters(self, recurse=True):
        return _ParamList(_orig_parameters(self, recurse=recurse))
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
        _set_is_train(self, mode)
        return self
    M.train = _train
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
            return _AutocastContext()
        GradScaler = _GradScaler
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
    cuda.memory_reserved = lambda *a, **k: 0
    cuda.max_memory_reserved = lambda *a, **k: 0
    cuda.memory_cached = lambda *a, **k: 0
    cuda.reset_peak_memory_stats = lambda *a, **k: None
    cuda.reset_max_memory_allocated = lambda *a, **k: None
    cuda.memory_stats = lambda *a, **k: {}
    cuda.mem_get_info = lambda *a, **k: (64*1024**3, 64*1024**3)
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

    # torch's new_*(size, *, dtype=, device=, requires_grad=) factory methods.
    # jittor's native new_ones/new_zeros only take a size, so override to accept
    # torch kwargs (dtype defaults to self's dtype, like torch).
    def _norm_size(args):
        # torch allows new_ones(2,3) or new_ones((2,3))
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return tuple(args)
    def _new_ones(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return jt.ones(_norm_size(size), dt)
    def _new_zeros(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return jt.zeros(_norm_size(size), dt)
    def _new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return jt.full(tuple(size) if isinstance(size, (tuple, list)) else (size,), fill_value).cast(dt)
    def _new_empty(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return jt.empty(_norm_size(size), dt)
    def _new_tensor(self, data, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return jt.array(data).cast(dt)
    Var.new_ones = _new_ones
    Var.new_zeros = _new_zeros
    Var.new_full = _new_full
    Var.new_empty = _new_empty
    Var.new_tensor = _new_tensor
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
    def _cumsum(x, dim=-1, dtype=None, out=None, **kw):
        if isinstance(x, jt.Var) and str(x.dtype) in ("bool", "uint8"):
            x = x.cast("int64")
        r = _native_cumsum(x, dim)
        if dtype is not None:
            r = r.cast(_dtype_to_str(dtype))
        return r
    g.cumsum = _cumsum
    Var.cumsum = lambda self, dim=-1, dtype=None, **kw: _cumsum(self, dim, dtype)
    # cumprod has the same ACL fragility; guard it the same way if present.
    if hasattr(jt, "cumprod"):
        _native_cumprod = jt.cumprod
        def _cumprod(x, dim=-1, dtype=None, out=None, **kw):
            if isinstance(x, jt.Var) and str(x.dtype) in ("bool", "uint8"):
                x = x.cast("int64")
            r = _native_cumprod(x, dim)
            if dtype is not None:
                r = r.cast(_dtype_to_str(dtype))
            return r
        g.cumprod = _cumprod
        Var.cumprod = lambda self, dim=-1, dtype=None, **kw: _cumprod(self, dim, dtype)

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

    # torch's Tensor.data returns a detached *tensor* (and is assignable:
    # `param.data = new_tensor`). jittor's native Var.data returns a numpy
    # ndarray, breaking `param.data.to(...)`. Override to torch semantics.
    if not getattr(Var, "_data_wrapped", False):
        def _data_get(self):
            return self.detach() if hasattr(self, "detach") else self
        def _data_set(self, value):
            src = value if isinstance(value, Var) else jt.array(value)
            was_trainable = not self.is_stop_grad()
            self.assign(src)
            if was_trainable:
                self.start_grad()
        Var.data = property(_data_get, _data_set)
        Var._data_wrapped = True

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
    def _backward(self, gradient=None, retain_graph=False, create_graph=False, **kw):
        opt = getattr(jt, "_current_optimizer", None)
        if opt is None:
            raise RuntimeError(
                "loss.backward() needs an active optimizer on the jittor backend; "
                "none was registered (construct a torch.optim optimizer first).")
        # compute & accumulate grads into the optimizer (jittor semantics)
        opt.backward(self, retain_graph)
        # publish grads onto params so `param.grad` works for clipping/logging
        try:
            opt._build_grad_map()
            for pg in opt.param_groups:
                for p, g in zip(pg["params"], pg.get("grads", [])):
                    object.__setattr__(p, "_torch_grad", g)
        except Exception:
            pass
        return None
    Var.backward = _backward

    def _grad_get(self):
        # prefer the optimizer-held grad Var (mutations propagate to step())
        opt = getattr(jt, "_current_optimizer", None)
        if opt is not None:
            try:
                return opt.find_grad(self)
            except Exception:
                pass
        return getattr(self, "_torch_grad", None)
    def _grad_set(self, value):
        object.__setattr__(self, "_torch_grad", value)
        # also write through to the optimizer's stored grad so step() uses it
        opt = getattr(jt, "_current_optimizer", None)
        if opt is not None and value is not None:
            try:
                opt.find_grad(self).update(value)
            except Exception:
                pass
    Var.grad = property(_grad_get, _grad_set)

    # torch's `is_leaf`: True for tensors not produced by a grad-tracked op
    # (user-created params/inputs). jittor has no autograd-graph leaf concept;
    # treat every Var as a leaf so peft's `if param.is_leaf:` guards pass.
    if not hasattr(Var, "is_leaf"):
        Var.is_leaf = property(lambda self: True)
    # torch's `grad_fn` is None for leaves; libs check `t.grad_fn is None`.
    if not hasattr(Var, "grad_fn"):
        Var.grad_fn = property(lambda self: None)
    # `retain_grad()` is a no-op for us (all Vars retain grad)
    if not hasattr(Var, "retain_grad"):
        Var.retain_grad = lambda self: self

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
            return _native_squeeze(self)
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

    # ---- save / load ----
    # torch.save must handle BOTH tensor state-dicts AND arbitrary Python
    # objects (e.g. TrainingArguments). jittor's jt.save is numpy/pickle based
    # but chokes on live Vars and some objects, so we use standard pickle with
    # Vars converted to a portable (numpy) form and restored on load.
    import pickle as _pickle
    _VAR_TAG = "__jt_var__"
    def _to_portable(obj, _seen=None):
        if isinstance(obj, jt.Var):
            return {_VAR_TAG: True, "data": obj.numpy(), "dtype": str(obj.dtype)}
        if isinstance(obj, dict):
            return {k: _to_portable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_to_portable(v) for v in obj)
        return obj
    def _from_portable(obj):
        if isinstance(obj, dict):
            if obj.get(_VAR_TAG):
                return jt.array(obj["data"])
            return {k: _from_portable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_from_portable(v) for v in obj)
        return obj
    def save(obj, f, *a, **k):
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
    def load(f, *a, **k):
        # accept map_location/weights_only/pickle_module kwargs (ignored).
        # Real torch .pt is a zip archive -> use the torch-format loader;
        # our own torch.save output is plain pickle -> _from_portable.
        try:
            if _is_zip(f):
                return _load_torch_pt(f)
        except Exception as _e:
            pass
        if hasattr(f, "read"):
            obj = _pickle.load(f)
        else:
            with open(f, "rb") as fh:
                obj = _pickle.load(fh)
        return _from_portable(obj)
    g.save = save
    g.load = load

    # ---- elementwise / reduction helpers that may be missing ----
    def _alias(name, fn):
        if not hasattr(g, name):
            setattr(g, name, fn)
    _alias("rsqrt", lambda x: 1.0 / jt.sqrt(x))
    _alias("empty_like", lambda x, **k: jt.empty(x.shape, x.dtype))
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
    _alias("diff", lambda x, n=1, dim=-1, prepend=None, append=None:
           _diff(x, n=n, dim=dim, prepend=prepend, append=append))
    _alias("repeat_interleave", _repeat_interleave)
    _alias("autocast", lambda *a, **k: _AutocastContext())
    _alias("vmap", lambda fn, *a, **k: fn)
    _alias("outer", lambda a, b: jt.matmul(a.reshape(-1, 1), b.reshape(1, -1)))
    _alias("isin", _isin)
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


def _diff(x, n=1, dim=-1, prepend=None, append=None):
    # torch.diff(input, n=1, dim=-1, prepend=None, append=None): prepend/append are
    # concatenated along `dim` before differencing (used by transformers' packed-
    # sequence detection via torch.diff(position_ids, prepend=..., dim=-1)).
    import jittor as _jt
    if prepend is not None or append is not None:
        parts = []
        if prepend is not None:
            parts.append(prepend if isinstance(prepend, _jt.Var) else _jt.array(prepend))
        parts.append(x)
        if append is not None:
            parts.append(append if isinstance(append, _jt.Var) else _jt.array(append))
        x = _jt.concat(parts, dim=dim)
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

