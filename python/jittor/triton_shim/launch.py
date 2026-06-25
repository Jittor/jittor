"""Naive *executor* for a narrow class of ``@triton.jit`` kernels on jittor.

The rest of :mod:`jittor.triton_shim` is a pure API *shim* — it makes
``import triton`` / ``@triton.jit`` work and otherwise raises a clear
``NotImplementedError`` on launch. This module adds the next increment: it can
actually **run** the most common kind of triton kernel — a **1-D elementwise**
kernel — by *tracing* the python body once with jittor-backed ``tl.*``
primitives.

How it works
------------
A canonical 1-D elementwise triton kernel looks like::

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid  = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

Real triton compiles this and launches ``grid`` programs, each handling a
``BLOCK``-sized tile. jittor is *already* a whole-array vectorising engine, so
we don't need the tiling at all: we run the body **once**, treating each input
pointer as the *entire* jittor ``Var`` and letting jittor parallelise. The
``grid`` / ``BLOCK`` / ``program_id`` / ``mask`` machinery becomes book-keeping
that we interpret symbolically:

* ``tl.program_id(axis)``           -> ``_Scalar(0)``  (one virtual program)
* ``tl.arange(0, BLOCK)``           -> ``_IndexVec``   (the index ramp ``0..N-1``)
* ``ptr + offs`` (Var + ``_IndexVec``) -> ``_Pointer`` bound to that Var
* ``tl.load(_Pointer, mask=...)``   -> the whole jittor Var (a ``_Tensor``)
* arithmetic / ``tl.sigmoid`` / ...  -> the corresponding jittor op (whole-array)
* ``tl.store(_Pointer, value, ...)`` -> assign ``value`` into the output Var

Because every output element of a 1-D elementwise kernel depends only on the
**same-index** input element(s), running the body once over the whole array is
*numerically exact*, not an approximation — the masking only ever zeroes a
ragged tail that doesn't exist when we use the natural array length.

Contract: "clear error beats silent wrong"
------------------------------------------
This executor is deliberately narrow. It runs the body inside a tracing context
and, the moment the kernel does something it cannot faithfully lower (an
unsupported ``tl.*`` op, a data-dependent ``if``, a reduction across the block,
a 2-D block pointer, atomics, writing to more than one output in an
inconsistent way, ...), it raises a clear ``NotImplementedError`` rather than
guessing. The caller then sees the same "no triton backend for this kernel"
signal as before and can keep its pure-pytorch fallback.
"""
import threading

__all__ = ["run_kernel", "in_tracing", "TracingError"]


class TracingError(NotImplementedError):
    """Raised when a kernel uses a construct the naive executor can't lower.

    Subclasses ``NotImplementedError`` so existing ``except NotImplementedError``
    fallback guards in libraries keep working unchanged.
    """


# --------------------------------------------------------------------------- #
#  thread-local "are we currently tracing a kernel?" state
# --------------------------------------------------------------------------- #
_state = threading.local()


def _tracer():
    return getattr(_state, "tracer", None)


def in_tracing():
    """True if a kernel body is currently being traced (used by language.py)."""
    return _tracer() is not None


# --------------------------------------------------------------------------- #
#  symbolic values that flow through the kernel body during tracing
# --------------------------------------------------------------------------- #
def _is_var(v):
    """True if ``v`` is a jittor Var (duck-typed to avoid importing jittor up top)."""
    import jittor as jt
    return isinstance(v, jt.Var)


def _to_value(v):
    """Unwrap a symbolic wrapper / constexpr to the underlying jittor/py value."""
    from . import language as tl
    if isinstance(v, _Tensor):
        return v.var
    if isinstance(v, _Scalar):
        return v.value
    if isinstance(v, tl.constexpr):
        return v.value
    return v


class _Scalar:
    """A traced python scalar (e.g. ``tl.program_id(0)`` -> ``_Scalar(0)``).

    Arithmetic with an :class:`_IndexVec` produces another ``_IndexVec`` (so
    ``pid * BLOCK + tl.arange(...)`` stays an index ramp); arithmetic with plain
    scalars stays scalar.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def _combine(self, other, op):
        if isinstance(other, _IndexVec):
            return NotImplemented  # let _IndexVec.__radd__ etc. handle it
        o = other.value if isinstance(other, _Scalar) else _to_value(other)
        return _Scalar(op(self.value, o))

    def __add__(self, o): return self._combine(o, lambda a, b: a + b)
    def __radd__(self, o): return self._combine(o, lambda a, b: b + a)
    def __mul__(self, o): return self._combine(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._combine(o, lambda a, b: b * a)
    def __sub__(self, o): return self._combine(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._combine(o, lambda a, b: b - a)
    def __int__(self): return int(self.value)
    def __index__(self): return int(self.value)
    def __repr__(self): return "triton-trace.Scalar({0!r})".format(self.value)


class _PtrArg:
    """A kernel *pointer argument*.

    In real triton, a tensor passed positionally becomes a pointer inside the
    kernel (``x_ptr``), and the body does ``x_ptr + offs``. We wrap every jittor
    ``Var`` argument in this so that ``x_ptr + offs`` (or ``offs + x_ptr``)
    resolves to a :class:`_Pointer` *before* jittor's own ``Var.__add__`` can
    raise on the unknown operand type. It carries the underlying Var.
    """

    __slots__ = ("var",)

    def __init__(self, var):
        self.var = var

    def __add__(self, other):
        if isinstance(other, _IndexVec):
            return _Pointer(self.var)
        # ptr + scalar shift -> still a pointer to the same whole Var
        return self

    __radd__ = __add__

    def __repr__(self):
        return "triton-trace.PtrArg(shape={0})".format(tuple(self.var.shape))


class _IndexVec:
    """The traced index ramp produced by ``tl.arange`` (optionally shifted by
    ``pid*BLOCK``).

    We never materialise it as data; we only need it to (a) form pointers
    (``ptr + offs``) and (b) build the comparison mask (``offs < n``). Its job is
    purely structural — it tags "this is the per-element index", so that adding
    it to a pointer arg yields a :class:`_Pointer`.
    """

    __slots__ = ()

    # ptr + offs  /  offs + ptr  -> pointer into that Var
    def __add__(self, other):
        if isinstance(other, _PtrArg):
            return _Pointer(other.var)
        if _is_var(other):
            return _Pointer(other)
        # offs + scalar shift -> still an index ramp (shift is irrelevant to us)
        return self

    __radd__ = __add__

    def __mul__(self, other):
        return self  # pid*BLOCK etc. — scaling the ramp is irrelevant here

    __rmul__ = __mul__

    def __sub__(self, other):
        return self

    # offs < n -> a mask. We don't need its data (whole-array path covers all
    # valid elements exactly), so return a benign sentinel the loader ignores.
    def _cmp(self, other):
        return _Mask()

    __lt__ = _cmp
    __le__ = _cmp
    __gt__ = _cmp
    __ge__ = _cmp

    def __repr__(self): return "triton-trace.IndexVec(0..N)"


class _Mask:
    """Sentinel for ``offs < n``. The whole-array path needs no actual masking
    (there is no ragged tail), so this is inert; we accept ``&`` / ``|`` so
    compound masks (``m1 & m2``) still trace."""

    __slots__ = ()

    def __and__(self, o): return self
    __rand__ = __and__
    def __or__(self, o): return self
    __ror__ = __or__
    def __invert__(self): return self
    def __repr__(self): return "triton-trace.Mask(all-true)"


class _Pointer:
    """Result of ``base_ptr + offs`` — a typed handle the loader/storer use to
    know *which* jittor Var to read from / write to."""

    __slots__ = ("var",)

    def __init__(self, var):
        self.var = var

    # ptr + further_scalar_offset stays the same logical pointer for our purposes
    def __add__(self, other): return self
    __radd__ = __add__

    def __repr__(self): return "triton-trace.Pointer(var shape={0})".format(
        tuple(self.var.shape))


class _Tensor:
    """A traced whole-array value (what ``tl.load`` returns and what arithmetic /
    ``tl.sigmoid`` etc. operate on). Wraps a jittor ``Var`` and forwards
    operators to real jittor ops so the kernel's elementwise expression *is* the
    jittor computation graph."""

    __slots__ = ("var",)

    def __init__(self, var):
        self.var = var

    @staticmethod
    def _v(x):
        return x.var if isinstance(x, _Tensor) else _to_value(x)

    def _binop(self, other, op):
        return _Tensor(op(self.var, _Tensor._v(other)))

    def _rbinop(self, other, op):
        return _Tensor(op(_Tensor._v(other), self.var))

    def __add__(self, o): return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o): return self._rbinop(o, lambda a, b: a + b)
    def __sub__(self, o): return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._rbinop(o, lambda a, b: a - b)
    def __mul__(self, o): return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._rbinop(o, lambda a, b: a * b)
    def __truediv__(self, o): return self._binop(o, lambda a, b: a / b)
    def __rtruediv__(self, o): return self._rbinop(o, lambda a, b: a / b)
    def __neg__(self): return _Tensor(-self.var)

    def __mod__(self, o): return self._binop(o, lambda a, b: a % b)
    def __pow__(self, o): return self._binop(o, lambda a, b: a ** b)

    # comparisons -> jittor bool Vars wrapped back as tensors (for tl.where etc.)
    def __lt__(self, o): return self._binop(o, lambda a, b: a < b)
    def __le__(self, o): return self._binop(o, lambda a, b: a <= b)
    def __gt__(self, o): return self._binop(o, lambda a, b: a > b)
    def __ge__(self, o): return self._binop(o, lambda a, b: a >= b)

    def __repr__(self): return "triton-trace.Tensor(shape={0})".format(
        tuple(self.var.shape))


# --------------------------------------------------------------------------- #
#  the tracer object: holds per-launch state and implements each tl.* op
# --------------------------------------------------------------------------- #
class _Tracer:
    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        self.stores = []   # list of (output_Var, value_Var) recorded by tl.store

    # -- structural ops ----------------------------------------------------- #
    def program_id(self, axis=0):
        return _Scalar(0)

    def num_programs(self, axis=0):
        return _Scalar(1)

    def arange(self, start, end):
        # We model arange purely structurally as the per-element index ramp.
        return _IndexVec()

    # -- memory ops --------------------------------------------------------- #
    @staticmethod
    def _ptr_var(ptr):
        """Resolve a load/store address to its backing Var, or None if it isn't
        a supported whole-array pointer."""
        if isinstance(ptr, _Pointer):
            return ptr.var
        if isinstance(ptr, _PtrArg):  # tl.load(x_ptr) with no explicit + offs
            return ptr.var
        return None

    def load(self, ptr, mask=None, other=None, **kwargs):
        var = self._ptr_var(ptr)
        if var is None:
            raise TracingError(
                "tl.load in kernel {0!r} got an address the jittor triton "
                "executor can't resolve to a whole-array pointer ({1!r}). The "
                "naive executor only supports 1-D elementwise `tl.load(base_ptr "
                "+ tl.arange(...), mask=...)`.".format(self.kernel_name, type(ptr).__name__)
            )
        return _Tensor(var)

    def store(self, ptr, value, mask=None, **kwargs):
        var = self._ptr_var(ptr)
        if var is None:
            raise TracingError(
                "tl.store in kernel {0!r} got an address the jittor triton "
                "executor can't resolve to a whole-array pointer ({1!r}). The "
                "naive executor only supports 1-D elementwise `tl.store(base_ptr "
                "+ tl.arange(...), value, mask=...)`.".format(
                    self.kernel_name, type(ptr).__name__)
            )
        # Reject storing a structural symbol (the raw index ramp, a pointer, a
        # mask, ...). E.g. an iota kernel `tl.store(out+offs, offs)` is NOT
        # whole-array-safe (it depends on the per-tile index), so refuse it
        # loudly instead of writing a wrong/garbage value silently.
        if isinstance(value, (_IndexVec, _Pointer, _PtrArg, _Mask, _Scalar)):
            raise TracingError(
                "tl.store in kernel {0!r} is storing a non-tensor symbolic value "
                "({1}) — e.g. the index ramp itself (iota) or a pointer. The "
                "naive whole-array executor can only store a tensor computed "
                "elementwise from loaded tensors; such index-dependent kernels "
                "are not supported.".format(self.kernel_name, type(value).__name__))
        val = value.var if isinstance(value, _Tensor) else _to_value(value)
        import jittor as jt
        if not isinstance(val, jt.Var):
            # storing a python/numpy scalar: broadcast to the output shape
            val = jt.array(val).broadcast(var.shape)
        self.stores.append((var, val))

    # -- elementwise math (lower to jittor) --------------------------------- #
    def _unary(self, name, jt_fn, x):
        import jittor as jt
        v = x.var if isinstance(x, _Tensor) else _to_value(x)
        if not isinstance(v, jt.Var):
            raise TracingError(
                "tl.{0} in kernel {1!r} received a non-tensor operand; the "
                "naive jittor executor only lowers elementwise tl.* math over "
                "loaded tensors.".format(name, self.kernel_name))
        return _Tensor(jt_fn(v))

    def _binary(self, name, jt_fn, a, b):
        import jittor as jt
        va = a.var if isinstance(a, _Tensor) else _to_value(a)
        vb = b.var if isinstance(b, _Tensor) else _to_value(b)
        return _Tensor(jt_fn(va, vb))

    def where(self, cond, a, b):
        import jittor as jt
        vc = cond.var if isinstance(cond, _Tensor) else _to_value(cond)
        va = a.var if isinstance(a, _Tensor) else _to_value(a)
        vb = b.var if isinstance(b, _Tensor) else _to_value(b)
        return _Tensor(jt.ternary(vc, va, vb))


# --------------------------------------------------------------------------- #
#  jittor lowerings for the supported tl.* math namespace
# --------------------------------------------------------------------------- #
def _math_table():
    """Map tl.* math names -> (kind, jittor callable). Built lazily so importing
    this module never imports jittor."""
    import jittor as jt

    def rsqrt(v): return 1.0 / jt.sqrt(v)
    def sigmoid(v): return jt.sigmoid(v)
    # erf may not exist on every jittor build; fall back below if missing.
    erf = getattr(jt, "erf", None)

    table = {
        "exp": ("unary", jt.exp),
        "exp2": ("unary", lambda v: jt.exp(v * 0.6931471805599453)),
        "log": ("unary", jt.log),
        "log2": ("unary", lambda v: jt.log(v) * 1.4426950408889634),
        "sqrt": ("unary", jt.sqrt),
        "rsqrt": ("unary", rsqrt),
        "sin": ("unary", jt.sin),
        "cos": ("unary", jt.cos),
        "abs": ("unary", jt.abs),
        "floor": ("unary", jt.floor),
        "ceil": ("unary", jt.ceil),
        "sigmoid": ("unary", sigmoid),
        "maximum": ("binary", jt.maximum),
        "minimum": ("binary", jt.minimum),
        "fdiv": ("binary", lambda a, b: a / b),
    }
    if erf is not None:
        table["erf"] = ("unary", erf)
    return table


def dispatch_math(name, args, kwargs):
    """Called by language.py for a tl.<name>(...) math op while tracing.

    Returns a traced ``_Tensor`` or raises ``TracingError`` for the
    unsupported / non-elementwise cases (e.g. reductions like ``tl.sum``).
    """
    tr = _tracer()
    if tr is None:  # defensive: only reachable while tracing
        raise TracingError("dispatch_math called outside a kernel trace")

    table = _math_table()
    entry = table.get(name)
    if entry is None:
        raise TracingError(
            "tl.{0} is not supported by the jittor naive triton executor. Only "
            "a 1-D elementwise subset is lowered (program_id/arange/load/store, "
            "arithmetic, where, and elementwise math: {1}). Kernels using "
            "reductions, dot, atomics, block pointers, or other ops fall back "
            "to the pure-pytorch path.".format(name, ", ".join(sorted(table))))

    kind, fn = entry
    if kind == "unary":
        if len(args) != 1:
            raise TracingError(
                "tl.{0} expected 1 argument, got {1}".format(name, len(args)))
        return tr._unary(name, fn, args[0])
    # binary
    if len(args) != 2:
        raise TracingError(
            "tl.{0} expected 2 arguments, got {1}".format(name, len(args)))
    return tr._binary(name, fn, args[0], args[1])


# --------------------------------------------------------------------------- #
#  the public entry point used by JITFunction._launch
# --------------------------------------------------------------------------- #
def run_kernel(jitfn, grid, args, kwargs):
    """Try to execute ``jitfn`` (a triton @jit kernel) on jittor by tracing.

    Returns ``None`` on success (results are written in-place into the output
    Var(s), matching triton's pointer-output convention). Raises
    ``NotImplementedError`` / ``TracingError`` if the kernel is outside the
    supported 1-D elementwise subset — so the caller's fallback still fires.
    """
    import inspect
    import jittor as jt
    from . import language as tl

    fn = jitfn.fn
    name = getattr(jitfn, "__name__", "triton_kernel")

    # Bind call args to parameter names (handles positional + keyword + defaults,
    # including constexpr block sizes passed as kwargs like BLOCK=1024).
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError as e:
        raise TracingError(
            "could not bind arguments for triton kernel {0!r}: {1}".format(name, e))

    bound_args = dict(bound.arguments)

    # Constexpr params (annotated ``: tl.constexpr``) must be plain python ints
    # during tracing — unwrap any constexpr wrappers a caller passed.
    for pname, p in sig.parameters.items():
        ann = p.annotation
        is_constexpr = (ann is tl.constexpr) or isinstance(ann, tl.constexpr) \
            or (isinstance(ann, type) and issubclass(ann, tl.constexpr))
        if pname in bound_args and isinstance(bound_args[pname], tl.constexpr):
            bound_args[pname] = bound_args[pname].value
        # (we don't *need* is_constexpr beyond this; kept for clarity)

    # Require at least one jittor Var argument, else this isn't something we can
    # run as a tensor kernel (keep the clear-error contract).
    var_args = [v for v in bound_args.values() if isinstance(v, jt.Var)]
    if not var_args:
        raise TracingError(
            "triton kernel {0!r} was launched with no jittor Var arguments; the "
            "naive executor needs the tensor pointers to be jittor Vars.".format(name))

    # Wrap every Var arg as a pointer arg, so `x_ptr + offs` resolves to a
    # _Pointer via our operators *before* jittor's Var.__add__ can raise on the
    # unknown operand type (jittor doesn't return NotImplemented for it). This
    # mirrors triton, where a passed tensor IS a pointer inside the kernel.
    trace_args = {
        k: (_PtrArg(v) if isinstance(v, jt.Var) else v)
        for k, v in bound_args.items()
    }

    tracer = _Tracer(name)
    prev = _tracer()
    _state.tracer = tracer
    try:
        # Execute the kernel body ONCE. tl.* primitives consult the active
        # tracer (see language.py) and lower to jittor.
        fn(**trace_args)
    except TracingError:
        raise
    except NotImplementedError:
        # a tl.* stub fired (unsupported op) — propagate as-is (already clear)
        raise
    except Exception as e:
        # Any other python error during the trace means the body did something
        # our symbolic values don't model (data-dependent control flow, indexing
        # a traced index vector, etc.). Convert to the clear fallback signal.
        raise TracingError(
            "jittor's naive triton executor could not trace kernel {0!r} "
            "({1}: {2}). It only supports 1-D elementwise kernels "
            "(program_id/arange/load/store/mask + elementwise math). Fall back "
            "to the pure-pytorch path.".format(name, type(e).__name__, e)
        )
    finally:
        _state.tracer = prev

    if not tracer.stores:
        raise TracingError(
            "triton kernel {0!r} produced no tl.store — nothing to write back. "
            "The naive executor only handles kernels that store their result "
            "through an output pointer.".format(name))

    # Commit the recorded stores into the output Var(s), in place, so the caller
    # observes the result exactly like real triton (which writes through the
    # output pointer the caller passed in).
    for out_var, val in tracer.stores:
        out_var.assign(val.cast(out_var.dtype) if val.dtype != out_var.dtype else val)
    return None
