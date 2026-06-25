"""``triton.language`` (``tl``) compatibility surface for the jittor torch shim.

This module provides the *names* that ``@triton.jit`` kernels reference at
definition time so that importing a triton-using library (transformers,
``flash-attn`` style kernels, ...) does not crash on a missing
``triton.language`` symbol.

It is **not** a triton compiler. The body of a ``@triton.jit`` kernel is never
actually traced/compiled by this shim, so the in-kernel primitives below
(``tl.load``, ``tl.store``, ``tl.dot``, ...) are *stubs*: if Python ever
executes one of them (i.e. someone called the kernel as a plain function or the
shim's naive interpreter reached it) they raise a **clear**
``NotImplementedError`` rather than failing with a confusing ``AttributeError``
or segfaulting.

A small number of symbols are genuinely host-callable in real triton and are
implemented for real here:

* ``tl.constexpr``      — a typing marker, usable as ``BLOCK: tl.constexpr``.
* ``tl.cdiv(a, b)``     — ceiling division (also exposed as ``triton.cdiv``).
* ``tl.max`` / ``tl.min`` / ``tl.minimum`` / ``tl.maximum`` — work on plain
  Python scalars (real triton also accepts tensors; that path is a stub).
* the dtype objects (``tl.float32`` ...) — lightweight value objects so code
  like ``acc = tl.zeros(..., dtype=tl.float32)`` parses and the dtype can be
  inspected / compared.

Everything else is an "in-kernel only" stub.
"""

__all__ = [
    # markers / dtypes
    "constexpr", "dtype",
    "void", "int1", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float8e4nv", "float8e5", "float16", "bfloat16", "float32", "float64",
    "pi32_t",
    # host-usable helpers
    "cdiv", "max", "min", "minimum", "maximum",
    # in-kernel primitives (stubs)
    "program_id", "num_programs", "arange", "load", "store",
    "make_block_ptr", "advance",
    "dot", "trans", "where", "zeros", "zeros_like", "full", "broadcast_to",
    "reshape", "view", "ravel", "expand_dims", "cat",
    "sum", "prod", "cumsum", "cumprod", "argmax", "argmin",
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt", "sin", "cos",
    "abs", "sigmoid", "softmax", "erf", "floor", "ceil", "fma",
    "atomic_add", "atomic_max", "atomic_min", "atomic_cas", "atomic_xchg",
    "atomic_and", "atomic_or", "atomic_xor",
    "debug_barrier", "multiple_of", "max_contiguous", "static_assert",
    "static_print", "device_print", "load_scalar", "associative_scan",
    "reduce", "histogram", "sort", "flip", "interleave", "join", "split",
    "dot_scaled", "clamp", "fdiv", "div_rn", "umulhi",
]

__triton_shim__ = True


def _in_kernel_only(name):
    """Build a stub callable for an in-kernel-only ``tl`` primitive.

    Outside a kernel trace it raises the clear "only usable inside @triton.jit"
    error (unchanged behaviour). *Inside* a trace (i.e. while
    :func:`jittor.triton_shim.launch.run_kernel` is executing the body), the
    structural / memory primitives delegate to the active tracer, and the math
    primitives lower to jittor. Anything the executor doesn't model still raises
    a clear ``NotImplementedError`` — never a silent wrong result.
    """

    def _stub(*args, **kwargs):
        from . import launch as _launch
        tr = _launch._tracer()
        if tr is not None:
            handler = getattr(tr, name, None)
            if handler is not None:
                # structural / memory ops implemented directly on the tracer
                return handler(*args, **kwargs)
            # otherwise it's an elementwise-math op: lower via the math table
            return _launch.dispatch_math(name, args, kwargs)
        raise NotImplementedError(
            "triton.language.{0} is only usable inside an @triton.jit kernel, "
            "which the jittor triton shim does not execute as a triton kernel. "
            "(The shim can run a narrow 1-D elementwise subset via "
            "`kernel[grid](var_args...)`; calling tl.{0} on its own is not "
            "supported.) Use the pure-PyTorch fallback path.".format(name)
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = (
        "Stub for triton.language.{0} (in-kernel only; outside an @triton.jit "
        "trace it raises NotImplementedError on jittor).".format(name)
    )
    _stub.__triton_shim_stub__ = True
    return _stub


# --------------------------------------------------------------------------- #
#  constexpr — kernel compile-time constant marker
# --------------------------------------------------------------------------- #
class constexpr:
    """Compile-time constant marker (``BLOCK_SIZE: tl.constexpr``).

    Real triton uses this both as a type annotation and as a wrapper carrying a
    value. We support both: ``tl.constexpr`` alone works as an annotation, and
    ``tl.constexpr(value)`` wraps a value while forwarding common operations so
    arithmetic in kernel signatures / defaults does not blow up.
    """

    def __init__(self, value=None):
        self.value = value

    # let it stand in for its value in the common arithmetic / comparison cases
    def __index__(self):
        return int(self.value)

    def __int__(self):
        return int(self.value)

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, other):
        ov = other.value if isinstance(other, constexpr) else other
        return self.value == ov

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return "constexpr[{0!r}]".format(self.value)


# --------------------------------------------------------------------------- #
#  dtypes — lightweight value objects
# --------------------------------------------------------------------------- #
class dtype:
    """Lightweight stand-in for a ``triton.language`` dtype object."""

    def __init__(self, name, itemsize=None):
        self.name = name
        self.itemsize = itemsize

    def __repr__(self):
        return "triton.language.{0}".format(self.name)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, dtype) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


void = dtype("void", 0)
int1 = dtype("int1", 1)
int8 = dtype("int8", 1)
int16 = dtype("int16", 2)
int32 = dtype("int32", 4)
int64 = dtype("int64", 8)
uint8 = dtype("uint8", 1)
uint16 = dtype("uint16", 2)
uint32 = dtype("uint32", 4)
uint64 = dtype("uint64", 8)
float8e4nv = dtype("float8e4nv", 1)
float8e5 = dtype("float8e5", 1)
float16 = dtype("float16", 2)
bfloat16 = dtype("bfloat16", 2)
float32 = dtype("float32", 4)
float64 = dtype("float64", 8)
# alias triton exposes for the program-id integer type
pi32_t = int32


# --------------------------------------------------------------------------- #
#  Host-usable helpers (real implementations)
# --------------------------------------------------------------------------- #
def cdiv(x, div):
    """Ceiling division ``ceil(x / div)`` — host-usable, matches ``triton.cdiv``."""
    return -(-int(x) // int(div))


def _is_scalar(v):
    return isinstance(v, (int, float)) or isinstance(v, constexpr)


def _scalar(v):
    return v.value if isinstance(v, constexpr) else v


def maximum(a, b):
    """Elementwise max. Scalar inputs are supported; tensor inputs are a stub."""
    if _is_scalar(a) and _is_scalar(b):
        import builtins
        return builtins.max(_scalar(a), _scalar(b))
    return _in_kernel_only("maximum")(a, b)


def minimum(a, b):
    """Elementwise min. Scalar inputs are supported; tensor inputs are a stub."""
    if _is_scalar(a) and _is_scalar(b):
        import builtins
        return builtins.min(_scalar(a), _scalar(b))
    return _in_kernel_only("minimum")(a, b)


def max(*args, **kwargs):
    """``tl.max``: host scalar reduction works; tensor reduction is a stub."""
    if kwargs:
        return _in_kernel_only("max")(*args, **kwargs)
    if len(args) >= 2 and all(_is_scalar(a) for a in args):
        import builtins
        return builtins.max(*[_scalar(a) for a in args])
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and all(_is_scalar(a) for a in args[0]):
        import builtins
        return builtins.max(_scalar(a) for a in args[0])
    return _in_kernel_only("max")(*args, **kwargs)


def min(*args, **kwargs):
    """``tl.min``: host scalar reduction works; tensor reduction is a stub."""
    if kwargs:
        return _in_kernel_only("min")(*args, **kwargs)
    if len(args) >= 2 and all(_is_scalar(a) for a in args):
        import builtins
        return builtins.min(*[_scalar(a) for a in args])
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and all(_is_scalar(a) for a in args[0]):
        import builtins
        return builtins.min(_scalar(a) for a in args[0])
    return _in_kernel_only("min")(*args, **kwargs)


# --------------------------------------------------------------------------- #
#  In-kernel-only primitives — clear NotImplementedError stubs
# --------------------------------------------------------------------------- #
program_id = _in_kernel_only("program_id")
num_programs = _in_kernel_only("num_programs")
arange = _in_kernel_only("arange")
load = _in_kernel_only("load")
store = _in_kernel_only("store")
make_block_ptr = _in_kernel_only("make_block_ptr")
advance = _in_kernel_only("advance")
dot = _in_kernel_only("dot")
dot_scaled = _in_kernel_only("dot_scaled")
trans = _in_kernel_only("trans")
where = _in_kernel_only("where")
zeros = _in_kernel_only("zeros")
zeros_like = _in_kernel_only("zeros_like")
full = _in_kernel_only("full")
broadcast_to = _in_kernel_only("broadcast_to")
reshape = _in_kernel_only("reshape")
view = _in_kernel_only("view")
ravel = _in_kernel_only("ravel")
expand_dims = _in_kernel_only("expand_dims")
cat = _in_kernel_only("cat")
sum = _in_kernel_only("sum")
prod = _in_kernel_only("prod")
cumsum = _in_kernel_only("cumsum")
cumprod = _in_kernel_only("cumprod")
argmax = _in_kernel_only("argmax")
argmin = _in_kernel_only("argmin")
exp = _in_kernel_only("exp")
exp2 = _in_kernel_only("exp2")
log = _in_kernel_only("log")
log2 = _in_kernel_only("log2")
sqrt = _in_kernel_only("sqrt")
rsqrt = _in_kernel_only("rsqrt")
sin = _in_kernel_only("sin")
cos = _in_kernel_only("cos")
abs = _in_kernel_only("abs")
sigmoid = _in_kernel_only("sigmoid")
softmax = _in_kernel_only("softmax")
erf = _in_kernel_only("erf")
floor = _in_kernel_only("floor")
ceil = _in_kernel_only("ceil")
fma = _in_kernel_only("fma")
clamp = _in_kernel_only("clamp")
fdiv = _in_kernel_only("fdiv")
div_rn = _in_kernel_only("div_rn")
umulhi = _in_kernel_only("umulhi")
atomic_add = _in_kernel_only("atomic_add")
atomic_max = _in_kernel_only("atomic_max")
atomic_min = _in_kernel_only("atomic_min")
atomic_cas = _in_kernel_only("atomic_cas")
atomic_xchg = _in_kernel_only("atomic_xchg")
atomic_and = _in_kernel_only("atomic_and")
atomic_or = _in_kernel_only("atomic_or")
atomic_xor = _in_kernel_only("atomic_xor")
debug_barrier = _in_kernel_only("debug_barrier")
multiple_of = _in_kernel_only("multiple_of")
max_contiguous = _in_kernel_only("max_contiguous")
load_scalar = _in_kernel_only("load_scalar")
associative_scan = _in_kernel_only("associative_scan")
reduce = _in_kernel_only("reduce")
histogram = _in_kernel_only("histogram")
sort = _in_kernel_only("sort")
flip = _in_kernel_only("flip")
interleave = _in_kernel_only("interleave")
join = _in_kernel_only("join")
split = _in_kernel_only("split")


# These two are harmless no-ops / passthroughs in real triton's host-trace and
# are sometimes evaluated at definition time, so keep them lenient.
def static_assert(cond, msg=""):
    """Compile-time assert. Lenient: raises only on a *statically* false cond."""
    if cond is False:
        raise AssertionError("tl.static_assert failed: {0}".format(msg))
    return None


def static_print(*args, **kwargs):
    """Compile-time print — no-op in this shim."""
    return None


def device_print(*args, **kwargs):
    """In-kernel print — no-op stub in this shim."""
    return None


# ``tl.extra`` namespace (e.g. ``tl.extra.cuda.libdevice``) shows up in some
# kernels at import time; expose a permissive attribute-fabricating object.
class _ExtraNS:
    def __getattr__(self, name):
        return _ExtraNS()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "triton.language.extra.* is only usable inside an @triton.jit "
            "kernel, which the jittor triton shim does not execute."
        )


extra = _ExtraNS()


# math / libdevice subnamespaces referenced as ``tl.math.*`` in older kernels
class _MathNS:
    def __getattr__(self, name):
        return _in_kernel_only("math." + name)


math = _MathNS()
libdevice = _MathNS()
