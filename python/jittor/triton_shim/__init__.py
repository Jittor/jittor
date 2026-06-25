"""``triton`` compatibility shim for jittor (``import jittor as torch``).

Goal
----
Make ``import triton`` / ``import triton.language as tl`` *succeed* on a jittor
environment so that torch-targeted libraries (transformers, ``diffusers``, ...)
can take their ``if is_triton_available(): ...`` branch without crashing at
import time, while keeping the contract **"clear error beats silent wrong /
crash"**:

* host-side utilities that are trivially implementable are implemented for real
  (``triton.cdiv``, ``triton.next_power_of_2``, ``tl.cdiv``);
* ``@triton.jit`` decorates the kernel, supports the ``kernel[grid](args...)``
  launch syntax, and raises a **clear** ``NotImplementedError`` on launch
  (jittor cannot yet JIT-compile a triton kernel) — it never silently no-ops
  and never segfaults;
* the in-kernel ``tl.*`` primitives are stubs that raise a clear "only usable
  inside @triton.jit" ``NotImplementedError`` if executed.

This is intentionally a *shim* (API surface), not a triton backend.

Enabling ``import triton``
--------------------------
Three ways, increasing in transparency:

1. Direct use (no global registration)::

       from jittor.triton_shim import triton
       import jittor.triton_shim.language as tl

2. Register into ``sys.modules`` for the current process so a *bare*
   ``import triton`` / ``import triton.language as tl`` resolves to this shim::

       import jittor.triton_shim
       jittor.triton_shim.install()        # idempotent
       import triton                        # -> this shim
       import triton.language as tl         # -> jittor.triton_shim.language

   ``install()`` is also run automatically the first time this package is
   imported (best-effort), so step 2 usually reduces to ``import
   jittor.triton_shim``.

3. Persistent, no jittor import required first (mirrors
   ``python -m jittor.torch_shim.deploy``)::

       python -m jittor.triton_shim.deploy        # writes triton/ into site-packages
       python -m jittor.triton_shim.deploy --check

   After deploy, a plain ``import triton`` in that environment resolves to a
   tiny redirect module that re-exports this shim — even before ``import
   jittor``.
"""
import sys
import types

from . import language

__all__ = [
    "jit", "JITFunction", "KernelInterface",
    "cdiv", "next_power_of_2", "Config", "autotune", "heuristics",
    "language", "runtime", "OutOfResources", "TritonError",
    "install", "is_shim", "__version__",
]

# Version string. transformers/accelerate gate features on triton version via
# ``importlib.metadata.version("triton")`` or ``triton.__version__``; advertise
# a recent-ish version so feature gates that *require* triton evaluate true.
__version__ = "3.1.0"

#: marker so callers can detect "this is the jittor stub, not real triton".
__triton_shim__ = True


def is_shim():
    """Return True — this ``triton`` is the jittor compatibility shim."""
    return True


# --------------------------------------------------------------------------- #
#  Errors (subset of triton's public exception surface)
# --------------------------------------------------------------------------- #
class TritonError(Exception):
    """Base triton error (shim)."""


class OutOfResources(TritonError):
    """Raised by real triton when a kernel exceeds HW limits (shim)."""

    def __init__(self, required=None, limit=None, name=None):
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(
            "out of resource: {0}, required: {1}, limit: {2}".format(name, required, limit)
        )


# --------------------------------------------------------------------------- #
#  Host-side utilities (real implementations)
# --------------------------------------------------------------------------- #
def cdiv(x, div):
    """Ceiling division ``ceil(x / div)``."""
    return -(-int(x) // int(div))


def next_power_of_2(n):
    """Smallest power of two ``>= n`` (matches ``triton.next_power_of_2``)."""
    n = int(n)
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


# --------------------------------------------------------------------------- #
#  @triton.jit  +  kernel[grid](...) launch syntax
# --------------------------------------------------------------------------- #
class KernelInterface:
    """Mixin providing the ``kernel[grid]`` indexing -> launcher protocol."""

    def __getitem__(self, grid):
        # real triton: ``kernel[grid]`` returns a callable bound to that grid.
        def _launcher(*args, **kwargs):
            return self._launch(grid, args, kwargs)

        _launcher.__name__ = getattr(self, "__name__", "triton_kernel") + "_launcher"
        return _launcher


class JITFunction(KernelInterface):
    """What ``@triton.jit`` returns: a record of the kernel + a launch stub.

    * Decorating works (no exception at definition time).
    * ``kernel.fn`` keeps the original Python function (real triton does too),
      so a library may still call it directly / introspect it.
    * Launching — ``kernel[grid](args...)`` (or a direct call, which real triton
      forbids) — raises a clear ``NotImplementedError`` instead of crashing.
    """

    def __init__(self, fn, **options):
        self.fn = fn
        self.options = options
        # mirror useful attributes of the wrapped function
        self.__name__ = getattr(fn, "__name__", "triton_kernel")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", None)
        self.__wrapped__ = fn
        # introspectable arg names (some libs read these)
        try:
            import inspect
            self.arg_names = list(inspect.signature(fn).parameters.keys())
        except (TypeError, ValueError):
            self.arg_names = []
        self.constexprs = []
        self.cache = {}

    # -- launch -------------------------------------------------------------- #
    def _launch(self, grid, args, kwargs):
        raise NotImplementedError(
            "triton kernel {0!r} cannot be executed on jittor: the jittor "
            "triton shim provides the triton API surface (so `import triton` "
            "and @triton.jit do not crash) but does not JIT-compile/run kernels. "
            "Use the pure-PyTorch path — guard the triton launch behind "
            "`is_triton_available()` or a try/except ImportError so the library "
            "falls back. (grid={1!r})".format(self.__name__, grid)
        )

    def __call__(self, *args, **kwargs):
        # real triton raises if you call a @jit fn outside another @jit fn;
        # give the same "must be launched" guidance, clearly.
        raise NotImplementedError(
            "triton kernel {0!r} was called directly. triton kernels must be "
            "launched as `{0}[grid](args...)`; and on jittor even that raises "
            "NotImplementedError (no triton backend). Use the pure-PyTorch "
            "fallback path.".format(self.__name__)
        )

    # let ``kernel.warmup(...)`` / ``kernel.run(...)`` exist but be clear
    def warmup(self, *args, **kwargs):
        raise NotImplementedError(
            "triton kernel {0!r}.warmup is not supported on the jittor triton "
            "shim (no triton backend).".format(self.__name__)
        )

    def run(self, *args, **kwargs):
        grid = kwargs.pop("grid", None)
        return self._launch(grid, args, kwargs)

    def __repr__(self):
        return "<jittor-triton-shim JITFunction {0!r} (no backend)>".format(self.__name__)


def jit(fn=None, **options):
    """``@triton.jit`` decorator (shim).

    Usable bare (``@triton.jit``) or parameterised
    (``@triton.jit(do_not_specialize=[...])``). Returns a :class:`JITFunction`
    that records the kernel and supports ``kernel[grid](...)`` launch syntax
    (which raises a clear ``NotImplementedError`` — no triton backend on jittor).
    """
    if fn is None:
        # called with kwargs: @triton.jit(...) -> return the real decorator
        def _decorator(real_fn):
            return JITFunction(real_fn, **options)

        return _decorator
    return JITFunction(fn, **options)


# --------------------------------------------------------------------------- #
#  autotune / heuristics / Config  (decorators must be no-op-passthrough-safe)
# --------------------------------------------------------------------------- #
class Config:
    """Autotuning config descriptor (shim) — stores kwargs, otherwise inert."""

    def __init__(self, kwargs=None, num_warps=4, num_stages=2, num_ctas=1,
                 maxnreg=None, pre_hook=None, **extra):
        self.kwargs = dict(kwargs or {})
        self.kwargs.update(extra)
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def __repr__(self):
        return "triton.Config({0!r}, num_warps={1}, num_stages={2})".format(
            self.kwargs, self.num_warps, self.num_stages
        )

    # some code does ``config.all_kwargs()``
    def all_kwargs(self):
        d = dict(self.kwargs)
        d.update(num_warps=self.num_warps, num_stages=self.num_stages,
                 num_ctas=self.num_ctas)
        return d


def autotune(configs=None, key=None, **kwargs):
    """``@triton.autotune`` (shim): returns a decorator that just wraps the
    kernel with :func:`jit` semantics (records configs, no actual tuning)."""

    def _decorator(fn):
        kf = fn if isinstance(fn, JITFunction) else JITFunction(fn)
        kf.configs = list(configs or [])
        kf.autotune_key = list(key or [])
        return kf

    return _decorator


def heuristics(values=None, **kwargs):
    """``@triton.heuristics`` (shim): returns a passthrough decorator."""

    def _decorator(fn):
        kf = fn if isinstance(fn, JITFunction) else JITFunction(fn)
        kf.heuristics = dict(values or {})
        return kf

    return _decorator


# --------------------------------------------------------------------------- #
#  ``triton.runtime`` namespace (a few libs touch it)
# --------------------------------------------------------------------------- #
runtime = types.ModuleType("triton.runtime")
runtime.__doc__ = "Stub triton.runtime namespace (jittor shim)."
runtime.driver = types.SimpleNamespace(
    active=types.SimpleNamespace(
        get_current_device=lambda: 0,
        get_current_stream=lambda *a, **k: 0,
        utils=types.SimpleNamespace(),
    )
)
runtime.jit = jit
runtime.Autotuner = autotune
runtime.Config = Config


# --------------------------------------------------------------------------- #
#  Registration:  make ``import triton`` resolve to this shim
# --------------------------------------------------------------------------- #
def install(force=False):
    """Register this shim as ``triton`` / ``triton.language`` in ``sys.modules``.

    After calling, a bare ``import triton`` and ``import triton.language as tl``
    resolve to this package (process-local). Idempotent.

    If a *real* ``triton`` is already importable / imported, this is a no-op
    unless ``force=True`` (we never clobber a real install silently).

    Returns the module now registered as ``triton`` (the shim, or the
    pre-existing real triton).
    """
    existing = sys.modules.get("triton")
    if existing is not None and existing is not _self_module:
        if not getattr(existing, "__triton_shim__", False) and not force:
            # a real triton (or another shim) already owns the name
            return existing

    sys.modules["triton"] = _self_module
    sys.modules["triton.language"] = language
    # common dotted submodules libraries import explicitly
    sys.modules.setdefault("triton.runtime", runtime)
    sys.modules.setdefault("triton.runtime.jit", runtime)
    sys.modules.setdefault("triton.runtime.autotuner", runtime)
    # ``triton.language`` should be reachable as an attribute too
    setattr(_self_module, "language", language)
    return _self_module


# resolve our own module object for self-registration
_self_module = sys.modules[__name__]
# advertise triton.language as a real submodule attribute immediately
language.__name__ = "triton.language"

# Convenience alias so the "no global registration" path works:
#     from jittor.triton_shim import triton
# `triton` here is simply this package module itself.
triton = _self_module

# Best-effort auto-install on import so ``import jittor.triton_shim`` is enough
# to make a later bare ``import triton`` work, without clobbering a real triton.
try:
    if "triton" not in sys.modules or getattr(sys.modules.get("triton"), "__triton_shim__", False):
        install()
except Exception:
    # never let registration failure break ``import jittor.triton_shim``
    pass
