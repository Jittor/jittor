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

Bridge mode — running **real** triton kernels
---------------------------------------------
If a genuine upstream ``triton`` *is* installed **and** jittor has CUDA, this
package switches to **bridge mode** instead of shadowing it: ``import triton``
keeps resolving to the real package, but its ``JITFunction.run`` is patched so
that a kernel launched with jittor ``Var`` arguments is **compiled by upstream
triton and executed by jittor** on its own device pointers
(:mod:`jittor.triton_shim.backend`). This runs the kernels the naive tracer
cannot — ``tl.dot`` matmul, 2-D softmax, fused layernorm, flash-attention,
fp16/bf16, ``@triton.autotune`` — verified in ``test_triton_backend.py``::

    import jittor as jt; jt.flags.use_cuda = 1
    import jittor.triton_shim            # auto-bridges a real triton
    import triton, triton.language as tl

    @triton.jit
    def add(x, y, o, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    x, y = jt.rand(1024), jt.rand(1024); o = jt.zeros(1024)
    add[(triton.cdiv(1024, 256),)](x, y, o, 1024, B=256)   # runs on the GPU

The backend uses triton purely as a *compiler* (``triton.compile`` with an
explicit ``GPUTarget``) and launches the resulting cubin itself via the CUDA
driver API — it never touches triton's own runtime driver. CUDA only; the shim /
tracer below remains the fallback when triton or a GPU is absent.

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
import os
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
        # First try the naive executor: it can actually *run* a narrow class of
        # 1-D elementwise kernels by tracing the body once and lowering tl.* to
        # jittor ops over whole Vars (grid/BLOCK are irrelevant — jittor
        # vectorises). If the kernel is outside that subset it raises a clear
        # NotImplementedError/TracingError and we keep the original contract.
        try:
            from . import launch as _launch_mod
        except Exception:
            _launch_mod = None
        if _launch_mod is not None:
            try:
                return _launch_mod.run_kernel(self, grid, args, kwargs)
            except _launch_mod.TracingError as e:
                # unsupported construct -> fall through to the clear stub error,
                # but surface *why* so the failure is diagnosable, not opaque.
                raise NotImplementedError(
                    "triton kernel {0!r} could not be run by the jittor naive "
                    "triton executor (only 1-D elementwise kernels are "
                    "supported). Reason: {1} Use the pure-PyTorch path — guard "
                    "the triton launch behind `is_triton_available()` or a "
                    "try/except so the library falls back. (grid={2!r})".format(
                        self.__name__, e, grid)
                ) from e
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
#: True once we have monkeypatched a real triton's JITFunction for jittor.
_bridge_active = False


def _detect_real_triton():
    """Return a genuine upstream ``triton`` module if installed, else ``None``.

    Never returns this shim or the deploy redirect (both carry a
    ``__*triton_shim__`` marker). Imports the real package only when one is
    actually on the path (so a no-triton environment stays lightweight).
    """
    import importlib
    import importlib.util

    existing = sys.modules.get("triton")
    if existing is not None:
        if existing is _self_module or \
           getattr(existing, "__triton_shim__", False) or \
           getattr(existing, "__jittor_triton_shim__", False):
            return None
        return existing  # a real triton is already imported
    try:
        spec = importlib.util.find_spec("triton")
    except Exception:
        spec = None
    if spec is None:
        return None
    # don't import our own deploy-redirect as if it were the real thing
    origin = getattr(spec, "origin", "") or ""
    if origin and os.path.isfile(origin):
        try:
            with open(origin) as f:
                head = f.read(2048)
            if "__jittor_triton_shim__" in head or "__triton_shim__" in head:
                return None
        except Exception:
            pass
    try:
        real = importlib.import_module("triton")
    except Exception:
        return None
    if getattr(real, "__triton_shim__", False) or \
       getattr(real, "__jittor_triton_shim__", False):
        return None
    return real


def _args_have_jittor_var(args, kwargs):
    try:
        import jittor as jt
    except Exception:
        return False
    Var = jt.Var
    return any(isinstance(v, Var) for v in args) or \
        any(isinstance(v, Var) for v in kwargs.values())


def activate_bridge(real=None):
    """Patch a real triton's ``JITFunction.run`` to run on jittor Vars.

    *Bridge mode*: when upstream triton + a CUDA GPU are present, a real
    ``@triton.jit`` kernel launched with jittor ``Var`` arguments is compiled by
    upstream triton and executed by jittor's CUDA-driver launcher
    (:mod:`jittor.triton_shim.backend`). Launches with non-jittor args fall
    through to triton's own runtime unchanged. Idempotent; returns True if the
    bridge is active. Safe to call when triton is absent (returns False).
    """
    global _bridge_active
    if _bridge_active:
        return True
    if real is None:
        real = _detect_real_triton()
    if real is None:
        return False
    try:
        from . import backend
        if not backend.is_available():
            return False
        JF = real.runtime.jit.JITFunction
        if getattr(JF, "_jittor_bridge", False):
            _bridge_active = True
            return True
        orig_run = JF.run

        def _bridge_run(self, *args, grid=None, warmup=False, **kwargs):
            if _args_have_jittor_var(args, kwargs):
                return backend.run(self, args, kwargs, grid)
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)

        _bridge_run._jittor_orig = orig_run
        JF.run = _bridge_run
        JF._jittor_bridge = True

        # @triton.autotune benchmarks configs via triton.testing.do_bench, which
        # routes through torch._dynamo (absent/incompatible here). Swap in a
        # torch-free timer so autotuned kernels select a config and run.
        try:
            bench = backend.make_do_bench()
            import triton.testing as _tt
            _tt.do_bench = bench
            try:
                import triton.runtime.autotuner as _at
                _at.do_bench = bench  # rebind the name already imported there
            except Exception:
                pass
        except Exception:
            pass

        _bridge_active = True
        return True
    except Exception:
        return False


def install(force=False):
    """Make ``import triton`` work for jittor — bridge a real triton, or shim it.

    Two modes, decided automatically:

    * **Bridge mode** — if a genuine upstream ``triton`` is installed (and a CUDA
      GPU is present), we do *not* shadow it: ``import triton`` keeps resolving
      to the real package, and we patch its ``JITFunction.run`` so kernels
      launched on jittor ``Var`` s are compiled by triton and run by jittor
      (:mod:`jittor.triton_shim.backend`). This unlocks real ``tl.dot`` / tiled /
      fused kernels that the naive tracer cannot run.
    * **Shim mode** — if no real triton is installed (or ``force=True``), register
      this shim as ``triton`` / ``triton.language`` in ``sys.modules`` so a bare
      ``import triton`` resolves here (graceful: ``@triton.jit`` works, the naive
      executor runs a 1-D elementwise subset, everything else raises clearly).

    Idempotent. Returns the module now acting as ``triton`` (the real package in
    bridge mode, or this shim).
    """
    if not force:
        real = _detect_real_triton()
        if real is not None:
            activate_bridge(real)
            return real

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
# to make ``import triton`` work — bridging a real triton (preferred) or
# shadowing with the shim when none is installed. install() never clobbers a
# real triton (it patches it for jittor instead), so this is safe to call here.
try:
    install()
except Exception:
    # never let registration failure break ``import jittor.triton_shim``
    pass
