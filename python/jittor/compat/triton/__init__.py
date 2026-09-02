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
(:mod:`jittor.compat.triton.backend`). This runs the kernels the naive tracer
cannot — ``tl.dot`` matmul, 2-D softmax, fused layernorm, flash-attention,
fp16/bf16, ``@triton.autotune`` — verified in ``test_triton_backend.py``::

    import jittor as jt; jt.flags.use_cuda = 1
    import jittor.compat.triton            # auto-bridges a real triton
    import triton, triton.language as tl

    @triton.jit
    def add(x, y, o, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    x, y = jt.rand(1024), jt.rand(1024); o = jt.zeros(1024)
    add[(triton.cdiv(1024, 256),)](x, y, o, 1024, B=256)   # runs on the GPU

The backend uses triton purely as a *compiler* (``triton.compile`` with an
explicit ``GPUTarget``) and launches the resulting cubin itself via the CUDA
driver API — it never touches triton's own runtime driver.

This makes triton **torch-compatible** as well: under jittor's torch shim a
``torch`` tensor *is* a jittor ``Var``, and the bridge additionally duck-types
any object with ``data_ptr`` / ``dtype`` / ``shape``, so existing ``import torch``
+ triton library code (transformers / unsloth-style fused kernels) launches on
jittor unmodified. CUDA only; the shim / tracer below remains the fallback when
triton or a GPU is absent.

Enabling ``import triton``
--------------------------
Three ways, increasing in transparency:

1. Direct use (no global registration)::

       from jittor.compat.triton import triton
       import jittor.compat.triton.language as tl

2. Register into ``sys.modules`` for the current process so a *bare*
   ``import triton`` / ``import triton.language as tl`` resolves to this shim::

       import jittor.compat.triton
       jittor.compat.triton.install()        # idempotent
       import triton                        # -> this shim
       import triton.language as tl         # -> jittor.compat.triton.language

   ``install()`` is also run automatically the first time this package is
   imported (best-effort), so step 2 usually reduces to ``import
   jittor.compat.triton``.

3. Persistent, no jittor import required first (via the canonical deploy CLI)::

       jittor-triton-shim
       python -m jittor.compat.triton.deploy        # writes triton/ into site-packages
       python -m jittor.compat.triton.deploy --check

   After deploy, a plain ``import triton`` in that environment resolves to a
   tiny redirect module that re-exports this shim — even before ``import
   jittor``.
"""
import os
import sys
import types

from . import language
from ..diagnostics import EXPECTED, swallowed

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
        except EXPECTED as exc:
            swallowed("triton/__init__.py _launch: from . import launch as _launch_mod", exc)
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
    except EXPECTED as exc:
        swallowed("triton/__init__.py _detect_real_triton: spec = importlib.util.find_spec('triton')", exc)
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
        except OSError as exc:
            swallowed("triton/__init__.py _detect_real_triton: with open(origin) as f:", exc)
    try:
        real = importlib.import_module("triton")
    except EXPECTED as exc:
        swallowed("triton/__init__.py _detect_real_triton: real = importlib.import_module('triton')", exc)
        return None
    if getattr(real, "__triton_shim__", False) or \
       getattr(real, "__jittor_triton_shim__", False):
        return None
    return real


def _args_have_jittor_var(args, kwargs):
    """True if any launch arg is a device tensor we should run on jittor.

    Covers a jittor ``Var`` (which is also what jittor's torch shim hands out, so
    ``import jittor as torch`` code is included) and, duck-typed, any torch-like
    tensor (has ``data_ptr`` + ``dtype`` + ``shape``) — so real torch+triton code
    routes through jittor's backend in bridge mode.
    """
    try:
        import jittor as jt
        Var = jt.Var
    except EXPECTED as exc:
        swallowed("triton/__init__.py _args_have_jittor_var: import jittor as jt", exc)
        Var = ()

    def _is_t(v):
        if Var and isinstance(v, Var):
            return True
        return hasattr(v, "data_ptr") and hasattr(v, "dtype") and hasattr(v, "shape")

    return any(_is_t(v) for v in args) or any(_is_t(v) for v in kwargs.values())


def _ensure_libcuda_linkable():
    """Make ``gcc -lcuda`` succeed for triton's *internal* driver compile.

    The jittor backend launches kernels itself (it never needs triton's runtime
    driver), but some triton versions (e.g. 3.2's ``Autotuner.__init__`` ->
    ``driver.active.get_benchmarker()``) **eagerly** initialise that driver at
    *import* time, which compiles a small ``driver.c`` linking ``-lcuda``. On
    boxes that ship only ``libcuda.so.1`` (the runtime soname) without the
    ``libcuda.so`` *dev symlink*, that link fails and import blows up before any
    kernel runs. The CUDA toolkit ships a stub ``libcuda.so`` for exactly this
    (link against the stub; the real driver is used at runtime), so we prepend a
    stub dir to ``LIBRARY_PATH`` when no linkable ``libcuda.so`` is found.

    Best-effort and idempotent; never raises.
    """
    import os
    import glob
    try:
        # already linkable? (dev symlink present on the default search path)
        for d in ("/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu",
                  "/usr/lib64", "/usr/lib"):
            if os.path.exists(os.path.join(d, "libcuda.so")):
                return
        for d in os.environ.get("LIBRARY_PATH", "").split(os.pathsep):
            if d and os.path.exists(os.path.join(d, "libcuda.so")):
                return
        # find a stub libcuda.so: jittor's jtcuda first, then CUDA toolkits.
        cands = []
        try:
            import jittor.compiler as _c
            nv = getattr(_c, "nvcc_path", "") or ""
            if nv:
                home = os.path.dirname(os.path.dirname(nv))
                cands.append(os.path.join(home, "lib64", "stubs", "libcuda.so"))
        except EXPECTED as exc:
            swallowed("triton/__init__.py _ensure_libcuda_linkable: import jittor.compiler as _c", exc)
        cands += glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so")
        cands += glob.glob("/usr/local/cuda*/lib64/stubs/libcuda.so")
        for so in cands:
            if os.path.isfile(so):
                d = os.path.dirname(so)
                cur = os.environ.get("LIBRARY_PATH", "")
                if d not in cur.split(os.pathsep):
                    os.environ["LIBRARY_PATH"] = (d + os.pathsep + cur) if cur else d
                return
    except EXPECTED as exc:
        swallowed("triton/__init__.py _ensure_libcuda_linkable: for d in ('/usr/lib/x86_64-linux-gnu', '/lib/x86_64-lin...", exc)


def activate_bridge(real=None):
    """Patch a real triton's ``JITFunction.run`` to run on jittor Vars.

    *Bridge mode*: when upstream triton + a CUDA GPU are present, a real
    ``@triton.jit`` kernel launched with jittor ``Var`` arguments is compiled by
    upstream triton and executed by jittor's CUDA-driver launcher
    (:mod:`jittor.compat.triton.backend`). Launches with non-jittor args fall
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
        # triton may eagerly init its runtime driver at import (compiling a
        # -lcuda stub); make that link succeed even without the dev symlink.
        _ensure_libcuda_linkable()
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
            except (AttributeError, TypeError) as exc:
                swallowed("triton/__init__.py activate_bridge: import triton.runtime.autotuner as _at", exc)
            # Triton >=3.2's Autotuner.__init__ does
            #   self.do_bench = driver.active.get_benchmarker()
            # instead of using triton.testing.do_bench directly. The jittor
            # backend's driver lacks get_benchmarker -> AttributeError at the
            # first @triton.autotune kernel (flaky: only when autotuning runs).
            # Provide it (returns our torch-free bench, which Triton calls as
            # do_bench(kernel_call, quantiles=...)). Patch instance AND class so
            # it survives driver.active re-resolution.
            try:
                from triton.runtime import driver as _drv
                _active = getattr(_drv, "active", None)
                if _active is not None and not hasattr(_active, "get_benchmarker"):
                    try:
                        _active.get_benchmarker = lambda *a, **k: bench
                    except (AttributeError, TypeError) as exc:
                        swallowed("triton/__init__.py activate_bridge: _active.get_benchmarker = lambda *a, **k: bench", exc)
                    try:
                        type(_active).get_benchmarker = lambda self, *a, **k: bench
                    except EXPECTED as exc:
                        swallowed("triton/__init__.py activate_bridge: type(_active).get_benchmarker = lambda self, *a, **k: b...", exc)
            except EXPECTED as exc:
                swallowed("triton/__init__.py activate_bridge: from triton.runtime import driver as _drv", exc)
        except EXPECTED as exc:
            swallowed("triton/__init__.py activate_bridge: bench = backend.make_do_bench()", exc)

        _bridge_active = True
        return True
    except EXPECTED as exc:
        swallowed("triton/__init__.py activate_bridge: from . import backend", exc)
        return False


_DEPLOYED_REDIRECT_SIGNATURES = {
    "triton": ("__init__.py", "from jittor.compat.triton import *"),
    "triton.language": (
        "language.py",
        "from jittor.compat.triton.language import *",
    ),
}


def _is_deployed_redirect(name, module):
    """Recognize only redirect modules written by this package's deploy CLI."""

    signature = _DEPLOYED_REDIRECT_SIGNATURES.get(name)
    if signature is None:
        return False
    expected_file, expected_import = signature
    source = getattr(module, "__file__", None)
    if not source:
        return False
    path = os.path.realpath(os.fspath(source))
    if not (
        getattr(module, "__jittor_triton_shim__", False)
        and getattr(module, "__name__", "") == name
        and os.path.basename(path) == expected_file
        and os.path.basename(os.path.dirname(path)) == "triton"
        and os.path.isfile(path)
    ):
        return False
    try:
        with open(path, "r", encoding="utf-8") as source:
            head = source.read(4096)
    except (OSError, UnicodeError):
        return False
    return (
        "__jittor_triton_shim__ = True" in head
        and expected_import in head
    )


def install(force=False):
    """Make ``import triton`` work for jittor — bridge a real triton, or shim it.

    Two modes, decided automatically:

    * **Bridge mode** — if a genuine upstream ``triton`` is installed (and a CUDA
      GPU is present), we do *not* shadow it: ``import triton`` keeps resolving
      to the real package, and we patch its ``JITFunction.run`` so kernels
      launched on jittor ``Var`` s are compiled by triton and run by jittor
      (:mod:`jittor.compat.triton.backend`). This unlocks real ``tl.dot`` / tiled /
      fused kernels that the naive tracer cannot run.
    * **Shim mode** — if no real triton is installed, register this shim as
      ``triton`` / ``triton.language`` in ``sys.modules`` so a bare ``import
      triton`` resolves here (graceful: ``@triton.jit`` works, the naive executor
      runs a 1-D elementwise subset, everything else raises clearly).

    ``force=True`` skips real-package detection, but it never overwrites a real
    or conflicting preloaded module graph. The only replaceable foreign objects
    are source-verified redirects generated by :mod:`jittor.compat.triton.deploy`.

    Idempotent. Returns the module now acting as ``triton`` (the real package in
    bridge mode, or this shim).
    """
    if not force:
        real = _detect_real_triton()
        if real is not None:
            activate_bridge(real)
            return real

    managed = {
        "triton": _self_module,
        "triton.language": language,
        "triton.runtime": runtime,
        "triton.runtime.jit": runtime,
        "triton.runtime.autotuner": runtime,
    }
    conflicts = []
    for name, module in managed.items():
        current = sys.modules.get(name)
        if current in (None, module) or _is_deployed_redirect(name, current):
            continue
        conflicts.append(name)
    if conflicts:
        raise RuntimeError(
            "cannot install the Jittor Triton shim over a preloaded Triton "
            "module graph: %s" % ", ".join(sorted(conflicts))
        )
    for name, module in managed.items():
        sys.modules[name] = module
    # ``triton.language`` should be reachable as an attribute too
    setattr(_self_module, "language", language)
    return _self_module


# resolve our own module object for self-registration
_self_module = sys.modules[__name__]

# Convenience alias so the "no global registration" path works:
#     from jittor.compat.triton import triton
# `triton` here is simply this package module itself.
triton = _self_module

# Best-effort auto-install on import so ``import jittor.compat.triton`` is enough
# to make ``import triton`` work — bridging a real triton (preferred) or
# shadowing with the shim when none is installed. install() never clobbers a
# real triton (it patches it for jittor instead), so this is safe to call here.
try:
    install()
except EXPECTED as exc:
    # never let registration failure break ``import jittor.compat.triton``
    swallowed("triton/__init__.py <module>: install()", exc)
