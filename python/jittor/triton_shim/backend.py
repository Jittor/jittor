"""Real Triton backend for jittor — *bridge mode*.

When the upstream ``triton`` package **and** a CUDA GPU are available, jittor's
:mod:`jittor.triton_shim` activates *bridge mode*: real ``@triton.jit`` kernels
are compiled by upstream triton (Python → PTX/cubin) and **launched by jittor**
on its own ``Var`` device pointers through the CUDA driver API. This runs the
kernels the naive tracer (:mod:`jittor.triton_shim.launch`) cannot — ``tl.dot`` /
matmul, 2-D tiles, fused softmax / layernorm, atomics, block pointers, …

Strategy — "compile-only"
-------------------------
We deliberately **do not** use triton's own runtime *driver*. That driver builds
a C launcher that links ``-lcuda`` and probes the device via torch; it is
fragile (and outright fails on some boxes — missing ``libcuda.so`` dev symlink).
Instead we:

1. build the kernel ``signature`` / ``constants`` from the call args ourselves,
2. call :func:`triton.compile` with an **explicit** ``GPUTarget`` (so triton
   never touches its driver / torch), obtaining the **cubin**,
3. load + launch that cubin ourselves with :mod:`ctypes` over ``libcuda.so.1``,
   passing each tensor arg's ``Var.raw_ptr`` as a ``CUdeviceptr``.

Triton kernels write their result **in place** through the output pointer the
caller passed, so after a launch the caller's pre-allocated output ``Var`` simply
*contains* the result — matching real triton's semantics exactly, with no
aliasing tricks.

Correctness model (phase 1)
---------------------------
The launch is bracketed by ``jt.sync_all(True)`` (so every input/output ``Var``
is materialised and its ``raw_ptr`` valid) and a ``cuCtxSynchronize`` afterwards.
This serialises at the boundary — correct but not maximally pipelined; phase 3
moves the launch onto jittor's own stream as a graph node. See the plan.

This module is imported lazily (only when bridge mode is actually used), so
importing :mod:`jittor.triton_shim` never imports jittor or triton eagerly.
"""
import ctypes
import threading

__all__ = ["is_available", "run", "make_do_bench", "JittorTritonError"]


class JittorTritonError(RuntimeError):
    """A real-triton-on-jittor launch failed in a way the user should see."""


# --------------------------------------------------------------------------- #
#  real triton resolution (never the shim)
# --------------------------------------------------------------------------- #
def real_triton():
    """Return the upstream ``triton`` module, or ``None`` if only the shim/none.

    Guards against our own shim or deploy-redirect shadowing ``triton`` in
    ``sys.modules``: such a module carries ``__triton_shim__`` /
    ``__jittor_triton_shim__`` and is rejected.
    """
    import sys
    mod = sys.modules.get("triton")
    if mod is not None:
        if getattr(mod, "__triton_shim__", False) or \
           getattr(mod, "__jittor_triton_shim__", False):
            mod = None
    if mod is None:
        # try a real import only if the name isn't currently shadowed by us
        shadow = sys.modules.get("triton")
        if shadow is not None and (getattr(shadow, "__triton_shim__", False) or
                                   getattr(shadow, "__jittor_triton_shim__", False)):
            return None  # don't fight an installed shadow here; caller handles it
        try:
            import triton as mod  # noqa: F401
        except Exception:
            return None
    # sanity: a real triton has a compiler with ASTSource
    try:
        from triton.compiler import ASTSource  # noqa: F401
        from triton.backends.compiler import GPUTarget  # noqa: F401
    except Exception:
        return None
    return mod


_available = None


def is_available():
    """True iff real triton + a CUDA GPU + jittor-with-cuda are all present."""
    global _available
    if _available is not None:
        return _available
    ok = False
    try:
        if real_triton() is not None:
            import jittor as jt
            ok = bool(getattr(jt, "has_cuda", 0))
    except Exception:
        ok = False
    _available = ok
    return ok


# --------------------------------------------------------------------------- #
#  ctypes CUDA driver wrapper (libcuda.so.1)
# --------------------------------------------------------------------------- #
class _Driver:
    """Minimal, error-checked CUDA *driver* API surface over ``libcuda.so.1``.

    We use the driver API (not the runtime) because triton emits cubin and we
    load/launch it with ``cuModuleLoadData`` / ``cuLaunchKernel`` — and because
    the driver's *primary context* is shared with jittor's runtime-API context,
    so launching here touches the very same device memory jittor allocated.
    """

    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        lib = None
        last = None
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as e:
                last = e
        if lib is None:
            raise JittorTritonError("could not load libcuda.so.1 (%s)" % last)
        self.lib = lib
        c_vp, c_vpp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        c_i, c_u = ctypes.c_int, ctypes.c_uint
        self._sig(lib.cuInit, [c_u])
        self._sig(lib.cuDeviceGet, [ctypes.POINTER(c_i), c_i])
        self._sig(lib.cuDeviceComputeCapability,
                  [ctypes.POINTER(c_i), ctypes.POINTER(c_i), c_i])
        self._sig(lib.cuDevicePrimaryCtxRetain, [c_vpp, c_i])
        self._sig(lib.cuCtxSetCurrent, [c_vp])
        self._sig(lib.cuCtxSynchronize, [])
        self._sig(lib.cuModuleLoadData, [c_vpp, c_vp])
        self._sig(lib.cuModuleGetFunction, [c_vpp, c_vp, ctypes.c_char_p])
        self._sig(lib.cuLaunchKernel,
                  [c_vp, c_u, c_u, c_u, c_u, c_u, c_u, c_u, c_vp, c_vpp, c_vpp])
        self._sig(lib.cuFuncSetAttribute, [c_vp, c_i, c_i])
        self._sig(lib.cuMemAlloc, [c_vpp, ctypes.c_size_t])
        lib.cuGetErrorString.argtypes = [c_i, ctypes.POINTER(ctypes.c_char_p)]

        self.check(lib.cuInit(0), "cuInit")
        dev = c_i(0)
        self.check(lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.device = dev
        ctx = ctypes.c_void_p()
        self.check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
                   "cuDevicePrimaryCtxRetain")
        self.check(lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self.ctx = ctx
        # compute capability -> triton arch (e.g. 8.9 -> 89)
        maj, mino = c_i(0), c_i(0)
        self.check(lib.cuDeviceComputeCapability(
            ctypes.byref(maj), ctypes.byref(mino), dev), "cuDeviceComputeCapability")
        self.arch = maj.value * 10 + mino.value
        self._modules = {}   # id(cubin_bytes) -> CUmodule handle
        self._funcs = {}     # (id(cubin_bytes), name) -> CUfunction handle

    @staticmethod
    def _sig(fn, argtypes):
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int

    def check(self, res, what):
        if res != 0:
            s = ctypes.c_char_p()
            try:
                self.lib.cuGetErrorString(res, ctypes.byref(s))
            except Exception:
                pass
            raise JittorTritonError("%s -> CUresult %d (%s)" % (
                what, res, s.value.decode() if s.value else "?"))

    @classmethod
    def get(cls):
        if cls._inst is None:
            with cls._lock:
                if cls._inst is None:
                    cls._inst = cls()
        return cls._inst

    def get_function(self, cubin, name):
        """Load (and cache) the cubin module + return the named CUfunction."""
        mkey = id(cubin)
        mod = self._modules.get(mkey)
        if mod is None:
            image = ctypes.create_string_buffer(cubin, len(cubin))
            mod = ctypes.c_void_p()
            self.check(self.lib.cuModuleLoadData(
                ctypes.byref(mod), ctypes.cast(image, ctypes.c_void_p)),
                "cuModuleLoadData")
            self._modules[mkey] = mod
        fkey = (mkey, name)
        fn = self._funcs.get(fkey)
        if fn is None:
            fn = ctypes.c_void_p()
            self.check(self.lib.cuModuleGetFunction(
                ctypes.byref(fn), mod, name.encode()), "cuModuleGetFunction")
            self._funcs[fkey] = fn
        return fn

    def alloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cuMemAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
                   "cuMemAlloc")
        return ptr

    # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    _ATTR_MAX_DYN_SHARED = 8
    #: kernels (by func handle value) already opted in to large dynamic smem
    _optedin = set()

    def ensure_dynamic_shared(self, func, nbytes):
        """Opt a kernel in to >48 KB dynamic shared memory (required on sm_70+).

        Triton's fused kernels (flash-attention, big tiles) routinely exceed the
        48 KB static cap; without this opt-in ``cuLaunchKernel`` returns
        ``CUDA_ERROR_INVALID_VALUE``. Idempotent per function handle.
        """
        if nbytes <= 49152:
            return
        key = (func.value, nbytes)
        if key in self._optedin:
            return
        self.check(self.lib.cuFuncSetAttribute(
            func, ctypes.c_int(self._ATTR_MAX_DYN_SHARED), ctypes.c_int(nbytes)),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)")
        self._optedin.add(key)

    def launch(self, func, grid, block, shared, params):
        gx, gy, gz = grid
        bx, by, bz = block
        self.check(self.lib.cuLaunchKernel(
            func,
            ctypes.c_uint(gx), ctypes.c_uint(gy), ctypes.c_uint(gz),
            ctypes.c_uint(bx), ctypes.c_uint(by), ctypes.c_uint(bz),
            ctypes.c_uint(shared), ctypes.c_void_p(0),
            ctypes.cast(params, ctypes.POINTER(ctypes.c_void_p)),
            ctypes.cast(None, ctypes.POINTER(ctypes.c_void_p))),
            "cuLaunchKernel")

    def synchronize(self):
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


# --------------------------------------------------------------------------- #
#  dtype mapping (jittor/torch-style dtype string -> triton type code)
# --------------------------------------------------------------------------- #
_DT = {
    "float64": "fp64", "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    "int64": "i64", "int32": "i32", "int16": "i16", "int8": "i8",
    "uint64": "u64", "uint32": "u32", "uint16": "u16", "uint8": "u8",
    "bool": "i1",
}


def _dtype_name(x):
    # Var.dtype may stringify as 'float32' or (under torch-compat) 'torch.float32'
    return str(x).split(".")[-1].strip()


def _is_var(v):
    import jittor as jt
    return isinstance(v, jt.Var)


def _ptr_sig(var):
    name = _dtype_name(var.dtype)
    code = _DT.get(name)
    if code is None:
        raise JittorTritonError("unsupported jittor Var dtype for triton: %r" % name)
    return "*" + code


def _scalar_sig(v):
    if isinstance(v, bool):
        return "i1"
    if isinstance(v, int):
        return "i32" if (-(2 ** 31) <= v < 2 ** 31) else "i64"
    if isinstance(v, float):
        return "fp32"
    raise JittorTritonError(
        "unsupported scalar kernel arg type %r (value %r); only int/float/bool "
        "scalars, jittor Vars, and tl.constexpr are supported." % (type(v).__name__, v))


def _pack_scalar(sig, v):
    """ctypes value object whose bytes are the kernel arg (width per ``sig``)."""
    if sig in ("i64", "u64"):
        return ctypes.c_int64(int(v))
    if sig in ("i32", "u32", "i16", "u16", "i8", "u8", "i1"):
        return ctypes.c_int32(int(v))
    if sig == "fp32":
        return ctypes.c_float(float(v))
    if sig == "fp64":
        return ctypes.c_double(float(v))
    raise JittorTritonError("cannot pack scalar of triton type %r" % sig)


# --------------------------------------------------------------------------- #
#  PTX parameter counting (safety net: detect arg-packing mismatch loudly)
# --------------------------------------------------------------------------- #
def _ptx_param_count(ptx, name):
    import re
    m = re.findall(re.escape(name) + r"_param_(\d+)", ptx)
    return (max(int(x) for x in m) + 1) if m else 0


# --------------------------------------------------------------------------- #
#  compile cache
# --------------------------------------------------------------------------- #
_COMPILE_CACHE = {}
_compile_lock = threading.Lock()


def _compile(jitfn, signature, constants, options=None):
    """Compile (cached) and return a dict with cubin/name/launch metadata.

    ``options`` is an optional dict of launch options forwarded to triton
    (currently ``num_warps`` / ``num_stages``); ``None``/empty lets triton pick
    its defaults.
    """
    triton = real_triton()
    from triton.compiler import AttrsDescriptor, ASTSource

    drv = _Driver.get()
    opt_items = tuple(sorted((options or {}).items()))
    key = (id(jitfn), tuple(sorted(signature.items())),
           tuple(sorted(constants.items())), opt_items, drv.arch)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    with _compile_lock:
        cached = _COMPILE_CACHE.get(key)
        if cached is not None:
            return cached
        from triton.backends.compiler import GPUTarget
        target = GPUTarget("cuda", drv.arch, 32)
        attrs = AttrsDescriptor()  # conservative: no divisibility/eq-1 assumptions
        try:
            src = ASTSource(jitfn, signature, constants, attrs)
            compiled = triton.compile(src, target=target,
                                      options=dict(options) if options else None)
        except Exception as e:
            raise JittorTritonError(
                "upstream triton failed to compile kernel %r: %s: %s" % (
                    getattr(jitfn, "__name__", "?"), type(e).__name__, e)) from e
        md = compiled.metadata
        name = md.name
        ptx = compiled.asm.get("ptx", "")
        info = {
            "cubin": compiled.asm["cubin"],
            "name": name,
            "num_warps": int(md.num_warps),
            "shared": int(getattr(md, "shared", 0) or 0),
            "scratch": int(getattr(md, "global_scratch_size", 0) or 0),
            "scratch_align": int(getattr(md, "global_scratch_align", 1) or 1),
            "n_ptx_params": _ptx_param_count(ptx, name),
        }
        _COMPILE_CACHE[key] = info
        return info


# --------------------------------------------------------------------------- #
#  the public entry point — called by the patched JITFunction.run
# --------------------------------------------------------------------------- #
def make_do_bench():
    """A torch-free replacement for ``triton.testing.do_bench``.

    Triton's own ``do_bench`` (used by ``@triton.autotune`` to time configs)
    routes through ``torch._dynamo`` for device/stream handling, which is not
    available / compatible in a jittor-without-torch (or mismatched-torch) env.
    Since our launcher is synchronous, we time ``fn`` with a plain wall clock
    bracketed by a context sync — good enough for *relative* config ranking,
    which is all the autotuner needs. The signature is permissive (``**kw``) to
    tolerate version drift in triton's call site.
    """
    def _do_bench(fn, warmup=25, rep=100, quantiles=None, grad_to_none=None,
                  return_mode="mean", **kw):
        import time
        drv = _Driver.get()
        for _ in range(5):          # warmup (also forces compile on miss)
            fn()
        drv.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        drv.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / iters
        if quantiles is not None:
            return [ms for _ in quantiles]
        return ms

    return _do_bench


def run(jitfn, args, kwargs, grid):
    """Compile + launch a real ``@triton.jit`` kernel on jittor Vars.

    ``jitfn`` is the upstream triton ``JITFunction``; ``args``/``kwargs`` are the
    launch arguments; ``grid`` is the tuple or callable from ``kernel[grid]``.
    Returns ``None`` — results are written in place into the output Var(s),
    matching real triton's pointer-output convention.
    """
    import inspect
    import jittor as jt

    fn = getattr(jitfn, "fn", jitfn)
    kname = getattr(jitfn, "__name__", getattr(fn, "__name__", "triton_kernel"))

    # -- separate launch-meta kwargs (num_warps/...) from real kernel kwargs --
    kwargs = dict(kwargs)
    options = {}
    for m in ("num_warps", "num_stages"):
        if m in kwargs:
            options[m] = kwargs.pop(m)
    for m in ("num_ctas", "maxnreg", "warmup", "grid", "extern_libs", "stream"):
        kwargs.pop(m, None)

    # -- which params are constexpr? (use triton's own param metadata if present)
    constexpr_names = set()
    params = getattr(jitfn, "params", None)
    if params:
        for p in params:
            if getattr(p, "is_constexpr", False):
                constexpr_names.add(p.name)
    else:
        for n, p in inspect.signature(fn).parameters.items():
            ann = p.annotation
            if "constexpr" in str(ann):
                constexpr_names.add(n)

    # -- bind call args to parameter names ---------------------------------- #
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError as e:
        raise JittorTritonError(
            "could not bind args for triton kernel %r: %s" % (kname, e))

    signature = {}        # runtime param name -> triton type code
    constants = {}        # constexpr param name -> python value
    runtime_vals = []     # (name, sig, value) in param order, runtime only
    out_vars = []         # jittor Vars touched (kept alive across launch)

    def _unwrap_constexpr(v):
        # tl.constexpr(value) -> value
        val = getattr(v, "value", v)
        return val

    for name, val in bound.arguments.items():
        if name in constexpr_names:
            constants[name] = _unwrap_constexpr(val)
            continue
        if _is_var(val):
            if not bool(getattr(val, "is_cuda", 0)):
                raise JittorTritonError(
                    "triton kernel %r received a CPU jittor Var argument %r; the "
                    "jittor triton backend launches on CUDA only. Set "
                    "`jt.flags.use_cuda = 1` before allocating the tensors (or "
                    "move them to GPU)." % (kname, name))
            signature[name] = _ptr_sig(val)
            runtime_vals.append((name, signature[name], val))
            out_vars.append(val)
        elif val is None:
            # triton models a None pointer arg as *i8; pass null
            signature[name] = "*i8"
            runtime_vals.append((name, "*i8", None))
        else:
            val = _unwrap_constexpr(val)
            sig = _scalar_sig(val)
            signature[name] = sig
            runtime_vals.append((name, sig, val))

    if not any(_is_var(v) for (_, _, v) in runtime_vals):
        raise JittorTritonError(
            "triton kernel %r launched with no jittor Var arguments; the jittor "
            "triton backend needs the tensor pointers to be jittor Vars." % kname)

    # ASTSource needs the JITFunction itself (it reads .cache_key), not fn.
    info = _compile(jitfn, signature, constants, options)

    # -- resolve grid (tuple or callable(meta)) ----------------------------- #
    if callable(grid):
        meta = dict(constants)
        meta["num_warps"] = info["num_warps"]
        meta["num_stages"] = options.get("num_stages", 3)
        g = grid(meta)
    else:
        g = grid
    if isinstance(g, int):
        g = (g,)
    g = tuple(int(x) for x in g)
    g = (g + (1, 1, 1))[:3]

    # -- materialise all Var memory so raw_ptr is valid + inputs computed ---- #
    jt.sync_all(True)

    # -- pack kernel params (in param order; pointers first only if scratch) - #
    drv = _Driver.get()
    keepalive = []
    cvals = []
    for (name, sig, val) in runtime_vals:
        if sig.startswith("*"):
            ptr = 0 if val is None else int(val.raw_ptr)
            cv = ctypes.c_uint64(ptr)
        else:
            cv = _pack_scalar(sig, val)
        keepalive.append(cv)
        cvals.append(cv)

    # global scratch (rare): triton prepends a scratch pointer as param 0
    n_expected = len(cvals)
    if info["scratch"] > 0:
        scratch_ptr = drv.alloc(info["scratch"])
        keepalive.append(scratch_ptr)
        cvals = [ctypes.c_uint64(scratch_ptr.value or 0)] + cvals
        n_expected += 1

    # safety net: refuse to launch if our packing disagrees with the cubin
    n_ptx = info["n_ptx_params"]
    if n_ptx and n_ptx != n_expected:
        raise JittorTritonError(
            "arg-count mismatch launching triton kernel %r: packed %d params but "
            "the compiled kernel declares %d (signature=%r, constants=%r). This "
            "usually means an unsupported arg kind or a global-scratch layout the "
            "jittor backend doesn't model yet — refusing to launch rather than "
            "corrupt memory." % (kname, n_expected, n_ptx, signature, constants))

    params = (ctypes.c_void_p * len(cvals))(
        *[ctypes.cast(ctypes.byref(cv), ctypes.c_void_p) for cv in cvals])

    block = (info["num_warps"] * 32, 1, 1)
    func = drv.get_function(info["cubin"], info["name"])
    drv.ensure_dynamic_shared(func, info["shared"])  # flash-attn etc. need >48KB
    drv.launch(func, g, block, info["shared"], params)
    drv.synchronize()

    # keep Vars alive until the (synchronous) launch is done
    del keepalive, out_vars
    return None
