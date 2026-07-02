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
import os
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

    @staticmethod
    def _load_cudart():
        """Load the CUDA *runtime* (``libcudart``) for memory ops, or ``None``.

        Why the runtime and not the driver here: jittor initialises the device and
        its memory pool through the CUDA *runtime* API. On CUDA 12 that leaves the
        device's primary context in a state where driver-API memory calls
        (``cuMemAlloc`` / ``cuMemcpyDtoD`` / ``cuMemsetD8``) return
        ``CUDA_ERROR_INVALID_CONTEXT`` (201) even though the context is current and
        matches jittor's — yet the *runtime* equivalents (``cudaMalloc`` /
        ``cudaMemcpy`` / ``cudaMemset``) all succeed (they share jittor's runtime
        context cleanly). Kernel *launch* via the driver (``cuLaunchKernel``) is
        unaffected, so we keep the split: driver for module-load+launch, runtime
        for the bounce-buffer memory ops.
        """
        names = ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0",
                 "libcudart.so.11")
        rt = None
        for n in names:
            try:
                rt = ctypes.CDLL(n)
                break
            except OSError:
                rt = None
        if rt is None:
            # fall back to the cudart shipped beside jittor's bundled nvcc
            try:
                import glob
                import jittor.compiler as _c
                nv = getattr(_c, "nvcc_path", "") or ""
                if nv:
                    home = os.path.dirname(os.path.dirname(nv))
                    for so in sorted(glob.glob(os.path.join(home, "lib*",
                                                            "libcudart.so*"))):
                        try:
                            rt = ctypes.CDLL(so)
                            break
                        except OSError:
                            rt = None
            except Exception:
                rt = None
        if rt is None:
            return None
        try:
            c_vp, c_vpp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            rt.cudaMalloc.argtypes = [c_vpp, ctypes.c_size_t]
            rt.cudaMalloc.restype = ctypes.c_int
            rt.cudaFree.argtypes = [c_vp]
            rt.cudaFree.restype = ctypes.c_int
            rt.cudaMemset.argtypes = [c_vp, ctypes.c_int, ctypes.c_size_t]
            rt.cudaMemset.restype = ctypes.c_int
            rt.cudaMemcpy.argtypes = [c_vp, c_vp, ctypes.c_size_t, ctypes.c_int]
            rt.cudaMemcpy.restype = ctypes.c_int
            rt.cudaDeviceSynchronize.argtypes = []
            rt.cudaDeviceSynchronize.restype = ctypes.c_int
        except Exception:
            return None
        return rt

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
        self.rt = self._load_cudart()      # runtime API for memory ops (see below)
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
        self._sig(lib.cuMemFree, [c_vp])
        self._sig(lib.cuMemsetD8, [c_vp, ctypes.c_ubyte, ctypes.c_size_t])
        # device<->device copy (both ptrs are CUdeviceptr-sized integers)
        self._sig(lib.cuMemcpyDtoD,
                  [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t])
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

    def ensure_ctx(self):
        """Make this driver's primary context current on the calling thread.

        Jittor drives the device through the CUDA *runtime* API, which keeps the
        device's primary context current — the very context we
        ``cuDevicePrimaryCtxRetain``ed. But the *current* context is thread-local
        and jittor (or a C++ extension) can leave a different one current, after
        which raw driver calls like ``cuMemAlloc`` fail with
        ``CUDA_ERROR_INVALID_CONTEXT`` (201). Re-asserting our context right before
        a driver op is cheap and idempotent, and keeps us on the same primary
        context jittor allocates from (so pointers stay mutually valid).
        """
        try:
            self.lib.cuCtxSetCurrent(self.ctx)
        except Exception:
            pass

    def alloc(self, nbytes):
        """Allocate device memory via the runtime API (works on jittor's context;
        the driver ``cuMemAlloc`` does not — see :meth:`_load_cudart`)."""
        if self.rt is None:
            # last-resort driver path (kept for completeness; usually fails here)
            self.ensure_ctx()
            ptr = ctypes.c_void_p()
            self.check(self.lib.cuMemAlloc(ctypes.byref(ptr),
                                           ctypes.c_size_t(nbytes)), "cuMemAlloc")
            return int(ptr.value or 0)
        ptr = ctypes.c_void_p()
        r = self.rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if r != 0:
            raise JittorTritonError("cudaMalloc -> cudaError %d" % r)
        return int(ptr.value or 0)

    def free(self, ptr_int):
        try:
            if self.rt is not None:
                self.rt.cudaFree(ctypes.c_void_p(int(ptr_int)))
            else:
                self.lib.cuMemFree(ctypes.c_void_p(int(ptr_int)))
        except Exception:
            pass

    def memset0(self, ptr_int, nbytes):
        if nbytes <= 0:
            return
        r = self.rt.cudaMemset(ctypes.c_void_p(int(ptr_int)), 0,
                               ctypes.c_size_t(int(nbytes)))
        if r != 0:
            raise JittorTritonError("cudaMemset -> cudaError %d" % r)

    #: cudaMemcpyKind.cudaMemcpyDeviceToDevice
    _MEMCPY_D2D = 3

    def copy_dtod(self, dst_ptr_int, src_ptr_int, nbytes):
        if nbytes <= 0:
            return
        r = self.rt.cudaMemcpy(ctypes.c_void_p(int(dst_ptr_int)),
                               ctypes.c_void_p(int(src_ptr_int)),
                               ctypes.c_size_t(int(nbytes)), self._MEMCPY_D2D)
        if r != 0:
            raise JittorTritonError("cudaMemcpy(D2D) -> cudaError %d" % r)

    # ------------------------------------------------------------------ #
    #  guarded bounce-buffer pool (over-read tolerance — see run())
    # ------------------------------------------------------------------ #
    #: reusable device buffers keyed by capacity bucket -> list of free device ptrs
    def _guard_pool(self):
        p = getattr(self, "_gpool", None)
        if p is None:
            p = self._gpool = {}
        return p

    def guard_acquire(self, payload_bytes, guard_bytes):
        """A device buffer holding ``payload_bytes`` data + ``guard_bytes`` slack.

        Backed by a runtime ``cudaMalloc`` (the driver ``cuMemAlloc`` fails on
        jittor's context — see :meth:`_load_cudart`). Buffers are pooled by a
        power-of-two capacity bucket so an autotuner sweep / a model re-running the
        same conv reuses allocations instead of churning malloc/free. The guard
        tail is zeroed so a masked over-read past the payload sees deterministic
        0.0, never NaN; the payload region is overwritten by the caller's copy-in.
        """
        need = int(payload_bytes) + int(guard_bytes)
        # round capacity up to a power of two (>= 4 KiB) to bound pool fragmentation
        cap = 4096
        while cap < need:
            cap <<= 1
        pool = self._guard_pool()
        free = pool.get(cap)
        base = free.pop() if free else self.alloc(cap)
        # zero the guard region right after the payload (payload is overwritten)
        self.memset0(base + int(payload_bytes), cap - int(payload_bytes))
        return base, cap

    def guard_release(self, base, cap):
        self._guard_pool().setdefault(cap, []).append(base)

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
        # Prefer the runtime ``cudaDeviceSynchronize``: it shares jittor's runtime
        # context cleanly and reliably surfaces a launch's async errors, whereas the
        # driver ``cuCtxSynchronize`` can disagree with the runtime's view on
        # jittor's context. Fall back to the driver call if the runtime is absent.
        if self.rt is not None:
            r = self.rt.cudaDeviceSynchronize()
            if r != 0:
                raise JittorTritonError("cudaDeviceSynchronize -> cudaError %d" % r)
            return
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


def _is_tensor(v):
    """True for a launchable tensor arg: a jittor ``Var`` *or* any torch-like
    tensor (duck-typed: has ``data_ptr`` + ``dtype`` + ``shape``).

    Under jittor's torch shim a "torch tensor" *is* a jittor ``Var``, so the
    isinstance branch already covers ``import jittor as torch`` code; the
    duck-typed branch additionally accepts a genuine torch.Tensor (or any
    tensor wrapper) so "torch-compatible triton" works regardless of how the
    tensor object is presented.
    """
    if _is_var(v):
        return True
    return hasattr(v, "data_ptr") and hasattr(v, "dtype") and hasattr(v, "shape")


def _tensor_ptr(v):
    """Device base pointer of a tensor arg (``raw_ptr`` for Var, else ``data_ptr``)."""
    if _is_var(v):
        return int(v.raw_ptr)
    return int(v.data_ptr())


#: bytes per element, by triton type code (the value side of ``_DT``)
_TCODE_BYTES = {
    "fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1,
    "u64": 8, "u32": 4, "u16": 2, "u8": 1, "i1": 1,
}


def _tensor_nbytes(v, ptr_sig):
    """Byte footprint of a tensor arg, from its element count and dtype width.

    ``ptr_sig`` is the ``"*xx"`` triton pointer signature already computed for the
    arg; we strip the ``*`` to size one element. Element count comes from the
    shape (contiguous tensors — which triton requires for these kernels anyway).
    """
    try:
        elsize = _TCODE_BYTES.get(ptr_sig[1:], 0)
        if elsize == 0:
            return 0
        shape = getattr(v, "shape", None)
        if shape is None:
            return 0
        n = 1
        for s in shape:
            n *= int(s)
        return n * elsize
    except Exception:
        return 0


def _tensor_is_cuda(v):
    ic = getattr(v, "is_cuda", None)
    if ic is not None:
        return bool(ic)
    dev = getattr(v, "device", None)
    return dev is not None and "cuda" in str(dev).lower()


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

# --------------------------------------------------------------------------- #
#  over-read guard configuration
# --------------------------------------------------------------------------- #
# Many real-world triton GEMM kernels (notably flex_gemm's split-K /
# masked-implicit-gemm sparse-conv) seed a tile pointer slightly past the end of
# a *small* operand (e.g. ``weight + k_start * BK`` when ``BK > Ci``) and rely on
# the load mask to drop the over-hanging lanes. On a torch caching allocator the
# over-read lands inside the (rounded-up) allocation and is harmless; jittor's
# Vars are exactly sized, so the same masked over-read crosses into an unmapped
# page and faults (cudaErrorIllegalAddress) or returns NaN. To match torch's
# de-facto contract we route every *small* tensor arg through a pooled bounce
# buffer that carries a zeroed trailing guard, then copy the (masked-store) result
# back. Large tensors (feature maps / big outputs) are passed straight through:
# correct kernels never over-read them, and copying them would be wasteful.
#
# GUARD_BYTES — slack appended after each bounced tensor. Bounded by the largest
#   plausible single-tile over-hang: B_tile(<=256) * stride. 256 KiB is generous.
# GUARD_MAX_PAYLOAD — only tensors at/under this size are bounced. Weights and
#   index/segment tables (the operands that get over-read) are far smaller; this
#   keeps multi-MB/GB feature & output tensors on the zero-copy fast path.
GUARD_BYTES = 256 * 1024
GUARD_MAX_PAYLOAD = 64 * 1024 * 1024
#: opt-out hook (set False to disable bounce buffers entirely, e.g. for A/B tests)
GUARD_ENABLE = True


def _make_ast_source(ASTSource, jitfn, signature, constants):
    """Construct a triton ``ASTSource`` across triton versions.

    triton 3.1 : ``ASTSource(fn, signature, constants, attrs)`` where ``attrs``
                 is a ``triton.compiler.AttrsDescriptor()`` (an opaque "no
                 special-case" specialization hint).
    triton 3.2+ : ``AttrsDescriptor`` was removed from ``triton.compiler``; the
                 4th positional became ``attrs`` (a plain dict, default {}), and
                 ``constants`` is passed as the keyword ``constexprs``. The exact
                 keyword names drifted again across 3.3/3.4, so we probe a few
                 conservative call forms and use the first that constructs.

    We never assert divisibility / eq-1 specializations (empty attrs) so the
    compiled kernel matches the conservative arg-passing the jittor launcher
    does (all pointers treated as un-aligned, all ints as generic).
    """
    # 3.1 / 3.2: AttrsDescriptor is re-exported from triton.compiler and
    # ASTSource(fn, signature, constants, attrs) accepts it positionally.
    try:
        from triton.compiler import AttrsDescriptor as _AD
        return ASTSource(jitfn, signature, constants, _AD())
    except Exception:
        pass
    # 3.2+: attrs defaults to None (ASTSource auto-builds an empty descriptor);
    # constants stays positional, but the keyword name drifted across releases.
    last = None
    try:
        return ASTSource(jitfn, signature, constants, None)
    except Exception as e:
        last = e
    for kw in ("constexprs", "constants"):
        try:
            return ASTSource(jitfn, signature, **{kw: constants})
        except Exception as e:        # noqa: PERF203
            last = e
    raise JittorTritonError(
        "could not construct triton ASTSource for kernel %r across known "
        "triton API versions: %s: %s" % (
            getattr(jitfn, "__name__", "?"),
            type(last).__name__ if last else "?", last))


def _compile(jitfn, signature, constants, options=None):
    """Compile (cached) and return a dict with cubin/name/launch metadata.

    ``options`` is an optional dict of launch options forwarded to triton
    (currently ``num_warps`` / ``num_stages``); ``None``/empty lets triton pick
    its defaults.
    """
    triton = real_triton()
    from triton.compiler import ASTSource

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
        try:
            src = _make_ast_source(ASTSource, jitfn, signature, constants)
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

    if os.environ.get("JT_TRITON_PTRTRACE"):
        import sys as _sys
        for _a in list(args) + list(kwargs.values()):
            try:
                if _is_var(_a) and tuple(_a.shape) == (64,) and "int" in str(_a.dtype):
                    print("  PTRTRACE run() ENTRY sorted_idx-like rp=%x" % int(_a.raw_ptr),
                          file=_sys.stderr)
            except Exception:
                pass

    # -- separate launch-meta kwargs (num_warps/...) from real kernel kwargs --
    kwargs = dict(kwargs)
    options = {}
    for m in ("num_warps", "num_stages"):
        if m in kwargs:
            options[m] = kwargs.pop(m)
    # triton launch-only kwargs that are NOT kernel parameters: drop them so they
    # don't get bound to the @triton.jit signature. The set grew across triton
    # versions (3.2 added Hopper warp-specialisation knobs), so keep it broad.
    for m in ("num_ctas", "maxnreg", "warmup", "grid", "extern_libs", "stream",
              "num_buffers_warp_spec", "num_consumer_groups",
              "reg_dec_producer", "reg_inc_consumer",
              "launch_cooperative_grid", "launch_pdl", "profile_scratch"):
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
        if _is_tensor(val):
            if os.environ.get("JT_TRITON_PTRTRACE") and _is_var(val) and \
               tuple(getattr(val, "shape", ())) == (64,) and "int" in str(val.dtype):
                import sys as _sys
                print("  PTRTRACE arg %s rp=%x is_cuda=%r" % (
                    name, int(val.raw_ptr), _tensor_is_cuda(val)), file=_sys.stderr)
            if not _tensor_is_cuda(val):
                # A jittor Var can be host-resident even under use_cuda=1 (e.g. a
                # `jt.array(...)` not yet consumed by a GPU op, or one that a C++
                # extension or a prior sync_all migrated to CPU — which happens
                # between an autotuner's benchmark re-launches). The bridge
                # launches on CUDA and the kernel reads/writes through the arg
                # pointer, so migrate the operand onto the GPU rather than failing.
                # Migrate to GPU via a SEPARATE, fully-materialised copy that we
                # bind to a LOCAL ``val`` only — we do NOT ``val.assign(...)`` the
                # caller's Var. Two reasons:
                #
                #  (1) Correctness of the caller's Var. ``assign(self.cuda())``
                #      rebinds the held Var to a fresh GPU buffer whose raw
                #      ``mem_ptr`` is NOT reliably committed (the cuda-copy is a lazy
                #      graph op; jittor's own ``.numpy()`` re-runs the graph so it
                #      reads correctly, but a *raw* device read of ``mem_ptr`` — e.g.
                #      a C++/CUDA extension that reads ``tensor.data_ptr()`` directly,
                #      like flex_gemm's ``neighbor_map_post_process`` reading a
                #      ``sorted_idx`` index buffer — sees garbage and dereferences it
                #      out of bounds → cudaErrorIllegalAddress). A CPU-resident Var
                #      handed to the bridge (flex_gemm returns its neighbor-cache
                #      index tensors host-resident) would thus be silently corrupted
                #      for every later native consumer. Leaving the caller's Var
                #      untouched and launching on a private copy avoids that entirely.
                #
                #  (2) Inputs don't need write-back. A real *output* is always a
                #      freshly-allocated GPU tensor (under use_cuda=1 ``jt.empty``
                #      lands on the GPU), so it never reaches this CPU-migration
                #      branch; the operands that do (weights / index & segment
                #      tables) are read-only, so a detached GPU copy as the kernel
                #      arg is exactly correct.
                moved = False
                if _is_var(val):
                    try:
                        import jittor as _jt
                        if getattr(_jt.flags, "use_cuda", 0):
                            gpu = val.detach().cuda()
                            gpu.sync(True)
                            if bool(_tensor_is_cuda(gpu)):
                                val = gpu
                                moved = True
                    except Exception:
                        moved = False
                if not moved:
                    raise JittorTritonError(
                        "triton kernel %r received a CPU tensor argument %r; the "
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

    if not any(_is_tensor(v) for (_, _, v) in runtime_vals):
        raise JittorTritonError(
            "triton kernel %r launched with no tensor arguments; the jittor "
            "triton backend needs at least one device tensor (jittor Var or a "
            "torch-shim tensor) to launch on." % kname)

    # ASTSource needs the JITFunction itself (it reads .cache_key), not fn.
    info = _compile(jitfn, signature, constants, options)

    if os.environ.get("JT_TRITON_DEBUG"):
        import sys as _sys
        print("[JT_TRITON_DEBUG] kernel=%s" % kname, file=_sys.stderr)
        try:
            pnames = [getattr(p, "name", "?") for p in (jitfn.params or [])]
            print("  jitfn.params order: %r" % pnames, file=_sys.stderr)
        except Exception:
            pass
        print("  constexpr_names: %r" % sorted(constexpr_names), file=_sys.stderr)
        print("  signature (runtime order): %r" % list(signature.items()),
              file=_sys.stderr)
        print("  constants: %r" % constants, file=_sys.stderr)
        print("  runtime_vals order: %r" % [
            (n, s, ("TENSOR%s" % (tuple(getattr(v, "shape", ())),)) if _is_tensor(v) else v)
            for (n, s, v) in runtime_vals], file=_sys.stderr)
        print("  compiled name=%s n_ptx_params=%d num_warps=%d shared=%d scratch=%d" % (
            info["name"], info["n_ptx_params"], info["num_warps"], info["shared"],
            info["scratch"]), file=_sys.stderr)

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
    _ptrtrace = os.environ.get("JT_TRITON_PTRTRACE")
    if _ptrtrace:
        import sys as _sys
        def _ptd():
            return {n: (int(v.raw_ptr) if _is_var(v) else -1)
                    for (n, s, v) in runtime_vals if _is_tensor(v)}
        print("  PTRTRACE before sync_all: %r" % {k: hex(x) for k, x in _ptd().items()}, file=_sys.stderr)
    jt.sync_all(True)
    if _ptrtrace:
        import sys as _sys
        print("  PTRTRACE after  sync_all: %r" % {k: hex(x) for k, x in _ptd().items()}, file=_sys.stderr)

    # -- pack kernel params (in param order; pointers first only if scratch) - #
    drv = _Driver.get()
    keepalive = []
    cvals = []
    # bounced tensors to copy back + guard buffers to recycle after the launch:
    #   (orig_ptr, bounce_ptr, payload_bytes, bounce_cap)
    bounced = []
    for (name, sig, val) in runtime_vals:
        if sig.startswith("*"):
            if val is None:
                cv = ctypes.c_uint64(0)
            else:
                ptr = _tensor_ptr(val)
                nbytes = _tensor_nbytes(val, sig) if GUARD_ENABLE else 0
                if 0 < nbytes <= GUARD_MAX_PAYLOAD:
                    # bounce through a guarded buffer so a masked over-read past the
                    # operand's end hits zeroed slack instead of an unmapped page.
                    try:
                        bbase, bcap = drv.guard_acquire(nbytes, GUARD_BYTES)
                        drv.copy_dtod(bbase, ptr, nbytes)
                        bounced.append((ptr, bbase, nbytes, bcap))
                        cv = ctypes.c_uint64(bbase)
                    except Exception:
                        # never let the guard path turn a working launch into a
                        # failure — fall back to the raw pointer.
                        cv = ctypes.c_uint64(ptr)
                else:
                    cv = ctypes.c_uint64(ptr)
        else:
            cv = _pack_scalar(sig, val)
        keepalive.append(cv)
        cvals.append(cv)

    # global scratch (rare): triton prepends a scratch pointer as param 0
    n_expected = len(cvals)
    scratch_base = 0
    if info["scratch"] > 0:
        scratch_base = drv.alloc(info["scratch"])
        cvals = [ctypes.c_uint64(scratch_base)] + cvals
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
    if os.environ.get("JT_TRITON_DEBUG"):
        import sys as _sys
        print("  grid=%r block=%r shared=%d n_cvals=%d bounced=%d" % (
            g, block, info["shared"], len(cvals), len(bounced)), file=_sys.stderr)
        _sys.stderr.flush()
    drv.launch(func, g, block, info["shared"], params)
    drv.synchronize()
    if _ptrtrace:
        import sys as _sys
        print("  PTRTRACE after  launch  : %r" % {k: hex(x) for k, x in _ptd().items()}, file=_sys.stderr)

    # Serialise with jittor too. The triton kernel ran on the driver-API default
    # stream while jittor schedules its own ops (and its allocator's frees) on its
    # runtime streams; a bare ``cuCtxSynchronize`` waits for the kernel but does
    # NOT tell jittor the launch is done, so jittor may recycle an arg/output
    # buffer out from under a *subsequent* back-to-back bridge launch (observed as
    # a racy cudaErrorIllegalAddress only when launches are tightly packed, e.g.
    # an autotuner sweep). A second jittor barrier closes that window.
    jt.sync_all(True)

    # -- copy bounced tensors' results back into their Vars, recycle guards ---- #
    # The kernel wrote into the bounce buffers' payload region (a masked store
    # never touches the guard tail), so DtoD the payload back to where the caller
    # will read it. Inputs copy back byte-identical (a no-op functionally).
    if bounced:
        for (orig_ptr, bbase, nbytes, bcap) in bounced:
            try:
                drv.copy_dtod(orig_ptr, bbase, nbytes)
            except Exception:
                pass
        drv.synchronize()
        for (_orig_ptr, bbase, _nbytes, bcap) in bounced:
            drv.guard_release(bbase, bcap)

    if scratch_base:
        drv.free(scratch_base)

    if _ptrtrace:
        import sys as _sys
        print("  PTRTRACE after  final   : %r" % {k: hex(x) for k, x in _ptd().items()}, file=_sys.stderr)

    # keep Vars alive until the (synchronous) launch is done
    del keepalive, out_vars
    return None
