# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Shared array algebra and native-complex bridges."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.dispatch import register_kernel
import numpy as np


def _linalg_errors_raise(array_module):
    """Make a numpy_code callback's array module raise on singular input.

    On CUDA the callback receives CuPy, whose ``linalg.solve``/``inv``/
    ``cholesky`` return NaN for a singular (or non-positive-definite) matrix
    where NumPy raises ``LinAlgError`` -- so the same call raised on CPU and
    went silently wrong on CUDA. CuPy raises the same error under
    ``cupyx.errstate(linalg="raise")``.
    """
    import contextlib
    if getattr(array_module, "__name__", "") != "cupy":
        return contextlib.nullcontext()
    import cupyx
    return cupyx.errstate(linalg="raise")


#: ``(set, get)`` for the thread count of the OpenBLAS numpy's LAPACK runs on,
#: ``False`` when there is none to steer (an MKL numpy, a build without the
#: controls), ``None`` until first looked up.
_OPENBLAS_THREADS = None

#: Below this many units of work (batch * m * n * min(m, n)) a host LAPACK call
#: runs on one OpenBLAS thread, below the second on at most four. Measured on
#: 16 cores inside a Jittor process (complex64, numpy 1.26's OpenBLAS 0.3.23):
#: QR 128x64 took 1.86 ms on the default 16 threads against 0.55 ms on one,
#: solve 64x64 0.48 ms against 0.23 ms, and 16 threads only came out ahead at
#: n = 1024. Small factorizations get nothing from OpenBLAS's threading, and
#: its pool contends with Jittor's own OpenMP team, which is still spinning
#: from the kernel before (the same contention `blas_runtime.cc` describes).
_LAPACK_SERIAL_WORK = 1 << 25
_LAPACK_FEW_WORK = 1 << 28
_LAPACK_FEW_THREADS = 4


#: ``(library, stem, cblas_prefix, suffix)`` of the OpenBLAS numpy runs on --
#: ``openblas_set_num_threads64_`` is ``stem + "_set_num_threads" + suffix``,
#: ``cblas_ztrsm64_`` is ``cblas_prefix + "cblas_ztrsm" + suffix`` -- ``False``
#: when there is none, ``None`` until first looked up.
_OPENBLAS = None


def _openblas():
    global _OPENBLAS
    if _OPENBLAS is not None:
        return _OPENBLAS
    _OPENBLAS = False
    import ctypes
    import glob
    import os
    numpy_dir = os.path.dirname(np.__file__)
    site = os.path.dirname(numpy_dir)
    # numpy's wheels bundle the library here (1.x: `64_` suffix, 2.x: a
    # `scipy_` prefix as well); see `_blas_candidates` in build/compile_extern.
    patterns = (os.path.join(site, "numpy.libs", "*openblas*.so*"),
                os.path.join(numpy_dir, ".libs", "*openblas*.so*"),
                os.path.join(numpy_dir, ".dylibs", "*openblas*.dylib"),
                os.path.join(site, "scipy_openblas*", "lib", "*openblas*.so*"))
    for path in sorted(p for pattern in patterns for p in glob.glob(pattern)):
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for stem, prefix in (("scipy_openblas", "scipy_"), ("openblas", "")):
            for suffix in ("64_", ""):
                if hasattr(library, f"{stem}_set_num_threads{suffix}"):
                    _OPENBLAS = (library, stem, prefix, suffix)
                    return _OPENBLAS
    return _OPENBLAS


def _openblas_threads():
    """``(set, get)`` for the OpenBLAS thread count, or ``False``."""
    global _OPENBLAS_THREADS
    if _OPENBLAS_THREADS is not None:
        return _OPENBLAS_THREADS
    _OPENBLAS_THREADS = False
    found = _openblas()
    if found:
        import ctypes
        library, stem, _prefix, suffix = found
        setter = getattr(library, f"{stem}_set_num_threads{suffix}", None)
        getter = getattr(library, f"{stem}_get_num_threads{suffix}", None)
        if setter is not None and getter is not None:
            setter.argtypes = [ctypes.c_int]
            getter.restype = ctypes.c_int
            _OPENBLAS_THREADS = (setter, getter)
    return _OPENBLAS_THREADS


def _host_trsm(a, b, out, upper):
    """``out = a^-1 b`` for one triangular (M, M) ``a`` with CBLAS ``trsm``,
    on numpy's own OpenBLAS; ``False`` when that library has none (the caller
    then solves in general). Only ``a``'s ``upper`` (or lower) triangle is
    read. Raises ``LinAlgError`` for a zero on the diagonal, as a general
    solve would: ``trsm`` would divide by it silently."""
    found = _openblas()
    t = {"f": "s", "d": "d", "F": "c", "D": "z"}.get(a.dtype.char)
    if not found or t is None or not (a.dtype == b.dtype == out.dtype):
        return False
    import ctypes
    library, _stem, prefix, suffix = found
    trsm = getattr(library, f"{prefix}cblas_{t}trsm{suffix}", None)
    if trsm is None:
        return False
    if not np.all(np.diagonal(a)):
        raise np.linalg.LinAlgError("Singular matrix")
    integer = ctypes.c_int64 if suffix == "64_" else ctypes.c_int
    n, nrhs = b.shape
    a = np.ascontiguousarray(a)
    x = np.array(b, order="C", copy=True)
    one = np.ones(1, dtype=a.dtype)
    # real alpha by value, complex through a pointer
    alpha = (ctypes.c_float(1) if t == "s" else ctypes.c_double(1) if t == "d"
             else ctypes.c_void_p(one.ctypes.data))
    row_major, left, upper_, lower_, no_trans, non_unit = 101, 141, 121, 122, 111, 131
    trsm(ctypes.c_int(row_major), ctypes.c_int(left), ctypes.c_int(upper_ if upper else lower_),
         ctypes.c_int(no_trans), ctypes.c_int(non_unit), integer(n), integer(nrhs), alpha,
         ctypes.c_void_p(a.ctypes.data), integer(n), ctypes.c_void_p(x.ctypes.data),
         integer(nrhs))
    out[...] = x
    return True


def _cuda_trsm(cp, a, b, out, upper):
    """``out = a^-1 b`` for one triangular (M, M) ``a`` on CUDA: one cuBLAS
    ``trsm``, where a general solve is an LU, a solve and a status read. The
    zero-diagonal check that keeps a singular ``a`` raising is the one host
    sync. ``a``'s row-major buffer is ``a^T`` to cuBLAS, so it is solved
    transposed with the other triangle named, which needs no copy of it."""
    from cupyx.scipy.linalg import solve_triangular
    if not bool(cp.all(cp.diagonal(a))):
        raise np.linalg.LinAlgError("Singular matrix")
    out[...] = solve_triangular(a.T, b, trans="T", lower=upper)


def _lapack_threads_for(arrays):
    work = 0
    for a in arrays:
        shape = a.shape
        if len(shape) < 2:
            continue
        m, n = shape[-2], shape[-1]
        batch = a.size // (m * n) if m * n else 0
        work = max(work, batch * m * n * min(m, n))
    if work <= _LAPACK_SERIAL_WORK:
        return 1
    if work <= _LAPACK_FEW_WORK:
        return _LAPACK_FEW_THREADS
    return 0


def _host_lapack(forward):
    """Run a linalg ``numpy_code`` callback with OpenBLAS threads sized to it.

    Only on the host: on CUDA the callback receives CuPy and is left alone.
    The library's own count is put back afterwards, so numpy outside Jittor
    is unaffected.
    """
    import functools

    @functools.wraps(forward)
    def run(array_module, data):
        if array_module is not np:
            return forward(array_module, data)
        threads = _openblas_threads()
        wanted = _lapack_threads_for(data["inputs"]) if threads else 0
        if not wanted:
            return forward(array_module, data)
        setter, getter = threads
        previous = getter()
        if wanted >= previous:
            return forward(array_module, data)
        setter(wanted)
        try:
            return forward(array_module, data)
        finally:
            setter(previous)
    return run


def _cuda_gesv(cp, a, b, out):
    """``out = solve(a, b)`` on CUDA for one (M, M) ``a`` and (M, K) ``b``,
    raising ``LinAlgError`` for a singular ``a`` -- with one host sync. The
    CuPy fallback of the native ``cusolver_solve``/``cusolver_inv`` rows
    below, for a CUDA stack without cuSOLVER.

    CuPy's ``solve`` under ``errstate(linalg="raise")`` checks the status of
    ``getrf`` and of ``getrs`` separately, each as an elementwise kernel, a
    reduction and a blocking read: two host syncs and ~200 us a call for a
    64x64 system, half the call. Only ``getrf``'s status can report a
    singular matrix (``getrs``'s flags an illegal argument), so both go into
    one two-int buffer that is read once, after both routines. That one sync
    is the price of raising rather than returning NaN, which torch pays too.

    ``a``, ``b`` and ``out`` share one dtype; ``a`` and ``b`` are the
    callback's dense row-major operands and are not written. A row-major ``a`` is ``a^T`` to cuSOLVER's column-major view, so
    its copy is factorized as it lies and solved with ``op = T``.
    """
    from cupy.cuda import device
    from cupy_backends.cuda.libs import cublas, cusolver
    t = {"f": "s", "d": "d", "F": "c", "D": "z"}[a.dtype.char]
    n, nrhs = b.shape
    lu = a.copy()
    x = cp.array(b, order="F", copy=True)
    handle = device.get_cusolver_handle()
    ipiv = cp.empty(n, dtype=np.int32)
    info = cp.zeros(2, dtype=np.int32)
    lwork = getattr(cusolver, t + "getrf_bufferSize")(handle, n, n, lu.data.ptr, n)
    work = cp.empty(max(lwork, 1), dtype=a.dtype)
    getattr(cusolver, t + "getrf")(handle, n, n, lu.data.ptr, n, work.data.ptr,
                                  ipiv.data.ptr, info.data.ptr)
    getattr(cusolver, t + "getrs")(handle, cublas.CUBLAS_OP_T, n, nrhs, lu.data.ptr, n,
                                  ipiv.data.ptr, x.data.ptr, n, info.data.ptr + 4)
    out[...] = x
    status = info.get()
    if status[0] > 0:
        raise np.linalg.LinAlgError("Singular matrix")
    if status[0] < 0 or status[1] != 0:
        raise RuntimeError("cuSOLVER %sgetrf/%sgetrs reported an illegal argument "
                           "(devInfo %d, %d)" % (t, t, status[0], status[1]))


def _gesvdj(cp, a):
    """Jacobi SVD of a batch ``a`` (B, M, N) on CUDA, converged to eps^2.

    Returns ``(u, s, v, info)``: reduced factors, V itself, and the
    per-matrix status on the device (0 when converged).

    cuSOLVER's default tolerance (machine eps, relative to ``||A||``) took
    ``I + E``, a near-identity matrix with two singular values 16 eps apart,
    for diagonal after zero sweeps and returned the identity as its vectors:
    the top projector was 0.33 off (torch's Jacobi 0.014, ``gesvd`` exact).
    Accuracy there is set by the gap, not by ``||A||``. Jacobi converges
    quadratically, so eps^2 costs one more sweep: 7 -> 8 for a 128x128
    complex64, 10-15% more time, and every size from 8 to 256 converged.
    CuPy's ``gesvdj`` does not take a tolerance, hence the direct calls.
    """
    from cupy_backends.cuda.libs import cusolver as _cs
    from cupy.cuda import device as _device
    batch, m, n = a.shape
    k = min(m, n)
    complex_input = a.dtype.kind == "c"
    real = np.float32 if a.dtype in (np.float32, np.complex64) else np.float64
    letter = {np.dtype(np.float32): "s", np.dtype(np.float64): "d",
              np.dtype(np.complex64): "c", np.dtype(np.complex128): "z"}[np.dtype(a.dtype)]
    handle = _device.get_cusolver_handle()
    _cs.setStream(handle, cp.cuda.get_current_stream().ptr)
    params = _cs.createGesvdjInfo()
    try:
        _cs.xgesvdjSetTolerance(params, float(np.finfo(real).eps) ** 2)
        # Column-major matrices: a C-ordered (B, N, M) array holds each (M, N)
        # matrix in column-major order.
        af = cp.ascontiguousarray(a.transpose(0, 2, 1))
        s = cp.empty((batch, k), real)
        info = cp.zeros(batch, np.int32)
        if batch > 1 and m <= 32 and n <= 32:
            # One launch for the batch, as torch takes small batches; full
            # U (M, M) and V (N, N), sliced to the reduced factors below.
            u = cp.empty((batch, m, m), a.dtype)
            v = cp.empty((batch, n, n), a.dtype)
            size = getattr(_cs, letter + "gesvdjBatched_bufferSize")(
                handle, 1, m, n, af.data.ptr, m, s.data.ptr, u.data.ptr, m,
                v.data.ptr, n, params, batch)
            work = cp.empty(max(size, 1), a.dtype)
            getattr(_cs, letter + "gesvdjBatched")(
                handle, 1, m, n, af.data.ptr, m, s.data.ptr, u.data.ptr, m,
                v.data.ptr, n, work.data.ptr, size, info.data.ptr, params, batch)
            u = u.transpose(0, 2, 1)[:, :, :k]
            v = v.transpose(0, 2, 1)[:, :, :k]
        else:
            u = cp.empty((batch, k, m), a.dtype)
            v = cp.empty((batch, k, n), a.dtype)
            size = getattr(_cs, letter + "gesvdj_bufferSize")(
                handle, 1, 1, m, n, af[0].data.ptr, m, s[0].data.ptr, u[0].data.ptr, m,
                v[0].data.ptr, n, params)
            work = cp.empty(max(size, 1), a.dtype)
            for b in range(batch):
                getattr(_cs, letter + "gesvdj")(
                    handle, 1, 1, m, n, af[b].data.ptr, m, s[b].data.ptr, u[b].data.ptr, m,
                    v[b].data.ptr, n, work.data.ptr, size, info[b:].data.ptr, params)
            u = u.transpose(0, 2, 1)
            v = v.transpose(0, 2, 1)
    finally:
        _cs.destroyGesvdjInfo(params)
    return u, s, v, info


def _cuda_eigh(cp, a):
    """``eigh(a, UPLO='L')`` of an (..., N, N) batch on CUDA.

    One matrix is cuSOLVER's divide-and-conquer ``syevd``/``heevd``, which
    CuPy's ``eigh`` calls and which is accurate there. CuPy sends a batch to
    Jacobi (``syevjBatched``) at its default tolerance, which -- as for the
    SVD (`_gesvdj`) -- stops on a near-identity matrix before rotating: the
    top eigenprojector of ``I + E`` with a 16 eps gap came back 0.33 off while
    the same matrix alone was exact. A batch of N <= 32 is Jacobi converged to
    eps^2, one launch; larger ones, and any matrix Jacobi did not converge
    on, take ``syevd`` one at a time.
    """
    from cupy_backends.cuda.libs import cusolver as _cs
    from cupy.cuda import device as _device
    if a.ndim == 2:
        return cp.linalg.eigh(a, UPLO="L")
    shape = a.shape
    n = shape[-1]
    flat = a.reshape((-1, n, n))
    batch = flat.shape[0]
    real = np.float32 if a.dtype in (np.float32, np.complex64) else np.float64
    w = cp.empty((batch, n), real)
    v = cp.empty((batch, n, n), a.dtype)
    converged = np.zeros(batch, bool)
    if n <= 32 and batch > 1:
        name = {np.dtype(np.float32): "ssyevj", np.dtype(np.float64): "dsyevj",
                np.dtype(np.complex64): "cheevj", np.dtype(np.complex128): "zheevj"}[np.dtype(a.dtype)]
        handle = _device.get_cusolver_handle()
        _cs.setStream(handle, cp.cuda.get_current_stream().ptr)
        params = _cs.createSyevjInfo()
        try:
            _cs.xsyevjSetTolerance(params, float(np.finfo(real).eps) ** 2)
            # Column-major (B, N, N): each matrix as it is, its lower
            # triangle read, overwritten with the eigenvectors as columns.
            work_a = cp.ascontiguousarray(flat.transpose(0, 2, 1))
            info = cp.zeros(batch, np.int32)
            lower = 0
            size = getattr(_cs, name + "Batched_bufferSize")(
                handle, 1, lower, n, work_a.data.ptr, n, w.data.ptr, params, batch)
            work = cp.empty(max(size, 1), a.dtype)
            getattr(_cs, name + "Batched")(
                handle, 1, lower, n, work_a.data.ptr, n, w.data.ptr, work.data.ptr, size,
                info.data.ptr, params, batch)
            # Rescaled as the SVD's (see `_cuda_svd_reduced`): Jacobi's vectors
            # come back some ten eps from unit norm, which the backward's
            # measured resolution then took for error and zeroed a 16 eps gap.
            vectors = work_a.transpose(0, 2, 1)
            v[...] = vectors / cp.linalg.norm(vectors, axis=-2, keepdims=True)
            converged = cp.asnumpy(info) == 0
        finally:
            _cs.destroySyevjInfo(params)
    for b in np.flatnonzero(~converged):
        w[b], v[b] = cp.linalg.eigh(flat[b], UPLO="L")
    return w.reshape(shape[:-1]), v.reshape(shape)


def _cuda_svd_reduced(cp, a):
    """Reduced ``(u, s, vh)`` of an (..., M, N) batch on CUDA.

    Jacobi (``gesvdj``, see `_gesvdj`) first, as torch does: for the 64-512
    square complex64 matrices of an MPS truncation it is 3x faster than the
    QR-iteration ``gesvd`` CuPy's ``svd`` calls (128x128: 3.1 ms against
    9.2 ms, torch 3.1 ms). A matrix it does not converge on goes to ``gesvd``
    instead -- the same fallback torch takes.

    The singular values are then re-read as ``||a v_i||``, whose error is
    second order in the vectors'. Jacobi's own values carry the rounding of
    every rotation: for ``diag(1, 1 + 2^-19)`` in complex64 one came back an
    ulp off, which is 1/16 of the gap, and the gradient through that gap 3%
    off (LAPACK's SVD, and torch's, exact). Re-read, the values may change
    order by a rounding; such a matrix is sorted again, vectors included.
    """
    shape = a.shape
    m, n = shape[-2:]
    flat = a.reshape((-1, m, n))
    u, s, v, info = _gesvdj(cp, flat)
    # Jacobi's vectors come back with norms some ten eps from 1 (2x2
    # complex64: 20 eps of non-orthogonality, almost all of it scale, where
    # LAPACK's are within one). Rescaled, the backward's measured resolution
    # (decompositions.py `_resolution`) sees the rotation's error only.
    u = u / cp.linalg.norm(u, axis=-2, keepdims=True)
    v = v / cp.linalg.norm(v, axis=-2, keepdims=True)
    # ||a v_i||, stationary at an exact right vector, so its error is second
    # order in v's: within one eps of LAPACK's values where Jacobi's own were
    # 240 eps off (128x128 complex64).
    s = cp.linalg.norm(cp.matmul(flat, v), axis=-2).astype(s.dtype)
    # The status and whether the re-read values still descend, in one read.
    unsorted = (s[:, 1:] > s[:, :-1]).any(axis=-1) if s.shape[-1] > 1 else cp.zeros_like(info, bool)
    status = cp.asnumpy(cp.stack([info != 0, unsorted]))
    for b in np.flatnonzero(status[1] & ~status[0]):
        order = cp.argsort(-s[b])
        s[b], u[b], v[b] = s[b][order], u[b][:, order], v[b][:, order]
    vh = v.conj().transpose(0, 2, 1)
    for b in np.flatnonzero(status[0]):
        u[b], s[b], vh[b] = cp.linalg.svd(flat[b], full_matrices=False)
    k = s.shape[-1]
    return (u.reshape(shape[:-2] + (m, k)), s.reshape(shape[:-2] + (k,)),
            vh.reshape(shape[:-2] + (k, n)))


#: The CUDA rows of the four 2-D linalg kernels below: native cuSOLVER and
#: cuBLAS ops on Jittor's own stream, the way torch calls them from ATen.
#: Through the ``numpy_code`` callbacks a 64x64 complex64 solve paid ~130 us
#: in the executor's Python callback and CuPy's array wrapping on top of the
#: library calls, and QR a host sync torch does not make. A batch of solves
#: takes cuBLAS's batched LU (``cublas_solve_batched``) or, for large matrices,
#: ``cusolver_solve`` one matrix after another; a batch of the other three, or
#: a CUDA toolchain without cuSOLVER (``setup_cusolver``), keeps the callbacks.
_LAPACK_DTYPES = {"float32", "float64", "complex64", "complex128"}


def _cusolver_ops():
    return get_library_ops("cusolver", load=True)


def _cublas_ops():
    return get_library_ops("cublas", load=True)


#: The largest order a batch of solves takes cuBLAS's batched LU for; above
#: it, cuSOLVER's getrf/getrs one matrix after another is faster, as torch
#: also chooses. Measured on an RTX 4090, complex64, a batch of 4.
_CUBLAS_BATCHED_SOLVE_MAX_N = 256


def _has_cublas_solve_batched():
    return getattr(_cublas_ops(), "cublas_solve_batched", None) is not None


def _supports_cuda_solve(a, b):
    # `solve` has broadcast the batch dimensions already, so a batch of `a`
    # and `b` share them.
    if a.dtype != b.dtype or b.ndim != a.ndim or a.ndim < 2 or _cusolver_ops() is None:
        return False
    return (a.ndim == 2 or a.shape[-1] > _CUBLAS_BATCHED_SOLVE_MAX_N
            or not _has_cublas_solve_batched())


def _supports_cuda_solve_batched(a, b):
    # One `getrfBatched`/`getrsBatched` pair takes the whole batch, where the
    # CuPy callback paid the executor's Python callback and array wrapping on
    # every call, and its backward three times.
    return (a.ndim > 2 and b.ndim == a.ndim and a.dtype == b.dtype
            and _has_cublas_solve_batched()
            and (a.shape[-1] <= _CUBLAS_BATCHED_SOLVE_MAX_N or _cusolver_ops() is None))


def _supports_cuda_square(a):
    return a.ndim == 2 and _cusolver_ops() is not None


def _supports_cuda_trsm(a, b, upper):
    return (a.ndim == 2 and b.ndim == 2 and a.dtype == b.dtype
            and getattr(_cublas_ops(), "cublas_trsm", None) is not None)


def _dense(x):
    # A broadcast batch is an expand view; the ops address matrices densely.
    return x if x._storage_is_contiguous() else x.contiguous()


def _cuda_solve(a, b):
    return _cusolver_ops().cusolver_solve(_dense(a), _dense(b))


def _cuda_solve_batched(a, b):
    return _cublas_ops().cublas_solve_batched(_dense(a), _dense(b))


def _cuda_inv(a):
    return _cusolver_ops().cusolver_inv(a)


def _cuda_qr(a):
    q, r = _cusolver_ops().cusolver_qr(a)
    return q, r


def _cuda_solve_triangular(a, b, upper):
    return _cublas_ops().cublas_trsm(a, b, upper)


register_kernel("linalg_solve", "cuda", _cuda_solve, dtypes=_LAPACK_DTYPES,
                supports=_supports_cuda_solve)
register_kernel("linalg_solve", "cuda", _cuda_solve_batched, dtypes=_LAPACK_DTYPES,
                supports=_supports_cuda_solve_batched)
register_kernel("linalg_inv", "cuda", _cuda_inv, dtypes=_LAPACK_DTYPES,
                supports=_supports_cuda_square)
register_kernel("linalg_qr", "cuda", _cuda_qr, dtypes=_LAPACK_DTYPES,
                supports=_supports_cuda_square)
register_kernel("linalg_solve_triangular", "cuda", _cuda_solve_triangular,
                dtypes=_LAPACK_DTYPES, supports=_supports_cuda_trsm)


def _is_cupy(array_module):
    return getattr(array_module, "__name__", "") == "cupy"


def _is_batched(value):
    # vmap's BatchedVar, recognised without importing jittor.vmap on the hot
    # path (and without touching BatchedVar.__getattr__, which raises for
    # unknown names).
    return type(value).__name__ == "BatchedVar" and type(value).__module__ == "jittor.vmap"


def _batching_aware(rule):
    """Route a BatchedVar argument to vmap's rule for this function.

    vmap installs its batching patches on first use, so a reference saved
    before that (``solve = jt.linalg.solve``) used to call the plain function
    with BatchedVar arguments. The check lives in the function itself; the
    rule receives the undecorated function as its base case.
    """
    import functools

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if any(_is_batched(arg) for arg in args):
                # The submodule, not the `jittor.vmap` function attribute.
                from jittor.vmap import dispatch_linalg
                return dispatch_linalg(rule, fn, *args, **kwargs)
            return fn(*args, **kwargs)
        wrapper.__wrapped_linalg__ = fn
        return wrapper
    return decorate


def _reconnect(cached, live):
    """Give a stop-grad'd numpy_code result a real, higher-order-differentiable
    gradient path, without changing its value.

    ``cached`` is a value already computed once inside a ``jt.Function``'s
    ``execute()`` -- ``jt.Function`` tapes and stop-grads its own execute()
    inputs/outputs, so a value cached there (e.g. ``self.mx = inv_via_numpy(x)``)
    is permanently detached from ``x``'s autograd graph. ``live`` is the SAME
    mathematical quantity recomputed via a live, autograd-connected recursive
    call (e.g. calling the public ``inv(x)`` again from inside ``inv``'s own
    ``grad()``, closing over the still-live outer ``x``).

    Returns a Var whose VALUE is exactly ``cached`` (bit-identical --
    ``live - live.detach()`` is a subtraction of a tensor from its own detached
    copy, so it is exactly zero regardless of ``live``'s own numerical
    precision) but whose GRADIENT w.r.t. ``live``'s inputs matches ``live``'s.
    This keeps first-order precision identical to a plain numpy_code op (no
    duplicated rounding from an independent recompute) while still making
    second/third-order derivatives available through the ``live`` path -- the
    standard "stop-gradient trick".
    """
    return cached + (live - live.detach())


def _conj_T(v):
    """Conjugate transpose of the last two axes of a Var (a plain transpose
    for real dtypes)."""
    return v.transpose(-1, -2).conj()


def _eye_like(ref, k):
    """A (k, k) identity Var in `ref`'s (real) dtype, broadcastable over batches."""
    import jittor as jt
    return jt.array(np.eye(k), dtype=_jittor_dtype_name(ref.dtype))


def _diag_of(m):
    """The diagonal of the last two axes of a square Var, differentiably."""
    k = m.shape[-1]
    return (m * _eye_like(m.real if "complex" in _jittor_dtype_name(m.dtype) else m, k)).sum(-1)


def _diag_embed(vec, k):
    """(..., k) -> (..., k, k) with `vec` on the diagonal, differentiably."""
    real = vec.real if "complex" in _jittor_dtype_name(vec.dtype) else vec
    return vec.unsqueeze(-1) * _eye_like(real, k)


def _transpose(x):
    """Batched transpose: swap the last two axes."""
    return np.swapaxes(x, -1, -2)


def _conj_transpose(x):
    """Batched conjugate transpose (the Hermitian adjoint)."""
    return np.conj(np.swapaxes(x, -1, -2))


def _matmul(a, b):
    """Batched matrix product over the last two axes."""
    return np.einsum('...ij,...jk->...ik', a, b)


def _stack_to_complex(x):
    """float ``[..., 2]`` (real, imag) stack -> complex array."""
    return x[..., 0] + 1j * x[..., 1]


def _complex_to_stack(x):
    """complex array -> float ``[..., 2]`` (real, imag) stack."""
    return np.stack([np.real(x), np.imag(x)], axis=-1)


def _is_native_complex(x):
    # A native complex64 Var (NOT an nn.ComplexNumber, which is a plain object).
    import jittor as jt
    return isinstance(x, jt.Var) and "complex" in _jittor_dtype_name(x.dtype)


def _native_to_cn(z):
    # native complex64 [...]  ->  nn.ComplexNumber (differentiable, via P1 bridge)
    import jittor as jt
    from ..nn import ComplexNumber
    return ComplexNumber(jt.nn.view_as_real(z), is_concat_value=True)


def _cn_to_native(cn):
    # nn.ComplexNumber  ->  native complex64/complex128 [...]  (differentiable,
    # via P1 bridge). cn.value is a float32-or-float64 [..., 2] stack;
    # _real2_to_complex rebuilds whichever native complex width matches --
    # NOT the width-fixed _real2_to_complex64, which would silently downcast
    # a complex128 result (e.g. from svd()/qr() on a native complex128 input)
    # back to complex64 on the way out of this bridge.
    from ..nn.functional.complex import _real2_to_complex
    return _real2_to_complex(cn.value)
