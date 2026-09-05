# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

# CuPy is an optional bridge for CUDA numpy-code operators. Keep its import out
# of ``import jittor``: importing CuPy scans installed distributions and
# initializes CUDA state even when no numpy-code operator is used.
has_cupy = None
_numpy2cupy_impl = None
_initialization_attempted = False


def _ensure_cupy():
    global has_cupy, _numpy2cupy_impl, _initialization_attempted
    if _initialization_attempted:
        return _numpy2cupy_impl

    # The converter only needs this bridge for CUDA. Avoid probing/importing
    # CuPy on CPU and accelerator-less processes as well.
    import jittor as jt
    if not getattr(jt, "has_cuda", False):
        has_cupy = 0
        _initialization_attempted = True
        return None

    try:
        import cupy as cp
    except Exception:
        has_cupy = 0
        _initialization_attempted = True
        return None

    has_cupy = 1
    import ctypes
    import os

    device_num = 0
    if jt.mpi:
        device_num = jt.mpi.local_rank()
        device_num = device_num % cp.cuda.runtime.getDeviceCount()
    cupy_device = cp.cuda.Device(device_num)
    cupy_device.__enter__()

    def cvt(a):
        a_pointer, read_only_flag = a.__array_interface__['data']
        aptr = cp.cuda.MemoryPointer(
            cp.cuda.memory.UnownedMemory(
                a_pointer, a.size * a.itemsize, a, device_num), 0)
        return cp.ndarray(a.shape, a.dtype, aptr)

    def numpy2cupy_impl(snp, data):
        for key in data:
            if isinstance(data[key], list):
                for i in range(len(data[key])):
                    data[key][i] = cvt(data[key][i])
            elif isinstance(data[key], int):
                pass
            else:
                data[key] = cvt(data[key])

    jt_allocator = ctypes.CDLL(
        os.path.join(jt.compiler.cache_path,
                     "jittor_core" + jt.compiler.extension_suffix),
        os.RTLD_NOW | os.RTLD_GLOBAL)
    malloc = jt_allocator.get_jittor_cuda_malloc()
    free = jt_allocator.get_jittor_cuda_free()
    # Keep allocator handles alive for the process, as in the eager path.
    _numpy2cupy_impl = numpy2cupy_impl
    _initialization_attempted = True
    return _numpy2cupy_impl


def numpy2cupy(snp, data):
    """Convert numpy-code argument arrays to CuPy arrays on first use."""
    impl = _ensure_cupy()
    if impl is not None:
        impl(snp, data)
