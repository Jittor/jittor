# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

has_cupy = 0
try:
    import cupy as cp
    has_cupy = 1
except:
    pass
if has_cupy:
    import jittor as jt
    import os
    import ctypes
    device_num = 0
    if jt.mpi:
        device_num = jt.mpi.local_rank()
    cupy_device = cp.cuda.Device(device_num)
    cupy_device.__enter__()

    def cvt(a):
        a_pointer, read_only_flag = a.__array_interface__['data']
        aptr=cp.cuda.MemoryPointer(cp.cuda.memory.UnownedMemory(a_pointer,a.size*a.itemsize,a, device_num),0)
        a = cp.ndarray(a.shape,a.dtype,aptr)
        return a

    def numpy2cupy(snp, data):
        for key in data:
            if isinstance(data[key], list):
                for i in range(len(data[key])):
                    data[key][i]=cvt(data[key][i])
            elif isinstance(data[key], int):
                pass
            else:
                data[key]=cvt(data[key])

    jt_allocator = ctypes.CDLL(os.path.join(
        jt.compiler.cache_path, 
        "jittor_core"+jt.compiler.extension_suffix), 
        os.RTLD_NOW | os.RTLD_GLOBAL)
    malloc = jt_allocator.get_jittor_cuda_malloc()
    free = jt_allocator.get_jittor_cuda_free()
else:
    def numpy2cupy(snp, data):
        pass