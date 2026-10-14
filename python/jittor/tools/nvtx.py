# ***************************************************************
# Copyright (c) 2023 Jittor.
# All Rights Reserved. 
# Maintainers:
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import os
import ctypes
from jittor_utils.cuda_wheel import NVTX_LIBRARIES

for _name in NVTX_LIBRARIES:
    nvtx_lib_path = jt.compiler.find_cuda_library(_name)
    if nvtx_lib_path:
        break
else:
    raise RuntimeError("NVTX library was not found in the active CUDA installation")
nvtx_lib = ctypes.CDLL(nvtx_lib_path, jt.compiler.dlopen_flags)

nvtxRangePushA = nvtx_lib.nvtxRangePushA
nvtxRangePop = nvtx_lib.nvtxRangePop

class nvtx_scope:
    '''
    Add a mark in nvprof timeline

    Example::

        from jittor.tools.nvtx import nvtx_scope
        with nvtx_scope("model"):
            ...

    '''
    def __init__(self, name):
        self.name = bytes(name, 'utf8')

    def __enter__(self):
        nvtxRangePushA(self.name)

    def __exit__(self, *exc):
        nvtxRangePop()

    def __call__(self, func):
        def inner(*args, **kw):
            with self:
                ret = func(*args, **kw)
            return ret
        return inner
