# ***************************************************************
# Copyright (c) 2021 Jittor.
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

nvtx_lib_path = os.path.join(jt.compiler.cuda_lib, "libnvToolsExt.so")
nvtx_lib = ctypes.CDLL(nvtx_lib_path, jt.compiler.dlopen_flags)

nvtxRangePushA = nvtx_lib.nvtxRangePushA
nvtxRangePop = nvtx_lib.nvtxRangePop

class nvtx_scope:
    '''
    Add a mark in nvprof timeline

    Example::

        from jittor.utils.nvtx import nvtx_scope
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
