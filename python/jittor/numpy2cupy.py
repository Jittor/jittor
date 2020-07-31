# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import numpy as np
import cupy as cp

def cvt(a):
    a_pointer, read_only_flag = a.__array_interface__['data']
    aptr=cp.cuda.MemoryPointer(cp.cuda.memory.UnownedMemory(a_pointer,a.size*a.itemsize,a,0),0)
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
