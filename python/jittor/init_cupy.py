# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import cupy
import os
import ctypes

jt_allocator = ctypes.CDLL(os.path.join(
    jt.compiler.cache_path, 
    "jittor_core"+jt.compiler.extension_suffix), 
    os.RTLD_NOW | os.RTLD_GLOBAL)
malloc = jt_allocator.get_jittor_cuda_malloc()
free = jt_allocator.get_jittor_cuda_free()