# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import init, Module
import numpy as np
import math

from ._pool.runtime import bind_runtime as _bind_pool_runtime


pool_use_code_op = True
_bind_pool_runtime(jt)

from ._pool.core_2d import Pool
from ._pool.core_3d import Pool3d, _triple
from ._pool.adaptive import (
    AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
)
from ._pool.pooling_1d import AdaptiveAvgPool1d, AvgPool1d, MaxPool1d
from ._pool.layers import (
    AvgPool2d, AvgPool3d, MaxPool2d, MaxPool3d, _no_dilation, avg_pool2d,
    max_pool2d, max_pool3d, pool, pool3d,
)

pool2d = pool

from ._pool.unpool import MaxUnpool2d, MaxUnpool3d
