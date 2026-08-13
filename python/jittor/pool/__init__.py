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

pool_use_code_op = True

from .core_2d import Pool
from .core_3d import Pool3d, _triple
from .adaptive import (
    AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
)
from .pooling_1d import AdaptiveAvgPool1d, AvgPool1d, MaxPool1d
from .layers import (
    AvgPool2d, AvgPool3d, MaxPool2d, MaxPool3d, _no_dilation, argmax_pool,
    avg_pool2d, max_pool2d, max_pool3d, pool, pool3d,
)

pool2d = pool

from .unpool import MaxUnpool2d, MaxUnpool3d
