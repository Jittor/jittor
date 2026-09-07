"""Deprecated pooling namespace; canonical implementations belong to jittor.nn.

Historical submodule and pickle globals remain same-object re-exports.
The legacy adaptive-average class intentionally retains its fixed-window rule.
"""

import math
import sys as _sys
from jittor.nn.functional.pooling._state import PoolingStateView as _PoolingStateView
from jittor.nn.functional.pooling import (
    argmax_pool, avg_pool2d, max_pool2d, max_pool3d, pool, pool2d, pool3d,
)
from jittor.nn.functional.pooling.core_3d import _triple
from jittor.nn.functional.pooling.entrypoints import _no_dilation
from jittor.nn.modules.pooling import (
    AdaptiveAvgPool1d, AdaptiveAvgPool3d, AdaptiveMaxPool2d, AdaptiveMaxPool3d,
    AvgPool1d, MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool2d, MaxUnpool3d,
    Pool, Pool3d,
)
from jittor.nn.modules.pooling_legacy import AdaptiveAvgPool2d, AvgPool2d, AvgPool3d

__all__ = [
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AvgPool1d", "AvgPool2d",
    "AvgPool3d", "MaxPool1d", "MaxPool2d", "MaxPool3d", "MaxUnpool2d",
    "MaxUnpool3d", "Pool", "Pool3d", "argmax_pool", "avg_pool2d",
    "max_pool2d", "max_pool3d", "pool", "pool2d", "pool3d", "pool_use_code_op",
]

_sys.modules[__name__].__class__ = _PoolingStateView
