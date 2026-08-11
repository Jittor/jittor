"""Optimization algorithms and learning-rate schedulers.

The package facade preserves the historical ``jittor.optim`` public surface
while implementations live in modules with stable, descriptive paths.
"""

from copy import deepcopy

import jittor as jt
import numpy as np

from .base import Optimizer, opt_grad
from .algorithms import Adan, Adam, AdamW, RMSprop, SGD
from .schedulers import LRScheduler, LambdaLR


_NATIVE_EXPORTS = (
    "jt",
    "np",
    "deepcopy",
    "Optimizer",
    "opt_grad",
    "SGD",
    "RMSprop",
    "Adam",
    "AdamW",
    "Adan",
    "LRScheduler",
    "LambdaLR",
)

__all__ = list(_NATIVE_EXPORTS)


def _refresh_public_exports():
    """Preserve star-import behavior after Torch compatibility adds modules."""
    implementation_modules = {"base", "algorithms", "schedulers"}
    __all__[:] = sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in implementation_modules
    )
