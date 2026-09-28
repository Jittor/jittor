# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Mixed-precision training support.

What lives here is the part of mixed precision that is a *policy* rather than a
kernel: choosing and moving the loss scale. Which ops run in which dtype is
decided by ``jt.flags.auto_mixed_precision_level`` and the ``amp_reg`` bits
(see ``src/type/nano_string.h`` and the levels in ``var.cc``), which are
settings on the executor rather than objects.
"""
from .grad_scaler import GradScaler

__all__ = ["GradScaler"]
