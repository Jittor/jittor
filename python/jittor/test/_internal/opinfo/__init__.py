# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""OpInfo metadata objects for jittor's operator test database."""
from .core import (  # noqa: F401
    SampleInput, OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo,
    DecorateInfo, skip, xfail,
)
