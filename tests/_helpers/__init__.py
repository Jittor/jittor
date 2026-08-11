# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reusable helpers for Jittor's repository-level test suite.

This package is the shared machinery for the modern jittor test suite. It mirrors
PyTorch's proven design so the suite measures real bugs the same way torch's does:

  * :mod:`common`              -- ``JittorTestCase`` (dtype/tensor-aware ``assertEqual``
                                   + tolerance policy), ``make_tensor``, ``parametrize``,
                                   dtype group helpers.
  * :mod:`device_types`        -- ``instantiate_device_type_tests`` and the
                                   ``@ops`` / ``@dtypes`` / ``@onlyCPU`` / ``@onlyCUDA`` /
                                   ``@onlyNPU`` parametrization decorators.
  * :mod:`gradcheck`           -- ``gradcheck`` / ``gradgradcheck`` (numerical-vs-analytical
                                   Jacobian), the backward-correctness oracle.
  * focused modules such as :mod:`logs`, :mod:`assertions`, and :mod:`devices`
    keep reusable behavior out of collectable test modules.
"""
