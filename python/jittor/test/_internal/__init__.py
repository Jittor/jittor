# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Jittor test infrastructure, modeled on ``torch.testing._internal``.

This package is the shared machinery for the modern jittor test suite. It mirrors
PyTorch's proven design so the suite measures real bugs the same way torch's does:

  * :mod:`common_utils`        -- ``JittorTestCase`` (dtype/tensor-aware ``assertEqual``
                                   + tolerance policy), ``make_tensor``, ``parametrize``,
                                   dtype group helpers.
  * :mod:`common_device_type`  -- ``instantiate_device_type_tests`` and the
                                   ``@ops`` / ``@dtypes`` / ``@onlyCPU`` / ``@onlyCUDA`` /
                                   ``@onlyNPU`` parametrization decorators.
  * :mod:`gradcheck`           -- ``gradcheck`` / ``gradgradcheck`` (numerical-vs-analytical
                                   Jacobian), the backward-correctness oracle.
  * :mod:`opinfo.core`         -- ``OpInfo`` / ``SampleInput`` and ufunc subclasses.
  * :mod:`common_methods_invocations` -- ``op_db``, the operator registry.

Generic test templates (``test_ops.py``) consume ``op_db`` to generate forward and
backward tests across every device and dtype an op supports.
"""
