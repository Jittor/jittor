# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Native initialization API, composed from single-owner implementations."""

from jittor import Var

from .basic import (
    eye,
    eye_,
    constant,
    constant_,
    zero,
    zero_,
    random_,
    one,
    one_,
    uniform,
    uniform_,
    gauss,
    gauss_,
)

from .scaling import (
    invariant_uniform,
    invariant_uniform_,
    relu_invariant_gauss,
    relu_invariant_gauss_,
    kaiming_uniform_,
    kaiming_normal_,
    xavier_uniform,
    xavier_uniform_,
    xavier_gauss,
    xavier_gauss_,
)

from ._fan import (
    _calculate_fan_in_and_fan_out,
    _fan_for_mode,
    calculate_std,
    calculate_gain,
)

from .truncated import (
    trunc_normal_,
    _no_grad_trunc_normal_,
)

Var.eye_ = eye_
Var.constant_ = constant_
fill = Var.fill_ = constant_
Var.zero_ = zero_
Var.random_ = random_
Var.one_ = one_
Var.uniform_ = uniform_
Var.gauss_ = gauss_
Var.normal_ = gauss_
Var.invariant_uniform_ = invariant_uniform_
Var.relu_invariant_gauss_ = relu_invariant_gauss_
Var.kaiming_uniform_ = kaiming_uniform_
Var.kaiming_normal_ = kaiming_normal_
Var.xavier_uniform_ = xavier_uniform_
Var.xavier_gauss_ = xavier_gauss_
Var.trunc_normal_ = trunc_normal_

__all__ = [
    "eye",
    "eye_",
    "constant",
    "constant_",
    "zero",
    "zero_",
    "random_",
    "one",
    "one_",
    "uniform",
    "uniform_",
    "gauss",
    "gauss_",
    "invariant_uniform",
    "invariant_uniform_",
    "relu_invariant_gauss",
    "relu_invariant_gauss_",
    "calculate_std",
    "kaiming_uniform_",
    "kaiming_normal_",
    "xavier_uniform",
    "xavier_uniform_",
    "xavier_gauss",
    "xavier_gauss_",
    "calculate_gain",
    "trunc_normal_",
    "fill",
]
