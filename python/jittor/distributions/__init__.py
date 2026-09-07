# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

"""Probability distributions, with one implementation owner per family."""

# Retain historical dependency attributes for direct import compatibility.
import math
import os
import types
import numpy as np
import jittor as jt
from jittor import nn, lgamma, igamma, digamma
from jittor.nn import binary_cross_entropy_with_logits
from jittor.math_util.gamma import gamma_grad, sample_gamma

from ._constraints import (
    _Constraint,
    _Real,
    _Interval,
    _GreaterThan,
    _GreaterThanEq,
    _LessThan,
    _DependentProperty,
    _dependent_property,
    _ConstraintsModule,
    constraints,
)

from ._utils import (
    _norm_sample_shape,
    _prod,
    _bshape,
    _broadcast_two,
    _full_shape,
    _broadcast_var,
    broadcast_all,
    simple_presum,
    _logsigmoid,
    _softplus,
    _log_temperature,
    _no_closed_form,
    _LOG2PI,
    _as_var,
    _lgamma,
    _digamma,
)

from .base import (
    Distribution,
    Independent,
)

from .discrete import (
    OneHotCategorical,
    Categorical,
    Geometric,
    Bernoulli,
    Poisson,
)

from .continuous import (
    Normal,
    Uniform,
    GammaDistribution,
    Exponential,
    Beta,
    Gamma,
    LogNormal,
)

from .relaxed import (
    LogitRelaxedBernoulli,
    RelaxedBernoulli,
    ExpRelaxedCategorical,
    RelaxedOneHotCategorical,
)

from .multivariate import (
    Dirichlet,
    LogisticNormal,
    MultivariateNormal,
)

from .divergence import (
    kl_divergence,
)

__all__ = [
    'constraints',
    'broadcast_all',
    'simple_presum',
    'Distribution',
    'Independent',
    'OneHotCategorical',
    'Categorical',
    'Geometric',
    'Bernoulli',
    'Poisson',
    'Normal',
    'Uniform',
    'GammaDistribution',
    'Exponential',
    'Beta',
    'Gamma',
    'LogNormal',
    'LogitRelaxedBernoulli',
    'RelaxedBernoulli',
    'ExpRelaxedCategorical',
    'RelaxedOneHotCategorical',
    'Dirichlet',
    'LogisticNormal',
    'MultivariateNormal',
    'kl_divergence',
]
