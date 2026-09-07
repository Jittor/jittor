# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


from .base import Distribution
from .continuous import Normal
from ._utils import _as_var, _full_shape, _lgamma, _digamma, _LOG2PI


class Dirichlet(Distribution):
    ''' torch.distributions.Dirichlet(concentration) -- last-dim parameter vector. '''
    def __init__(self, concentration):
        self.concentration = _as_var(concentration)
        # torch parity: last dim is the event; batch_shape = concentration.shape[:-1],
        # event_shape = (concentration.shape[-1],)
        self.batch_shape = tuple(self.concentration.shape[:-1])
        self.event_shape = (int(self.concentration.shape[-1]),)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape + event_shape. Gamma-draw at the
        # FULL shape (concentration broadcasts in) then normalize over the event axis;
        # was: sample_shape alone, which dropped batch dims and crashed when batched.
        from jittor.math_util.gamma import sample_gamma
        a = self.concentration
        shape = _full_shape(sample_shape, self.batch_shape, self.event_shape)
        g = sample_gamma(a, shape)
        return g / g.sum(-1, keepdims=True)

    def log_prob(self, value):
        import jittor as jt
        value = _as_var(value)
        a = self.concentration
        a0 = a.sum(-1)
        return ((a - 1) * jt.log(value)).sum(-1) + _lgamma(a0) - _lgamma(a).sum(-1)

    def entropy(self):
        a = self.concentration
        k = a.shape[-1]
        a0 = a.sum(-1)
        return _lgamma(a).sum(-1) - _lgamma(a0) - (k - a0) * _digamma(a0) \
            - ((a - 1) * _digamma(a)).sum(-1)

    @property
    def mean(self):
        a = self.concentration
        return a / a.sum(-1, keepdims=True)

class LogisticNormal(Distribution):
    ''' torch.distributions.LogisticNormal(loc, scale).

    This is a lightweight transformed Normal for PyTorch-ecosystem import paths
    (tensordict patches deterministic_sample at import time). For vector events
    it maps through softmax; for scalar events it maps through sigmoid.
    '''
    def __init__(self, loc, scale, validate_args=None):
        import jittor as jt
        from jittor import nn
        self.loc = _as_var(loc)
        self.scale = _as_var(scale)
        self.base_dist = Normal(self.loc, self.scale)
        self.batch_shape = tuple(self.loc.shape[:-1]) if self.loc.ndim > 1 else ()
        self.event_shape = (int(self.loc.shape[-1]),) if self.loc.ndim > 0 else ()

        def _logistic_transform(x):
            if x.ndim > 0 and int(x.shape[-1]) > 1:
                return nn.softmax(x, dim=-1)
            return jt.sigmoid(x)
        self.transforms = [_logistic_transform]

    def rsample(self, sample_shape=None):
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        # Best-effort inverse transform. This is mainly for compatibility; verl's
        # DataProto/tensordict import path only needs the class to exist.
        import jittor as jt
        value = _as_var(value)
        eps = 1e-6
        if value.ndim > 0 and int(value.shape[-1]) > 1:
            z = jt.log(jt.maximum(value, eps))
        else:
            v = jt.minimum(jt.maximum(value, eps), 1 - eps)
            z = jt.log(v) - jt.log(1 - v)
        return self.base_dist.log_prob(z)

    @property
    def mean(self):
        x = self.loc
        for transform in self.transforms:
            x = transform(x)
        return x

class MultivariateNormal(Distribution):
    ''' torch.distributions.MultivariateNormal(loc, covariance_matrix). Supports a full
    (k,k) covariance shared across an optional leading batch of loc/value (the common
    case: e.g. a continuous policy with fixed covariance). '''
    def __init__(self, loc, covariance_matrix):
        import jittor as jt
        self.loc = _as_var(loc)
        self.covariance_matrix = _as_var(covariance_matrix)
        self._L = jt.linalg.cholesky(self.covariance_matrix)        # lower-tri (k,k)
        self._Linv = jt.linalg.inv(self._L)
        self._half_logdet = jt.log(jt.diag(self._L)).sum()          # 0.5*log|cov|
        # torch parity: last dim of loc is the event; batch_shape = loc.shape[:-1],
        # event_shape = (k,)
        self.batch_shape = tuple(self.loc.shape[:-1])
        self.event_shape = (int(self.loc.shape[-1]),)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape + event_shape. eps of the FULL
        # shape, color by L, then add loc (loc broadcasts over the leading sample
        # dims) -- was: sample_shape alone, which dropped batch dims and produced a
        # matmul/broadcast error for sample_shape != () (and any batched loc).
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape, self.event_shape)
        eps = jt.randn(shape)
        return self.loc + eps.matmul(self._L.transpose(1, 0))

    def log_prob(self, value):
        value = _as_var(value)
        k = self.loc.shape[-1]
        diff = value - self.loc
        z = diff.matmul(self._Linv.transpose(1, 0))   # solves L z = diff per row
        maha = (z * z).sum(-1)
        return -0.5 * (k * _LOG2PI + 2 * self._half_logdet + maha)

    def entropy(self):
        k = self.loc.shape[-1]
        return 0.5 * k * (1 + _LOG2PI) + self._half_logdet

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        # diagonal of the covariance, broadcast to loc's batch shape (torch semantics)
        import jittor as jt
        return jt.diag(self.covariance_matrix) + jt.zeros_like(self.loc)
