# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import math
import numpy as np
from .base import Distribution
from ._utils import (_bshape, _full_shape, _as_var, _lgamma, _digamma, _LOG2PI)


class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        # torch parity: batch_shape = broadcast(mu, sigma), event_shape = ()
        self.batch_shape = _bshape(mu, sigma)

    def _sample_impl(self, sample_shape=None):
        # torch semantics: sample() returns
        # sample_shape + batch_shape. Build eps of the FULL shape, then affine-map
        # mu + sigma*eps (parameters broadcast in); the base class detaches.
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        mu = self.mu if isinstance(self.mu, jt.Var) else jt.array(self.mu)
        sigma = self.sigma if isinstance(self.sigma, jt.Var) else jt.array(self.sigma)
        return mu + sigma * jt.randn(shape)

    def rsample(self, sample_shape=None):
        # reparameterized (pathwise) sample: mu + sigma*eps, eps~N(0,1).
        # Keeps the autodiff graph to mu/sigma (do NOT re-wrap Vars in jt.array,
        # which would detach). This is what VAEs/VI backprop through.
        import jittor as jt
        mu = self.mu if isinstance(self.mu, jt.Var) else jt.array(self.mu)
        sigma = self.sigma if isinstance(self.sigma, jt.Var) else jt.array(self.sigma)
        shape = _full_shape(sample_shape, self.batch_shape)
        return mu + sigma * jt.randn(shape)

    def log_prob(self, x):
        import jittor as jt
        var = self.sigma**2
        log_scale = jt.safe_log(self.sigma)
        return -((x-self.mu)**2) / (2*var) - log_scale-np.log(np.sqrt(2*np.pi))

    def cdf(self, x):
        import jittor as jt
        return 0.5 * (1 + jt.erf((x - self.mu) / (self.sigma * np.sqrt(2.0))))

    def entropy(self):
        import jittor as jt
        return 0.5+0.5*np.log(2*np.pi)+jt.safe_log(self.sigma)

    @property
    def mode(self):
        return self.mu

    @property
    def mean(self):
        return self.mu

class Uniform(Distribution):
    def __init__(self,low,high):
        import jittor as jt
        self.low = low
        self.high = high
        # torch parity: batch_shape = broadcast(low, high), event_shape = ()
        self.batch_shape = _bshape(low, high)
        # assert on python scalars only (elementwise high>low not checked for Vars)
        if not isinstance(low, jt.Var) and not isinstance(high, jt.Var):
            assert high > low

    def _sample_impl(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. jittor has no jt.uniform; draw
        # U[0,1) of the FULL shape and affine-map to [low, high) (params broadcast).
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        low = self.low if isinstance(self.low, jt.Var) else jt.array(self.low)
        high = self.high if isinstance(self.high, jt.Var) else jt.array(self.high)
        return low + (high - low) * jt.random(shape)

    def log_prob(self,x):
        # density is 1/(high-low) inside [low,high), else 0 -> log_prob -inf.
        # Elementwise (torch semantics) so it works for batched x / params; a
        # scalar python x still reduces to a scalar.
        import jittor as jt
        if isinstance(x, jt.Var) or isinstance(self.low, jt.Var) or isinstance(self.high, jt.Var):
            x = x if isinstance(x, jt.Var) else jt.array(x)
            lb = -jt.safe_log(self.high - self.low) + jt.zeros_like(x)
            inside = jt.logical_and(x >= self.low, x < self.high)
            return jt.ternary(inside, lb, jt.full_like(x, -math.inf))
        if x < self.low or x >= self.high:
            return -math.inf
        return -jt.safe_log(self.high - self.low)

    def entropy(self):
        import jittor as jt
        return jt.safe_log(self.high - self.low)

class GammaDistribution(Distribution):
    '''
    For now only support gamma distribution.
    '''
    def __init__(self, concentration, rate):
        import jittor as jt
        from jittor import lgamma, igamma
        self.concentration = concentration
        self.rate = rate
        self.lgamma_alpha = lgamma.apply(jt.array([concentration,]))

    def _sample_impl(self, shape):
        from jittor.math_util.gamma import sample_gamma
        return sample_gamma(self.concentration, shape)

    def cdf(self, value):
        from jittor import lgamma, igamma
        return igamma(self.concentration, value)

    def log_prob(self, value):
        import jittor as jt
        return (self.concentration * jt.log(self.rate) +
                (self.concentration - 1) * jt.log(value) -
                self.rate * value - self.lgamma_alpha)

    def mean(self):
        return self.concentration / self.rate

    def mode(self):
        return np.minimum((self.concentration - 1) / self.rate, 1)

    def variance(self):
        return self.concentration / (self.rate * self.rate)

class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate
        # torch parity: batch_shape from the RAW rate (python scalar -> ())
        self.batch_shape = _bshape(rate)

    def _sample_impl(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. inverse-CDF -log(1-U)/rate with
        # U of the FULL shape so rate broadcasts in (was: sample_shape alone, which
        # dropped batch dims and raised a broadcast error for batched rate).
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        u = jt.rand(shape)
        return -jt.safe_log(1 - u) / self.rate

    def log_prob(self, x):
        import jittor as jt
        return jt.safe_log(self.rate) - self.rate * x

    def entropy(self):
        import jittor as jt
        return 1 - jt.safe_log(self.rate)

class Beta(Distribution):
    ''' torch.distributions.Beta(concentration1, concentration0). '''
    def __init__(self, concentration1, concentration0):
        self.concentration1 = _as_var(concentration1)  # alpha
        self.concentration0 = _as_var(concentration0)  # beta
        # torch parity: batch_shape from the RAW args (python scalar -> ())
        self.batch_shape = _bshape(concentration1, concentration0)

    @property
    def _lbeta(self):
        a, b = self.concentration1, self.concentration0
        return _lgamma(a) + _lgamma(b) - _lgamma(a + b)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. Draw the two gammas at the FULL
        # shape (sample_gamma broadcasts the concentration into it) -- was: sample_shape
        # alone, which dropped the batch dims and crashed for batched concentrations.
        from jittor.math_util.gamma import sample_gamma
        a, b = self.concentration1, self.concentration0
        shape = _full_shape(sample_shape, self.batch_shape)
        x = sample_gamma(a, shape)
        y = sample_gamma(b, shape)
        return x / (x + y)

    def log_prob(self, value):
        import jittor as jt
        value = _as_var(value)
        a, b = self.concentration1, self.concentration0
        return (a - 1) * jt.log(value) + (b - 1) * jt.log(1 - value) - self._lbeta

    def entropy(self):
        a, b = self.concentration1, self.concentration0
        return self._lbeta - (a - 1) * _digamma(a) - (b - 1) * _digamma(b) \
            + (a + b - 2) * _digamma(a + b)

    @property
    def mean(self):
        a, b = self.concentration1, self.concentration0
        return a / (a + b)

    @property
    def variance(self):
        a, b = self.concentration1, self.concentration0
        s = a + b
        return a * b / (s * s * (s + 1))

class Gamma(Distribution):
    ''' torch.distributions.Gamma(concentration, rate) -- shape/rate parameterization.
    (The pre-existing GammaDistribution is kept for backward-compat; this adds entropy,
    torch-flexible Var args, and stays differentiable.) '''
    def __init__(self, concentration, rate):
        self.concentration = _as_var(concentration)
        self.rate = _as_var(rate)
        # torch parity: batch_shape from the RAW args (python scalar -> ())
        self.batch_shape = _bshape(concentration, rate)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape (sample_gamma broadcasts the
        # concentration into the FULL shape; rate then broadcasts elementwise).
        from jittor.math_util.gamma import sample_gamma
        shape = _full_shape(sample_shape, self.batch_shape)
        return sample_gamma(self.concentration, shape) / self.rate

    def log_prob(self, value):
        import jittor as jt
        value = _as_var(value)
        c, r = self.concentration, self.rate
        return c * jt.log(r) + (c - 1) * jt.log(value) - r * value - _lgamma(c)

    def entropy(self):
        import jittor as jt
        c, r = self.concentration, self.rate
        return c - jt.log(r) + _lgamma(c) + (1 - c) * _digamma(c)

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate * self.rate)

class LogNormal(Distribution):
    ''' torch.distributions.LogNormal(loc, scale) -- exp of a Normal(loc, scale). '''
    def __init__(self, loc, scale):
        self.loc = _as_var(loc)
        self.scale = _as_var(scale)
        # torch parity: batch_shape from the RAW args (python scalar -> ())
        self.batch_shape = _bshape(loc, scale)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. eps of the FULL shape, then
        # exp(loc + scale*eps) (loc/scale broadcast in) -- was: sample_shape alone,
        # which dropped batch dims and crashed for batched loc/scale.
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        eps = jt.randn(shape)
        return jt.exp(self.loc + self.scale * eps)

    def log_prob(self, value):
        import jittor as jt
        value = _as_var(value)
        log_x = jt.log(value)
        return -0.5 * ((log_x - self.loc) / self.scale) ** 2 \
            - jt.log(self.scale) - 0.5 * _LOG2PI - log_x

    def entropy(self):
        import jittor as jt
        return 0.5 + 0.5 * _LOG2PI + jt.log(self.scale) + self.loc

    @property
    def mean(self):
        import jittor as jt
        return jt.exp(self.loc + self.scale * self.scale / 2)

    @property
    def variance(self):
        import jittor as jt
        s2 = self.scale * self.scale
        return (jt.exp(s2) - 1) * jt.exp(2 * self.loc + s2)
