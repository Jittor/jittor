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
import os
import numpy as np
import jittor as jt
from jittor import nn
from jittor.nn import binary_cross_entropy_with_logits
from jittor import lgamma, igamma, digamma
from jittor.math_util.gamma import gamma_grad, sample_gamma

def simple_presum(x):
    src = '''
__inline_static__
@python.jittor.auto_parallel(1)
void kernel(int n0, int i0, in0_type* x, in0_type* out, int nl) {
    out[i0*(nl+1)] = 0;
    for (int i=0; i<nl; i++)
        out[i0*(nl+1)+i+1] = out[i0*(nl+1)+i] + x[i0*nl+i];
}
kernel(in0->num/in0->shape[in0->shape.size()-1], 0, in0_p, out0_p, in0->shape[in0->shape.size()-1]);
    '''
    return jt.code(x.shape[:-1]+(x.shape[-1]+1,), x.dtype, [x],
        cpu_src=src, cuda_src=src)


class OneHotCategorical:
    def __init__(self, probs=None, logits=None):
        Categorical.__init__(self, probs, logits)

    def sample(self, sample_shape=[]):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r).float()
        return one_hot
    
    def log_prob(self, x):
        x = jt.argmax(x, dim=-1)[0]
        return Categorical.log_prob(self, x)
    
    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)
    
    
class Categorical:
    def __init__(self, probs=None, logits=None):
        assert not (probs is None and logits is None)
        # Align to torch.distributions.Categorical: logits map to probs via SOFTMAX
        # (not sigmoid+renorm), and `logits` are stored as normalized log-probs
        # (log_softmax) so log_prob/entropy are correct. probs/logits are kept
        # differentiable (only the sampling helpers are detached) so policy-gradient
        # methods (PPO/RLHF) can backprop through log_prob/entropy.
        if logits is not None:
            logits = nn.log_softmax(logits, dim=-1)
            probs = jt.exp(logits)
        else:
            probs = probs / probs.sum(-1, True)
            logits = jt.safe_log(probs)
        self.probs = probs
        self.logits = logits
        with jt.no_grad():
            self.cum_probs = simple_presum(self.probs)
            self.cum_probs_l = self.cum_probs[..., :-1]
            self.cum_probs_r = self.cum_probs[..., 1:]

    def sample(self, sample_shape=()):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r)
        index = one_hot.index(one_hot.ndim - 1)
        return (one_hot * index).sum(-1)

    def log_prob(self, x):
        a = self.probs.ndim
        b = x.ndim
        indexes = tuple( f'i{i}' for i in range(b-a+1, b) )
        indexes = indexes + (x,)
        return jt.safe_log(self.probs).getitem(indexes)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)


class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, sample_shape=None):
        return jt.normal(jt.array(self.mu), jt.array(self.sigma),size=sample_shape)

    def log_prob(self, x):
        var = self.sigma**2
        log_scale = jt.safe_log(self.sigma)
        return -((x-self.mu)**2) / (2*var) - log_scale-np.log(np.sqrt(2*np.pi))
    
    def entropy(self):
        return 0.5+0.5*np.log(2*np.pi)+jt.safe_log(self.sigma)


class Uniform:
    def __init__(self,low,high):
        self.low = low
        self.high = high
        assert high > low
    
    def sample(self,sample_shape):
        return jt.uniform(self.low,self.high,sample_shape)
    
    def log_prob(self,x):
        if x < self.low or x >= self.high:
            return math.inf
        return -jt.safe_log(self.high - self.low)
    
    def entropy(self):
        return jt.safe_log(self.high - self.low)


class Geometric:
    def __init__(self,p=None,logits=None):
        assert (p is not None) or (logits is not None)
        assert 0 < p and p < 1
        if p is None:
            self.prob = jt.sigmoid(logits)
            self.logits = logits
        elif logits is None:
            self.prob = p
            self.logits = -jt.safe_log(1. / p - 1)
        
    def sample(self, sample_shape):
        u = jt.rand(sample_shape)
        return (jt.safe_log(u) / (jt.safe_log(-self.probs+1))).floor_int()
    
    def log_prob(self, x):
        return x*jt.safe_log(-self.prob+1)+jt.safe_log(self.prob)
    
    def entropy(self):
        return binary_cross_entropy_with_logits(jt.array(self.logits),jt.array(self.prob)) / self.prob


class GammaDistribution:
    '''
    For now only support gamma distribution.
    '''
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate
        self.lgamma_alpha = lgamma.apply(jt.array([concentration,]))

    def sample(self, shape):
        return sample_gamma(self.concentration, shape)
    
    def cdf(self, value):
        return igamma(self.concentration, value)
    
    def log_prob(self, value):
        return (self.concentration * jt.log(self.rate) +
                (self.concentration - 1) * jt.log(value) -
                self.rate * value - self.lgamma_alpha)
    
    def mean(self):
        return self.concentration / self.rate
    
    def mode(self):
        return np.minimum((self.concentration - 1) / self.rate, 1)
    
    def variance(self):
        return self.concentration / (self.rate * self.rate)


def kl_divergence(cur_dist, old_dist):
    assert isinstance(cur_dist, type(old_dist))
    if isinstance(cur_dist, Normal):
        vr = (cur_dist.sigma / old_dist.sigma)**2
        t1 = ((cur_dist.mu - old_dist.mu) / old_dist.sigma)**2
        return 0.5*(vr+t1-1-jt.safe_log(vr))
    if isinstance(cur_dist, Categorical) or isinstance(cur_dist,OneHotCategorical):
        t = cur_dist.probs * (cur_dist.logits-old_dist.logits)
        return t.sum(-1)
    if isinstance(cur_dist, Uniform):
        res = jt.safe_log((old_dist.high - old_dist.low) / (cur_dist.high - cur_dist.low))
        if old_dist.low > cur_dist.low or old_dist.high < cur_dist.high:
            res = math.inf
        return res
    if isinstance(cur_dist, Geometric):
        return -cur_dist.entropy() - jt.safe_log(-old_dist.prob+1) / cur_dist.prob - old_dist.logits
    if isinstance(cur_dist, Bernoulli):
        # KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
        p, q = cur_dist.probs, old_dist.probs
        return p * (jt.safe_log(p) - jt.safe_log(q)) + (1 - p) * (jt.safe_log(1 - p) - jt.safe_log(1 - q))


def _logsigmoid(z):
    # stable log(sigmoid(z)) = min(z,0) - log(1+exp(-|z|))
    return jt.minimum(z, 0.0) - jt.safe_log(1.0 + jt.exp(-jt.abs(z)))


class Distribution:
    ''' Minimal base class for torch.distributions.Distribution (used for isinstance
    checks and as a common interface). '''
    def sample(self, sample_shape=None):
        raise NotImplementedError
    def rsample(self, sample_shape=None):
        return self.sample(sample_shape)
    def log_prob(self, value):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError


class Bernoulli(Distribution):
    ''' torch.distributions.Bernoulli. NB: for Bernoulli the logits->probs map IS
    sigmoid (unlike Categorical, where it is softmax -- see the Categorical fix). '''
    def __init__(self, probs=None, logits=None):
        assert (probs is not None) or (logits is not None)
        if logits is not None:
            self.logits = logits
            self.probs = jt.sigmoid(logits)
        else:
            self.probs = probs
            self.logits = jt.safe_log(probs) - jt.safe_log(1 - probs)

    def sample(self, sample_shape=None):
        shape = self.probs.shape if not sample_shape else sample_shape
        return (jt.rand(shape) < self.probs).float32()

    def log_prob(self, x):
        # x*log(p) + (1-x)*log(1-p), stable via logsigmoid of +/- logits
        return x * _logsigmoid(self.logits) + (1 - x) * _logsigmoid(-self.logits)

    def entropy(self):
        p = self.probs
        return -(p * _logsigmoid(self.logits) + (1 - p) * _logsigmoid(-self.logits))


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate

    def sample(self, sample_shape=None):
        shape = self.rate.shape if (not sample_shape and hasattr(self.rate, "shape")) else (sample_shape or (1,))
        u = jt.rand(shape)
        return -jt.safe_log(1 - u) / self.rate

    def log_prob(self, x):
        return jt.safe_log(self.rate) - self.rate * x

    def entropy(self):
        return 1 - jt.safe_log(self.rate)


class Independent(Distribution):
    ''' torch.distributions.Independent: reinterpret the last
    `reinterpreted_batch_ndims` batch dims of `base_distribution` as event dims, i.e.
    sum log_prob/entropy over them. Common in RL continuous control:
    Independent(Normal(mu, sigma), 1). '''
    def __init__(self, base_distribution, reinterpreted_batch_ndims):
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def sample(self, sample_shape=None):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=None):
        return self.base_dist.rsample(sample_shape) if hasattr(self.base_dist, "rsample") \
            else self.base_dist.sample(sample_shape)

    def log_prob(self, x):
        lp = self.base_dist.log_prob(x)
        for _ in range(self.reinterpreted_batch_ndims):
            lp = lp.sum(-1)
        return lp

    def entropy(self):
        ent = self.base_dist.entropy()
        for _ in range(self.reinterpreted_batch_ndims):
            ent = ent.sum(-1)
        return ent


# ---- torch.distributions parity: Beta / Gamma / Poisson / Dirichlet / LogNormal /
# ---- MultivariateNormal. log_prob/entropy/mean/variance verified bit-exact (~1e-7)
# ---- vs real torch 2.12. lgamma/digamma are jittor Functions -> stay differentiable.

_LOG2PI = math.log(2 * math.pi)


def _as_var(x):
    return x if isinstance(x, jt.Var) else jt.array(x, dtype="float32")


def _lgamma(x):
    return lgamma.apply(_as_var(x))


def _digamma(x):
    return digamma.apply(_as_var(x))


class Beta(Distribution):
    ''' torch.distributions.Beta(concentration1, concentration0). '''
    def __init__(self, concentration1, concentration0):
        self.concentration1 = _as_var(concentration1)  # alpha
        self.concentration0 = _as_var(concentration0)  # beta

    @property
    def _lbeta(self):
        a, b = self.concentration1, self.concentration0
        return _lgamma(a) + _lgamma(b) - _lgamma(a + b)

    def rsample(self, sample_shape=None):
        a, b = self.concentration1, self.concentration0
        shape = sample_shape if sample_shape else a.shape
        x = sample_gamma(a, shape)
        y = sample_gamma(b, shape)
        return x / (x + y)

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape)

    def log_prob(self, value):
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

    def rsample(self, sample_shape=None):
        shape = sample_shape if sample_shape else self.concentration.shape
        return sample_gamma(self.concentration, shape) / self.rate

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        value = _as_var(value)
        c, r = self.concentration, self.rate
        return c * jt.log(r) + (c - 1) * jt.log(value) - r * value - _lgamma(c)

    def entropy(self):
        c, r = self.concentration, self.rate
        return c - jt.log(r) + _lgamma(c) + (1 - c) * _digamma(c)

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate * self.rate)


class Poisson(Distribution):
    ''' torch.distributions.Poisson(rate). NB: torch defines no closed-form entropy
    (neither do we); sampling is non-reparameterizable (numpy poisson). '''
    def __init__(self, rate):
        self.rate = _as_var(rate)

    def sample(self, sample_shape=None):
        lam = self.rate.numpy()
        if sample_shape:
            lam = np.broadcast_to(lam, sample_shape)
        return jt.array(np.random.poisson(lam).astype("float32"))

    def log_prob(self, value):
        value = _as_var(value)
        return value * jt.log(self.rate) - self.rate - _lgamma(value + 1)

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate


class Dirichlet(Distribution):
    ''' torch.distributions.Dirichlet(concentration) -- last-dim parameter vector. '''
    def __init__(self, concentration):
        self.concentration = _as_var(concentration)

    def rsample(self, sample_shape=None):
        a = self.concentration
        shape = sample_shape if sample_shape else a.shape
        g = sample_gamma(a, shape)
        return g / g.sum(-1, keepdims=True)

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape)

    def log_prob(self, value):
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


class LogNormal(Distribution):
    ''' torch.distributions.LogNormal(loc, scale) -- exp of a Normal(loc, scale). '''
    def __init__(self, loc, scale):
        self.loc = _as_var(loc)
        self.scale = _as_var(scale)

    def rsample(self, sample_shape=None):
        shape = sample_shape if sample_shape else self.loc.shape
        eps = jt.normal(jt.zeros(shape), jt.ones(shape))
        return jt.exp(self.loc + self.scale * eps)

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        value = _as_var(value)
        log_x = jt.log(value)
        return -0.5 * ((log_x - self.loc) / self.scale) ** 2 \
            - jt.log(self.scale) - 0.5 * _LOG2PI - log_x

    def entropy(self):
        return 0.5 + 0.5 * _LOG2PI + jt.log(self.scale) + self.loc

    @property
    def mean(self):
        return jt.exp(self.loc + self.scale * self.scale / 2)

    @property
    def variance(self):
        s2 = self.scale * self.scale
        return (jt.exp(s2) - 1) * jt.exp(2 * self.loc + s2)


class MultivariateNormal(Distribution):
    ''' torch.distributions.MultivariateNormal(loc, covariance_matrix). Supports a full
    (k,k) covariance shared across an optional leading batch of loc/value (the common
    case: e.g. a continuous policy with fixed covariance). '''
    def __init__(self, loc, covariance_matrix):
        self.loc = _as_var(loc)
        self.covariance_matrix = _as_var(covariance_matrix)
        self._L = jt.linalg.cholesky(self.covariance_matrix)        # lower-tri (k,k)
        self._Linv = jt.linalg.inv(self._L)
        self._half_logdet = jt.log(jt.diag(self._L)).sum()          # 0.5*log|cov|

    def rsample(self, sample_shape=None):
        shape = sample_shape if sample_shape else self.loc.shape
        eps = jt.normal(jt.zeros(shape), jt.ones(shape))
        return self.loc + eps.matmul(self._L.transpose(1, 0))

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape)

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
        return jt.diag(self.covariance_matrix) + jt.zeros_like(self.loc)
