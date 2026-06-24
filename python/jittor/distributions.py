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
from jittor import lgamma, igamma
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
