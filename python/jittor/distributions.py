# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import math
import numpy as np
import jittor as jt

def simple_presum(x):
    src = '''
__inline_static__
@python.jittor.auto_parallel(1)
void kernel(int n0, int i0, in0_type* x, in0_type* out, int nl) {
    out[i0*(nl+1)] = 0;
    for (int i=0; i<nl; i++)
        out[i0*(nl+1)+i+1] = out[i0*(nl+1)+i] + x[i0*(nl+1)+i];
}

kernel(in0->num/in0->shape[in0->shape.size()-1], 0, in0_p, out0_p, in0->num);
    '''
    return jt.code(x.shape[:-1]+(x.shape[-1]+1,), x.dtype, [x],
        cpu_src=src, cuda_src=src)


class OneHotCategorical:
    def __init__(self, probs=None, logits=None):
        assert not (probs is None and logits is None)
        if probs is None:
            # cannot align to pytorch
            probs = jt.sigmoid(logits)
        elif logits is None:
            logits = jt.log(probs)
        with jt.no_grad():
            self.probs = probs / probs.sum(-1, True)
            self.logits = logits
            self.cum_probs = simple_presum(probs)
            self.cum_probs_l = self.cum_probs[..., :-1]
            self.cum_probs_r = self.cum_probs[..., 1:]

    def sample(self, sample_shape=[]):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r).float()
        return one_hot
    
    def log_prob(self,x):
        return jt.log(self.probs)[0,x]
    
    def entropy(self):
        min_real = -(math.pow(2,23)-1) / math.pow(2,22) * math.pow(2,127)
        logits = jt.clamp(self.logits,min_v=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)
    
    
class Categorical:
    def __init__(self, probs=None, logits=None):
        assert not (probs is None and logits is None)
        if probs is None:
            # cannot align to pytorch
            probs = jt.sigmoid(logits)
        elif logits is None:
            logits = jt.log(probs)
        with jt.no_grad():
            self.probs = probs / probs.sum(-1, True)
            self.logits = logits
            self.cum_probs = simple_presum(probs)
            self.cum_probs_l = self.cum_probs[..., :-1]
            self.cum_probs_r = self.cum_probs[..., 1:]

    def sample(self, sample_shape=[]):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r)
        index = one_hot.index(one_hot.ndim-1)
        return (one_hot * index).sum(-1)
    
    def log_prob(self, x):
        return jt.log(self.probs)[0,x]
    
    def entropy(self):
        min_real = -(math.pow(2,23)-1) / math.pow(2,22) * math.pow(2,127)
        logits = jt.clamp(self.logits,min_v=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)


class Normal:
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self,sample_shape):
        return jt.normal(mu,sigma,sample_shape)

    def log_prob(self,x):
        var = self.sigma**2
        log_scale = jt.log(self.sigma)
        return -((x-self.mu)**2) / (2*var) - log_scale-np.log(np.sqrt(2*np.pi))
    
    def entropy(self):
        return 0.5+0.5*np.log(2*np.pi)+jt.log(self.sigma)


def kl_divergence(cur_dist,old_dist):
    assert isinstance(cur_dist,type(old_dist))
    if isinstance(cur_dist,Normal):
        vr = (cur_dist.sigma / old_dist.sigma)**2
        t1 = ((cur_dist.mu - old_dist.mu) / old_dist.sigma)**2
        return 0.5*(vr+t1-1-jt.log(vr))
    if isinstance(cur_dist,Categorical) or isinstance(cur_dist,OneHotCategorical):# ?
        t = cur_dist.probs * (cur_dist.logits-old_dist.logits)
        t[jt.array((old_dist.probs == 0))] = math.inf
        t[jt.array((cur_dist.probs == 0))] = 0
        return t.sum(-1)
    