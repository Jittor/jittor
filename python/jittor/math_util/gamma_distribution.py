import os

import numpy as np
import jittor as jt
from jittor import nn
from jittor.math_util import lgamma, igamma
from jittor.math_util.gamma import gamma_grad, sample_gamma

class GammaDistribution(object):
    '''
    For now only support gamma distribution.
    '''
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate
        self.lgamma_alpha = lgamma.apply(jt.array([concentration,]))

    def rsample(self, shape):
        return sample_gamma(self.concentration, shape)
    
    def cdf(self, value):
        return igamma(alpha, value)
    
    def log_prob(self, value):
        return (self.concentration * jt.log(self.rate) +
                (self.concentration - 1) * jt.log(value) -
                self.rate * value - self.lgamma_alpha)
    
    def mean(self):
        return self.concentration / self.rate
    
    def mode(self):
        return jt.clamp((self.concentration - 1) / self.rate, min_v=1)
    
    def variance(self):
        return self.concentration / (self.rate * self.rate)

if __name__ == "__main__":
    jt.flags.use_cuda=1
    alpha = 3.
    distribute = GammaDistribution(3., 1.)
    samples = distribute.rsample((6, 6))
    print(samples.mean())
    print(distribute.cdf(samples).max())
    print(distribute.log_prob(samples).mean())
