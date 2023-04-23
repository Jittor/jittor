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
