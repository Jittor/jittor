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
from .base import Distribution
from .discrete import Bernoulli, Categorical, OneHotCategorical
from ._utils import (_bshape, _full_shape, _norm_sample_shape, _log_temperature,
                     _softplus, _no_closed_form)


class LogitRelaxedBernoulli(Distribution):
    ''' torch.distributions.LogitRelaxedBernoulli.

    The relaxed Bernoulli *in logit space*: samples are unbounded reals, and
    ``sigmoid`` of them is what :class:`RelaxedBernoulli` returns. This is a
    distinct distribution, not an alias of RelaxedBernoulli -- aliasing the two
    made every ``LogitRelaxedBernoulli`` sample come back already squashed into
    (0, 1) and every ``log_prob`` answer the wrong density.
    '''
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        import jittor as jt
        if probs is None and logits is None:
            raise ValueError("RelaxedBernoulli requires probs or logits")
        self.temperature = temperature
        if logits is not None:
            self.logits = logits
            self.probs = jt.sigmoid(logits)
        else:
            self.probs = probs
            self.logits = jt.safe_log(probs) - jt.safe_log(1 - probs)
        self.batch_shape = _bshape(logits if logits is not None else probs)
        self.event_shape = ()

    def rsample(self, sample_shape=None):
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        u = jt.rand(shape)
        logit = self.logits + jt.safe_log(u) - jt.safe_log(1 - u)
        return logit / self.temperature

    def log_prob(self, value):
        # log T + diff - 2*softplus(diff), diff = logits - T*value
        diff = self.logits - value * self.temperature
        return _log_temperature(self.temperature) + diff - 2 * _softplus(diff)

    def entropy(self):
        _no_closed_form(type(self).__name__, "entropy")

class RelaxedBernoulli(Bernoulli):
    ''' torch.distributions.RelaxedBernoulli: sigmoid of a
    :class:`LogitRelaxedBernoulli`, so samples live in (0, 1). '''
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self.temperature = temperature
        super().__init__(probs=probs, logits=logits)
        self.base_dist = LogitRelaxedBernoulli(
            temperature, probs=probs, logits=logits)

    def rsample(self, sample_shape=None):
        import jittor as jt
        return jt.sigmoid(self.base_dist.rsample(sample_shape))

    def _sample_impl(self, sample_shape=None):
        # Must be spelled out: this class derives from the DISCRETE Bernoulli,
        # whose concrete _sample_impl would otherwise shadow the base class's
        # "fall back to rsample" rule and make sample() return a hard 0/1 draw
        # instead of a relaxed one in (0, 1).
        return self.rsample(sample_shape)

    def log_prob(self, value):
        # sigmoid transform of the base distribution:
        #   log p(y) = log p_base(x) - log|dy/dx|,  x = logit(y),
        #   -log|dy/dx| = softplus(x) + softplus(-x)
        import jittor as jt
        x = jt.safe_log(value) - jt.safe_log(1 - value)
        return self.base_dist.log_prob(x) + _softplus(x) + _softplus(-x)

    def entropy(self):
        _no_closed_form(type(self).__name__, "entropy")

    @property
    def mean(self):
        _no_closed_form(type(self).__name__, "mean")

    @property
    def mode(self):
        _no_closed_form(type(self).__name__, "mode")

class ExpRelaxedCategorical(Distribution):
    ''' torch.distributions.relaxed_categorical.ExpRelaxedCategorical: the
    relaxed one-hot categorical in *log* space. Samples are log-probability
    vectors (they exponentiate to the simplex). '''
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self.temperature = temperature
        self._categorical = Categorical(probs=probs, logits=logits)
        self.probs = self._categorical.probs
        self.logits = self._categorical.logits
        self.batch_shape = tuple(self.probs.shape[:-1])
        self.event_shape = (self.probs.shape[-1],)

    def rsample(self, sample_shape=None):
        import jittor as jt
        from jittor import nn
        shape = _norm_sample_shape(sample_shape) + tuple(self.probs.shape)
        u = jt.rand(shape)
        g = -jt.safe_log(-jt.safe_log(u + 1e-20) + 1e-20)
        scores = (self.logits + g) / self.temperature
        return nn.log_softmax(scores, dim=-1)

    def log_prob(self, value):
        # value is a vector of log-probabilities
        from jittor import nn
        K = self.probs.shape[-1]
        log_scale = (math.lgamma(K)
                     + (K - 1) * _log_temperature(self.temperature))
        score = self.logits - value * self.temperature
        score = nn.log_softmax(score, dim=-1).sum(-1)
        return score + log_scale

    def entropy(self):
        _no_closed_form(type(self).__name__, "entropy")

class RelaxedOneHotCategorical(OneHotCategorical):
    ''' torch.distributions.RelaxedOneHotCategorical: exp of an
    :class:`ExpRelaxedCategorical`, so samples are points on the simplex.

    The discrete ``OneHotCategorical.log_prob`` it used to inherit reads the
    argmax of a *relaxed* (non-one-hot) sample and returns the categorical mass
    of that index -- a different, silently wrong number.
    '''
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self.temperature = temperature
        super().__init__(probs=probs, logits=logits)
        self.base_dist = ExpRelaxedCategorical(
            temperature, probs=probs, logits=logits)

    def rsample(self, sample_shape=None):
        import jittor as jt
        return jt.exp(self.base_dist.rsample(sample_shape))

    def _sample_impl(self, sample_shape=None):
        # As in RelaxedBernoulli: the discrete OneHotCategorical._sample_impl
        # would otherwise shadow the rsample fallback and hand back a hard
        # one-hot vector instead of a point on the simplex.
        return self.rsample(sample_shape)

    def log_prob(self, value):
        # exp transform of the base distribution: x = log(y),
        # log|dy/dx| summed over the event dim is sum(log y)
        import jittor as jt
        log_value = jt.safe_log(value)
        return self.base_dist.log_prob(log_value) - log_value.sum(-1)

    def entropy(self):
        _no_closed_form(type(self).__name__, "entropy")

    @property
    def mode(self):
        _no_closed_form(type(self).__name__, "mode")
