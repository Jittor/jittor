# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import numpy as np
from .base import Distribution
from ._utils import (_norm_sample_shape, simple_presum, _bshape, _full_shape,
                     _logsigmoid, _as_var, _lgamma)


class OneHotCategorical(Distribution):
    def __init__(self, probs=None, logits=None):
        Categorical.__init__(self, probs, logits)

    def _sample_impl(self, sample_shape=[]):
        # torch parity: sample_shape + batch_shape + event_shape, where for a
        # one-hot draw event_shape = (num_categories,). The cum_probs comparison
        # already produces the one-hot over the last (category) axis.
        import jittor as jt
        shape = _norm_sample_shape(sample_shape) + tuple(self.probs.shape[:-1]) + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r).float()
        return one_hot

    def log_prob(self, x):
        # recover the category index from the one-hot, then defer to Categorical.
        # NB jt.argmax (the torch_compat shim) returns a single index Var of shape
        # batch_shape; the old `[0]` assumed the jittor-native (idx, val) 2-tuple and
        # silently grabbed element 0, collapsing the whole result to shape (1,).
        import jittor as jt
        idx = jt.argmax(x, dim=-1)
        if isinstance(idx, tuple):       # jittor-native argmax -> (indices, values)
            idx = idx[0]
        return Categorical.log_prob(self, idx)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)

    @property
    def mode(self):
        return (self.probs == self.probs.max(-1, keepdims=True)).int64()

class Categorical(Distribution):
    def __init__(self, probs=None, logits=None):
        import jittor as jt
        from jittor import nn
        if probs is None and logits is None:
            raise ValueError("Categorical requires probs or logits")
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

    def _sample_impl(self, sample_shape=()):
        # torch parity: returns sample_shape + batch_shape, batch_shape = probs.shape[:-1].
        import jittor as jt
        shape = _norm_sample_shape(sample_shape) + tuple(self.probs.shape[:-1]) + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r)
        index = one_hot.index(one_hot.ndim - 1)
        return (one_hot * index).sum(-1)

    def log_prob(self, x):
        import jittor as jt
        a = self.probs.ndim
        b = x.ndim
        indexes = tuple( f'i{i}' for i in range(b-a+1, b) )
        indexes = indexes + (x,)
        return jt.safe_log(self.probs).getitem(indexes)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)

    @property
    def mode(self):
        return self.probs.argmax(dim=-1)

class Geometric(Distribution):
    def __init__(self,p=None,logits=None):
        import jittor as jt
        if p is None and logits is None:
            raise ValueError("Geometric requires p or logits")
        if p is None:
            self.prob = jt.sigmoid(logits)
            self.logits = logits
        else:
            # assert range on python scalars only (batched Var probs allowed)
            if not isinstance(p, jt.Var):
                if not (0 < p < 1):
                    raise ValueError("Geometric probability p must be between 0 and 1")
            self.prob = p
            self.logits = -jt.safe_log(1. / p - 1)
        # torch parity: batch_shape = broadcast(prob), event_shape = ()
        self.batch_shape = _bshape(self.prob)

    def _sample_impl(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. inverse-CDF: floor(log(U)/log(1-p))
        # with U of the FULL shape so prob broadcasts in (was self.probs typo + drop).
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        u = jt.rand(shape)
        return (jt.safe_log(u) / jt.safe_log(-self.prob + 1)).floor_int()

    def log_prob(self, x):
        import jittor as jt
        return x*jt.safe_log(-self.prob+1)+jt.safe_log(self.prob)

    def entropy(self):
        import jittor as jt
        from jittor.nn import binary_cross_entropy_with_logits
        return binary_cross_entropy_with_logits(jt.array(self.logits),jt.array(self.prob)) / self.prob

class Bernoulli(Distribution):
    ''' torch.distributions.Bernoulli. NB: for Bernoulli the logits->probs map IS
    sigmoid (unlike Categorical, where it is softmax -- see the Categorical fix). '''
    def __init__(self, probs=None, logits=None):
        import jittor as jt
        if probs is None and logits is None:
            raise ValueError("Bernoulli requires probs or logits")
        if logits is not None:
            self.logits = logits
            self.probs = jt.sigmoid(logits)
        else:
            self.probs = probs
            self.logits = jt.safe_log(probs) - jt.safe_log(1 - probs)
        # torch parity: batch_shape = broadcast(params), event_shape = ().
        # Compute from the RAW arg so a python scalar -> () (torch 0-d), not the
        # (1,) that _as_var/jt.array forces (jittor has no 0-d Var).
        self.batch_shape = _bshape(logits if logits is not None else probs)

    def _sample_impl(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. Draw U of the FULL shape so
        # probs broadcasts in (was: sample_shape used as the whole output shape,
        # which dropped batch dims and raised a broadcast error for batched probs).
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        return (jt.rand(shape) < self.probs).float32()

    def log_prob(self, x):
        # x*log(p) + (1-x)*log(1-p), stable via logsigmoid of +/- logits
        return x * _logsigmoid(self.logits) + (1 - x) * _logsigmoid(-self.logits)

    def entropy(self):
        p = self.probs
        return -(p * _logsigmoid(self.logits) + (1 - p) * _logsigmoid(-self.logits))

    @property
    def mode(self):
        return (self.probs >= 0.5).float32()

    @property
    def mean(self):
        return self.probs

class Poisson(Distribution):
    ''' torch.distributions.Poisson(rate). NB: torch defines no closed-form entropy
    (neither do we); sampling is non-reparameterizable (numpy poisson). '''
    def __init__(self, rate):
        self.rate = _as_var(rate)
        # torch parity: batch_shape from the RAW rate (python scalar -> ())
        self.batch_shape = _bshape(rate)

    def _sample_impl(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. Broadcast the rate into the FULL
        # shape before drawing (was: np.broadcast_to(lam, sample_shape), which dropped
        # the batch dims and raised a numpy broadcast error for batched rate).
        import jittor as jt
        shape = _full_shape(sample_shape, self.batch_shape)
        lam = np.broadcast_to(self.rate.numpy(), shape)
        return jt.array(np.random.poisson(lam).astype("float32"))

    def log_prob(self, value):
        import jittor as jt
        value = _as_var(value)
        return value * jt.log(self.rate) - self.rate - _lgamma(value + 1)

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate
