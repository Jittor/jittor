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
import types
import numpy as np
import jittor as jt
from jittor import nn
from jittor.nn import binary_cross_entropy_with_logits
from jittor import lgamma, igamma, digamma
from jittor.math_util.gamma import gamma_grad, sample_gamma


class _Constraint:
    def __init__(self, *args, **kwargs):
        pass

    def check(self, value):
        if isinstance(value, jt.Var):
            return jt.ones(value.shape, dtype="bool")
        return True


class _Real(_Constraint):
    pass


class _Interval(_Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def check(self, value):
        return (value >= self.lower_bound) & (value <= self.upper_bound)


class _GreaterThan(_Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return value > self.lower_bound


class _GreaterThanEq(_Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def check(self, value):
        return value >= self.lower_bound


class _LessThan(_Constraint):
    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, value):
        return value < self.upper_bound


class _DependentProperty(property):
    def __init__(self, fn=None, is_discrete=False, event_dim=None):
        self.is_discrete = is_discrete
        self.event_dim = event_dim
        super().__init__(fn) if fn is not None else super().__init__()

    def __call__(self, fn):
        return type(self)(fn, self.is_discrete, self.event_dim)


def _dependent_property(fn=None, *, is_discrete=False, event_dim=None):
    prop = _DependentProperty(is_discrete=is_discrete, event_dim=event_dim)
    return prop(fn) if fn is not None else prop


class _ConstraintsModule(types.ModuleType):
    Constraint = _Constraint
    _Real = _Real
    real = _Real()
    positive = _GreaterThan(0)
    nonnegative = _GreaterThanEq(0)
    nonnegative_integer = _GreaterThanEq(0)
    positive_integer = _GreaterThan(0)
    unit_interval = _Interval(0, 1)
    simplex = _Constraint()
    lower_cholesky = _Constraint()
    positive_definite = _Constraint()
    boolean = _Constraint()
    real_vector = _Constraint()
    dependent = _Constraint()
    independent = _Constraint()
    dependent_property = staticmethod(_dependent_property)
    greater_than = staticmethod(lambda lower_bound: _GreaterThan(lower_bound))
    greater_than_eq = staticmethod(lambda lower_bound: _GreaterThanEq(lower_bound))
    less_than = staticmethod(lambda upper_bound: _LessThan(upper_bound))
    interval = staticmethod(lambda lower_bound, upper_bound: _Interval(lower_bound, upper_bound))
    half_open_interval = staticmethod(lambda lower_bound, upper_bound: _Interval(lower_bound, upper_bound))
    integer_interval = staticmethod(lambda lower_bound, upper_bound: _Interval(lower_bound, upper_bound))
    cat = staticmethod(lambda constraints, dim=0: _Constraint())
    stack = staticmethod(lambda constraints, dim=0: _Constraint())


constraints = _ConstraintsModule("torch.distributions.constraints")


# ---- torch.distributions SHAPE semantics ----------------------------------
# torch's Distribution.sample(sample_shape) returns
#     sample_shape + batch_shape + event_shape
# with the *batch* dims (broadcast of the parameters) preserved and sample_shape
# PREPENDED. The helpers below give every distribution that contract.
#
# NB jittor has NO 0-d (scalar) Var: jt.zeros(()), jt.randn(()) and reshape(())
# are all rejected at the C++ level (reshape_op.cc), so a scalar parameter -- a
# python float OR jt.array(0.5) -- always materializes as shape (1,), and is
# therefore INDISTINGUISHABLE from a genuine 1-element batch. We resolve this the
# only consistent way jittor can: a parameter with a single element (prod(shape)==1)
# is treated as a SCALAR, i.e. batch_shape = (). Consequences vs real torch:
#   * scalar params + sample_shape=()      -> jittor (1,)  where torch gives ()
#       (jittor has no 0-d, so a length-1 vector is the scalar representation);
#   * scalar params + sample_shape (n,)/(n,m) -> EXACT match (n,) / (n,m);
#   * ALL multi-element batched-parameter cases -> EXACT match with torch.
# The pre-existing code instead used sample_shape AS the whole output shape, which
# silently DROPPED the batch dims and raised a broadcast error the moment the
# parameters were batched -- that is the real gap TASK #12 fixes.

def _norm_sample_shape(sample_shape):
    ''' Normalize a torch-style sample_shape (None / int / tuple / list /
    jt.NanoVector / torch.Size) to a plain tuple of ints. '''
    if sample_shape is None:
        return ()
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(int(s) for s in sample_shape)


def _prod(shape):
    p = 1
    for d in shape:
        p *= d
    return p


def _bshape(*params):
    ''' Broadcast the parameter shapes to obtain batch_shape (torch semantics).
    A single-element parameter (python number, or a length-1 Var that jittor uses
    to stand in for a 0-d scalar) contributes () -- see the module note: jittor has
    no 0-d Var so a scalar and a 1-element batch are indistinguishable, and we pick
    the scalar reading so Normal(jt.array(0.5), ...).sample((n,)) is (n,), not (n,1).'''
    shapes = []
    for p in params:
        if hasattr(p, "shape"):
            s = tuple(p.shape)
            shapes.append(() if _prod(s) == 1 else s)   # length-1 Var == scalar
        else:
            shapes.append(())                            # python number == scalar
    out = ()
    for s in shapes:
        out = _broadcast_two(out, s)
    return out


def _broadcast_two(a, b):
    ''' numpy/torch broadcast of two shape tuples. '''
    res = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da == 1:
            res.append(db)
        elif db == 1 or da == db:
            res.append(da)
        else:
            raise ValueError(f"incompatible parameter shapes for broadcast: {a} vs {b}")
    return tuple(reversed(res))


def _full_shape(sample_shape, batch_shape, event_shape=()):
    ''' torch's sample_shape + batch_shape + event_shape. A scalar (empty
    batch+event) collapses to (1,) because jittor has no 0-d Var. '''
    out = _norm_sample_shape(sample_shape) + tuple(batch_shape) + tuple(event_shape)
    return out if len(out) > 0 else (1,)


def _broadcast_var(value, shape):
    value = value if isinstance(value, jt.Var) else jt.array(value)
    cur = tuple(value.shape)
    if cur == tuple(shape) or not shape:
        return value
    if _prod(cur) == 1:
        value = value.reshape((1,) * len(shape))
    return value.broadcast(shape)


def broadcast_all(*values):
    batch_shape = _bshape(*values)
    return tuple(_broadcast_var(value, batch_shape) for value in values)


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
        # torch parity: sample_shape + batch_shape + event_shape, where for a
        # one-hot draw event_shape = (num_categories,). The cum_probs comparison
        # already produces the one-hot over the last (category) axis.
        shape = _norm_sample_shape(sample_shape) + tuple(self.probs.shape[:-1]) + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r).float()
        return one_hot
    
    def log_prob(self, x):
        # recover the category index from the one-hot, then defer to Categorical.
        # NB jt.argmax (the torch_compat shim) returns a single index Var of shape
        # batch_shape; the old `[0]` assumed the jittor-native (idx, val) 2-tuple and
        # silently grabbed element 0, collapsing the whole result to shape (1,).
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
        # torch parity: returns sample_shape + batch_shape, batch_shape = probs.shape[:-1].
        shape = _norm_sample_shape(sample_shape) + tuple(self.probs.shape[:-1]) + (1,)
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

    @property
    def mode(self):
        return self.probs.argmax(dim=-1)


class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        # torch parity: batch_shape = broadcast(mu, sigma), event_shape = ()
        self.batch_shape = _bshape(mu, sigma)

    def sample(self, sample_shape=None):
        # torch semantics: sample() is non-differentiable (detached) and returns
        # sample_shape + batch_shape. Build eps of the FULL shape, then affine-map
        # mu + sigma*eps (parameters broadcast in); stop_grad to detach.
        shape = _full_shape(sample_shape, self.batch_shape)
        mu = self.mu if isinstance(self.mu, jt.Var) else jt.array(self.mu)
        sigma = self.sigma if isinstance(self.sigma, jt.Var) else jt.array(self.sigma)
        return (mu + sigma * jt.randn(shape)).stop_grad()

    def rsample(self, sample_shape=None):
        # reparameterized (pathwise) sample: mu + sigma*eps, eps~N(0,1).
        # Keeps the autodiff graph to mu/sigma (do NOT re-wrap Vars in jt.array,
        # which would detach). This is what VAEs/VI backprop through.
        mu = self.mu if isinstance(self.mu, jt.Var) else jt.array(self.mu)
        sigma = self.sigma if isinstance(self.sigma, jt.Var) else jt.array(self.sigma)
        shape = _full_shape(sample_shape, self.batch_shape)
        return mu + sigma * jt.randn(shape)

    def log_prob(self, x):
        var = self.sigma**2
        log_scale = jt.safe_log(self.sigma)
        return -((x-self.mu)**2) / (2*var) - log_scale-np.log(np.sqrt(2*np.pi))

    def cdf(self, x):
        return 0.5 * (1 + jt.erf((x - self.mu) / (self.sigma * np.sqrt(2.0))))
    
    def entropy(self):
        return 0.5+0.5*np.log(2*np.pi)+jt.safe_log(self.sigma)

    @property
    def mode(self):
        return self.mu

    @property
    def mean(self):
        return self.mu


class Uniform:
    def __init__(self,low,high):
        self.low = low
        self.high = high
        # torch parity: batch_shape = broadcast(low, high), event_shape = ()
        self.batch_shape = _bshape(low, high)
        # assert on python scalars only (elementwise high>low not checked for Vars)
        if not isinstance(low, jt.Var) and not isinstance(high, jt.Var):
            assert high > low

    def sample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. jittor has no jt.uniform; draw
        # U[0,1) of the FULL shape and affine-map to [low, high) (params broadcast).
        shape = _full_shape(sample_shape, self.batch_shape)
        low = self.low if isinstance(self.low, jt.Var) else jt.array(self.low)
        high = self.high if isinstance(self.high, jt.Var) else jt.array(self.high)
        return low + (high - low) * jt.random(shape)

    def log_prob(self,x):
        # density is 1/(high-low) inside [low,high), else 0 -> log_prob -inf.
        # Elementwise (torch semantics) so it works for batched x / params; a
        # scalar python x still reduces to a scalar.
        if isinstance(x, jt.Var) or isinstance(self.low, jt.Var) or isinstance(self.high, jt.Var):
            x = x if isinstance(x, jt.Var) else jt.array(x)
            lb = -jt.safe_log(self.high - self.low) + jt.zeros_like(x)
            inside = jt.logical_and(x >= self.low, x < self.high)
            return jt.ternary(inside, lb, jt.full_like(x, -math.inf))
        if x < self.low or x >= self.high:
            return -math.inf
        return -jt.safe_log(self.high - self.low)

    def entropy(self):
        return jt.safe_log(self.high - self.low)


class Geometric:
    def __init__(self,p=None,logits=None):
        assert (p is not None) or (logits is not None)
        if p is None:
            self.prob = jt.sigmoid(logits)
            self.logits = logits
        else:
            # assert range on python scalars only (batched Var probs allowed)
            if not isinstance(p, jt.Var):
                assert 0 < p and p < 1
            self.prob = p
            self.logits = -jt.safe_log(1. / p - 1)
        # torch parity: batch_shape = broadcast(prob), event_shape = ()
        self.batch_shape = _bshape(self.prob)

    def sample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. inverse-CDF: floor(log(U)/log(1-p))
        # with U of the FULL shape so prob broadcasts in (was self.probs typo + drop).
        shape = _full_shape(sample_shape, self.batch_shape)
        u = jt.rand(shape)
        return (jt.safe_log(u) / jt.safe_log(-self.prob + 1)).floor_int()

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
    has_rsample = False
    arg_constraints = {}

    def __init__(self, batch_shape=(), event_shape=(), validate_args=None):
        self.batch_shape = tuple(batch_shape)
        self.event_shape = tuple(event_shape)
        self._validate_args = validate_args

    def _extended_shape(self, sample_shape=None):
        return _full_shape(sample_shape, self.batch_shape, self.event_shape)

    def _validate_sample(self, value):
        return None

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
        # torch parity: batch_shape = broadcast(params), event_shape = ().
        # Compute from the RAW arg so a python scalar -> () (torch 0-d), not the
        # (1,) that _as_var/jt.array forces (jittor has no 0-d Var).
        self.batch_shape = _bshape(logits if logits is not None else probs)

    def sample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. Draw U of the FULL shape so
        # probs broadcasts in (was: sample_shape used as the whole output shape,
        # which dropped batch dims and raised a broadcast error for batched probs).
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


def _softplus(z):
    # stable log(1+exp(z)) = max(z,0) + log(1+exp(-|z|))
    return jt.maximum(z, 0.0) + jt.safe_log(1.0 + jt.exp(-jt.abs(z)))


def _log_temperature(temperature):
    if isinstance(temperature, jt.Var):
        return jt.safe_log(temperature)
    return math.log(float(temperature))


def _no_closed_form(cls_name, name):
    raise NotImplementedError(
        f"{cls_name}.{name} has no closed form (torch.distributions raises here "
        f"too). The discrete parent's {name} describes a different random "
        f"variable and would be silently wrong.")


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
        assert (probs is not None) or (logits is not None)
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
        shape = _full_shape(sample_shape, self.batch_shape)
        u = jt.rand(shape)
        logit = self.logits + jt.safe_log(u) - jt.safe_log(1 - u)
        return logit / self.temperature

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()

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
        return jt.sigmoid(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()

    def log_prob(self, value):
        # sigmoid transform of the base distribution:
        #   log p(y) = log p_base(x) - log|dy/dx|,  x = logit(y),
        #   -log|dy/dx| = softplus(x) + softplus(-x)
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
        shape = _norm_sample_shape(sample_shape) + tuple(self.probs.shape)
        u = jt.rand(shape)
        g = -jt.safe_log(-jt.safe_log(u + 1e-20) + 1e-20)
        scores = (self.logits + g) / self.temperature
        return nn.log_softmax(scores, dim=-1)

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()

    def log_prob(self, value):
        # value is a vector of log-probabilities
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
        return jt.exp(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()

    def log_prob(self, value):
        # exp transform of the base distribution: x = log(y),
        # log|dy/dx| summed over the event dim is sum(log y)
        log_value = jt.safe_log(value)
        return self.base_dist.log_prob(log_value) - log_value.sum(-1)

    def entropy(self):
        _no_closed_form(type(self).__name__, "entropy")

    @property
    def mode(self):
        _no_closed_form(type(self).__name__, "mode")


class Exponential(Distribution):
    def __init__(self, rate):
        self.rate = rate
        # torch parity: batch_shape from the RAW rate (python scalar -> ())
        self.batch_shape = _bshape(rate)

    def sample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. inverse-CDF -log(1-U)/rate with
        # U of the FULL shape so rate broadcasts in (was: sample_shape alone, which
        # dropped batch dims and raised a broadcast error for batched rate).
        shape = _full_shape(sample_shape, self.batch_shape)
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
        a, b = self.concentration1, self.concentration0
        shape = _full_shape(sample_shape, self.batch_shape)
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
        # torch parity: batch_shape from the RAW args (python scalar -> ())
        self.batch_shape = _bshape(concentration, rate)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape (sample_gamma broadcasts the
        # concentration into the FULL shape; rate then broadcasts elementwise).
        shape = _full_shape(sample_shape, self.batch_shape)
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
        # torch parity: batch_shape from the RAW rate (python scalar -> ())
        self.batch_shape = _bshape(rate)

    def sample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. Broadcast the rate into the FULL
        # shape before drawing (was: np.broadcast_to(lam, sample_shape), which dropped
        # the batch dims and raised a numpy broadcast error for batched rate).
        shape = _full_shape(sample_shape, self.batch_shape)
        lam = np.broadcast_to(self.rate.numpy(), shape)
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
        # torch parity: last dim is the event; batch_shape = concentration.shape[:-1],
        # event_shape = (concentration.shape[-1],)
        self.batch_shape = tuple(self.concentration.shape[:-1])
        self.event_shape = (int(self.concentration.shape[-1]),)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape + event_shape. Gamma-draw at the
        # FULL shape (concentration broadcasts in) then normalize over the event axis;
        # was: sample_shape alone, which dropped batch dims and crashed when batched.
        a = self.concentration
        shape = _full_shape(sample_shape, self.batch_shape, self.event_shape)
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
        # torch parity: batch_shape from the RAW args (python scalar -> ())
        self.batch_shape = _bshape(loc, scale)

    def rsample(self, sample_shape=None):
        # torch parity: sample_shape + batch_shape. eps of the FULL shape, then
        # exp(loc + scale*eps) (loc/scale broadcast in) -- was: sample_shape alone,
        # which dropped batch dims and crashed for batched loc/scale.
        shape = _full_shape(sample_shape, self.batch_shape)
        eps = jt.randn(shape)
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


class LogisticNormal(Distribution):
    ''' torch.distributions.LogisticNormal(loc, scale).

    This is a lightweight transformed Normal for PyTorch-ecosystem import paths
    (tensordict patches deterministic_sample at import time). For vector events
    it maps through softmax; for scalar events it maps through sigmoid.
    '''
    def __init__(self, loc, scale, validate_args=None):
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

    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()

    def log_prob(self, value):
        # Best-effort inverse transform. This is mainly for compatibility; verl's
        # DataProto/tensordict import path only needs the class to exist.
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
        shape = _full_shape(sample_shape, self.batch_shape, self.event_shape)
        eps = jt.randn(shape)
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
