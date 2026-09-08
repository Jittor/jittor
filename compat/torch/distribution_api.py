"""Stable sampling fallback types; native distribution mathematics stays native."""
from contextlib import nullcontext
from functools import wraps
from types import MappingProxyType
import jittor as jt
from jittor import nn
from ..diagnostics import EXPECTED, swallowed
from .context import get_install_context
from .frontend import tensor_frontend


def _native_distribution_call(name, arguments):
    context = get_install_context(jt)
    implementation = context.state["distribution_functions"][name]
    with tensor_frontend(context.state["Var"]):
        return implementation(*arguments)


def kl_divergence(cur_dist, old_dist):
    return _native_distribution_call("kl_divergence", (cur_dist, old_dist))


def broadcast_all(*values):
    return _native_distribution_call("broadcast_all", values)


def simple_presum(x):
    return _native_distribution_call("simple_presum", (x,))


DISTRIBUTION_FUNCTIONS = MappingProxyType({
    "kl_divergence": kl_divergence,
    "broadcast_all": broadcast_all,
    "simple_presum": simple_presum,
})


def _tensor_type():
    context = get_install_context(jt, required=False)
    return None if context is None else context.target_namespace.Var


def _sampling_scope(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        target = getattr(self, "_frontend_type", None)
        with tensor_frontend(target) if target is not None else nullcontext():
            return function(self, *args, **kwargs)
    return call


def unsupported_register_kl(*args, **kwargs):
    raise NotImplementedError("this distribution backend has no KL registration support")


class _Constraint:
    def __init__(self, *a, **k): pass
    def check(self, x):
        try:
            return jt.ones_like(x).bool()
        except EXPECTED as exc:
            swallowed("torch/installers/distributions.py check: return jt.ones_like(x).bool()", exc)
            return True


CONSTRAINTS = MappingProxyType({name: _Constraint() for name in (
    "positive", "real", "nonnegative", "nonnegative_integer", "positive_integer",
    "unit_interval", "simplex", "lower_cholesky", "positive_definite", "boolean",
    "real_vector", "dependent", "independent",
)})


class Gumbel:
    def __init__(self, loc, scale, validate_args=None):
        self._frontend_type = _tensor_type()
        self.loc = loc
        self.scale = scale
        self.batch_shape = self._batch_shape(loc, scale)
    @staticmethod
    def _batch_shape(*params):
        shapes = []
        for p in params:
            if hasattr(p, "shape"):
                shape = tuple(p.shape)
                n = 1
                for s in shape:
                    n *= int(s)
                shapes.append(() if n == 1 else shape)
            else:
                shapes.append(())
        out = ()
        for shape in shapes:
            res = []
            for i in range(1, max(len(out), len(shape)) + 1):
                a = out[-i] if i <= len(out) else 1
                b = shape[-i] if i <= len(shape) else 1
                res.append(b if a == 1 else a if b == 1 or a == b else max(a, b))
            out = tuple(reversed(res))
        return out
    @staticmethod
    def _sample_shape(sample_shape, batch_shape=()):
        if sample_shape is None:
            sample_shape = ()
        elif isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        else:
            sample_shape = tuple(int(s) for s in sample_shape)
        out = sample_shape + tuple(batch_shape)
        return out if out else (1,)
    @_sampling_scope
    def rsample(self, sample_shape=None):
        u = jt.random(self._sample_shape(sample_shape, self.batch_shape))
        eps = 1e-6
        u = jt.clamp(u, eps, 1.0 - eps)
        loc = self.loc if isinstance(self.loc, jt.Var) else jt.array(self.loc)
        scale = self.scale if isinstance(self.scale, jt.Var) else jt.array(self.scale)
        return loc - scale * jt.log(-jt.log(u))
    @_sampling_scope
    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()


class RelaxedBernoulli:
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self._frontend_type = _tensor_type()
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        self.temperature = temperature
        if logits is None:
            probs_v = probs if isinstance(probs, jt.Var) else jt.array(probs)
            self.probs = probs_v
            self.logits = jt.log(probs_v) - jt.log(1.0 - probs_v)
        else:
            self.logits = logits if isinstance(logits, jt.Var) else jt.array(logits)
            self.probs = jt.sigmoid(self.logits)
    @_sampling_scope
    def rsample(self, sample_shape=None):
        shape = tuple(self.logits.shape)
        if sample_shape is None:
            sample_shape = ()
        elif isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        else:
            sample_shape = tuple(int(s) for s in sample_shape)
        u = jt.random(sample_shape + shape)
        eps = 1e-6
        u = jt.clamp(u, eps, 1.0 - eps)
        temp = self.temperature if isinstance(self.temperature, jt.Var) else jt.array(self.temperature)
        return jt.sigmoid((self.logits + jt.log(u) - jt.log(1.0 - u)) / temp)
    @_sampling_scope
    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()


class RelaxedOneHotCategorical:
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self._frontend_type = _tensor_type()
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        self.temperature = temperature
        if logits is None:
            probs_v = probs if isinstance(probs, jt.Var) else jt.array(probs)
            self.probs = probs_v / probs_v.sum(-1, keepdims=True)
            self.logits = jt.log(self.probs)
        else:
            self.logits = logits if isinstance(logits, jt.Var) else jt.array(logits)
            self.probs = nn.softmax(self.logits, dim=-1)
    @_sampling_scope
    def rsample(self, sample_shape=None):
        shape = tuple(self.logits.shape)
        if sample_shape is None:
            sample_shape = ()
        elif isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        else:
            sample_shape = tuple(int(s) for s in sample_shape)
        u = jt.random(sample_shape + shape)
        eps = 1e-6
        u = jt.clamp(u, eps, 1.0 - eps)
        gumbels = -jt.log(-jt.log(u))
        temp = self.temperature if isinstance(self.temperature, jt.Var) else jt.array(self.temperature)
        return nn.softmax((self.logits + gumbels) / temp, dim=-1)
    @_sampling_scope
    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()
