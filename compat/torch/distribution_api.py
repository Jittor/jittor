"""Stable sampling fallback types; native distribution mathematics stays native."""
from contextlib import nullcontext
from functools import wraps
from types import MappingProxyType
from typing import Any, ContextManager, cast
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
        scope: ContextManager[Any]
        if target is not None:
            scope = tensor_frontend(target)
        else:
            scope = nullcontext()
        with scope:
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
        u = cast(Any, jt.random(self._sample_shape(sample_shape, self.batch_shape)))
        eps = 1e-6
        u = jt.clamp(u, eps, 1.0 - eps)
        loc = self.loc if isinstance(self.loc, jt.Var) else jt.array(self.loc)
        scale = self.scale if isinstance(self.scale, jt.Var) else jt.array(self.scale)
        return cast(Any, loc - cast(Any, scale) * cast(Any, jt.log(-cast(Any, jt.log(u)))))
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
            probs_v = cast(Any, probs if isinstance(probs, jt.Var) else jt.array(probs))
            self.probs = probs_v
            self.logits = cast(Any, jt.log(probs_v) - cast(Any, jt.log(1.0 - cast(Any, probs_v))))
        else:
            self.logits = cast(Any, logits if isinstance(logits, jt.Var) else jt.array(logits))
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
        u = cast(Any, jt.random(sample_shape + shape))
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
            probs_v = cast(Any, probs if isinstance(probs, jt.Var) else jt.array(probs))
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
        u = cast(Any, jt.random(sample_shape + shape))
        eps = 1e-6
        u = jt.clamp(u, eps, 1.0 - eps)
        gumbels = cast(Any, -cast(Any, jt.log(-cast(Any, jt.log(u)))))
        temp = self.temperature if isinstance(self.temperature, jt.Var) else jt.array(self.temperature)
        return nn.softmax((self.logits + gumbels) / temp, dim=-1)
    @_sampling_scope
    def sample(self, sample_shape=None):
        return self.rsample(sample_shape).stop_grad()


class Transform:
    """Minimal torch.distributions.transforms.Transform base class."""
    bijective = False
    domain = CONSTRAINTS["real"]
    codomain = CONSTRAINTS["real"]

    def __init__(self, cache_size=0):
        self._cache_size = cache_size

    def __call__(self, x):
        return self._call(x)

    def _call(self, x):
        raise NotImplementedError

    def _inverse(self, y):
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        raise NotImplementedError

    @property
    def inv(self):
        return _InverseTransform(self)


class _InverseTransform(Transform):
    def __init__(self, transform):
        super().__init__(cache_size=getattr(transform, "_cache_size", 0))
        self._inv = transform
        self.bijective = transform.bijective
        self.domain = transform.codomain
        self.codomain = transform.domain

    def _call(self, x):
        return self._inv._inverse(x)

    def _inverse(self, y):
        return self._inv._call(y)

    def log_abs_det_jacobian(self, x, y):
        return -self._inv.log_abs_det_jacobian(y, x)

    @property
    def inv(self):
        return self._inv


class AffineTransform(Transform):
    """Affine bijection y = loc + scale * x."""
    bijective = True

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self._event_dim = event_dim

    @property
    def sign(self):
        if isinstance(self.scale, jt.Var):
            return jt.sign(self.scale)
        return 1 if self.scale >= 0 else -1

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        scale = self.scale if isinstance(self.scale, jt.Var) else jt.array(self.scale)
        result = jt.log(jt.abs(scale))
        if self._event_dim > 0:
            for _ in range(self._event_dim):
                result = result.sum(-1)
        return result


class ComposeTransform(Transform):
    """Composition of a sequence of transforms."""

    def __init__(self, parts, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.parts = list(parts)
        self.bijective = all(getattr(p, "bijective", False) for p in self.parts)

    def _call(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def _inverse(self, y):
        for part in reversed(self.parts):
            y = part._inverse(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        if not self.parts:
            return jt.zeros_like(x)
        total = None
        cur = x
        for part in self.parts:
            nxt = part(cur)
            term = part.log_abs_det_jacobian(cur, nxt)
            total = term if total is None else total + term
            cur = nxt
        return total


class TransformedDistribution:
    """Distribution obtained by applying one or more transforms to a base distribution."""

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms]
        else:
            self.transforms = list(transforms)
        self.batch_shape = getattr(base_distribution, "batch_shape", ())
        self.event_shape = getattr(base_distribution, "event_shape", ())

    def sample(self, sample_shape=()):
        x = self.base_dist.sample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def rsample(self, sample_shape=()):
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        y = value
        log_det = 0
        for transform in reversed(self.transforms):
            x = transform._inverse(y)
            log_det = log_det + transform.log_abs_det_jacobian(x, y)
            y = x
        return self.base_dist.log_prob(y) - log_det

    @property
    def mean(self):
        x = self.base_dist.mean
        for transform in self.transforms:
            x = transform(x)
        return x

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        var = self.base_dist.variance
        for transform in self.transforms:
            if isinstance(transform, AffineTransform):
                var = var * (transform.scale * transform.scale)
        return var


class StudentT:
    """Student's t-distribution over native Jittor tensor arithmetic."""

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self._frontend_type = _tensor_type()
        self.df = df if isinstance(df, jt.Var) else jt.array(df).float32()
        self.loc = loc if isinstance(loc, jt.Var) else jt.array(loc).float32()
        self.scale = scale if isinstance(scale, jt.Var) else jt.array(scale).float32()
        self.batch_shape = Gumbel._batch_shape(self.df, self.loc, self.scale)
        self.event_shape = ()

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale * self.scale * self.df / jt.clamp(self.df - 2.0, min_v=1e-6)

    @property
    def stddev(self):
        return self.variance.sqrt()

    @_sampling_scope
    def log_prob(self, value):
        import math
        v = value if isinstance(value, jt.Var) else jt.array(value).float32()
        y = (v - self.loc) / self.scale
        z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + jt.lgamma.apply(0.5 * self.df)
            - jt.lgamma.apply(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * jt.log1p((y * y) / self.df) - z

    @_sampling_scope
    def rsample(self, sample_shape=()):
        shape = Gumbel._sample_shape(sample_shape, self.batch_shape)
        normal = jt.randn(shape)
        return self.loc + self.scale * normal

    @_sampling_scope
    def sample(self, sample_shape=()):
        return self.rsample(sample_shape).stop_grad()


class NegativeBinomial:
    """Negative binomial distribution over native Jittor tensor arithmetic."""

    def __init__(self, total_count, probs=None, logits=None, validate_args=None):
        self._frontend_type = _tensor_type()
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        self.total_count = total_count if isinstance(total_count, jt.Var) else jt.array(total_count).float32()
        if logits is None:
            self.probs = probs if isinstance(probs, jt.Var) else jt.array(probs).float32()
            self.logits = jt.log(jt.clamp(self.probs, min_v=1e-6)) - jt.log(jt.clamp(1.0 - self.probs, min_v=1e-6))
        else:
            self.logits = logits if isinstance(logits, jt.Var) else jt.array(logits).float32()
            self.probs = jt.sigmoid(self.logits)
        self.batch_shape = Gumbel._batch_shape(self.total_count, self.logits)
        self.event_shape = ()

    @property
    def mean(self):
        return self.total_count * jt.exp(self.logits)

    @property
    def variance(self):
        return self.mean / jt.clamp(1.0 - self.probs, min_v=1e-6)

    @_sampling_scope
    def log_prob(self, value):
        v = value if isinstance(value, jt.Var) else jt.array(value).float32()
        log_unnormalized = v * jt.log(jt.clamp(self.probs, min_v=1e-6)) + self.total_count * jt.log(
            jt.clamp(1.0 - self.probs, min_v=1e-6)
        )
        log_norm = (
            jt.lgamma.apply(self.total_count + v)
            - jt.lgamma.apply(1.0 + v)
            - jt.lgamma.apply(self.total_count)
        )
        return log_unnormalized + log_norm

    @_sampling_scope
    def sample(self, sample_shape=()):
        shape = Gumbel._sample_shape(sample_shape, self.batch_shape)
        return jt.floor(jt.clamp(self.mean + jt.randn(shape) * self.variance.sqrt(), min_v=0.0)).stop_grad()
