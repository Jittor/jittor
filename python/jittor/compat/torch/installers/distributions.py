"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt
from jittor import nn
from ...diagnostics import EXPECTED, swallowed


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    # ---- torch.distributions package layout ----
    # jittor.distributions already implements the common distribution classes;
    # expose the PyTorch package/submodule names that transformers imports.
    import types as _types_dist
    def _install_distribution_surface():
        import jittor.distributions as _dist
        _dist.__path__ = getattr(_dist, "__path__", [])
        if not hasattr(_dist, "constraints"):
            _constraints = _types_dist.ModuleType("torch.distributions.constraints")
            class _Constraint:
                def __init__(self, *a, **k): pass
                def check(self, x):
                    try:
                        return jt.ones_like(x).bool()
                    except EXPECTED as exc:
                        swallowed("torch/installers/distributions.py check: return jt.ones_like(x).bool()", exc)
                        return True
            for _cn in ("positive", "real", "nonnegative", "nonnegative_integer",
                        "positive_integer", "unit_interval", "simplex",
                        "lower_cholesky", "positive_definite", "boolean",
                        "real_vector", "dependent", "independent"):
                setattr(_constraints, _cn, _Constraint())
            _constraints.Constraint = _Constraint
            _dist.constraints = _constraints
        _modules["torch.distributions"] = _dist
        _modules["torch.distributions.constraints"] = _dist.constraints
        g.distributions = _dist
        _dist_utils = _types_dist.ModuleType("torch.distributions.utils")
        _dist_utils.broadcast_all = getattr(_dist, "broadcast_all")
        _modules["torch.distributions.utils"] = _dist_utils
        _dist.utils = _dist_utils
        for _cls_name, _mod_suffix in (
            ("Distribution", "distribution"),
            ("Bernoulli", "bernoulli"),
            ("Categorical", "categorical"),
            ("OneHotCategorical", "one_hot_categorical"),
            ("Normal", "normal"),
            ("Uniform", "uniform"),
            ("RelaxedBernoulli", "relaxed_bernoulli"),
            ("LogitRelaxedBernoulli", "relaxed_bernoulli"),
            ("RelaxedOneHotCategorical", "relaxed_categorical"),
            ("Beta", "beta"),
            ("Gamma", "gamma"),
            ("Poisson", "poisson"),
            ("Dirichlet", "dirichlet"),
            ("LogNormal", "log_normal"),
            ("LogisticNormal", "logistic_normal"),
            ("MultivariateNormal", "multivariate_normal"),
        ):
            if hasattr(_dist, _cls_name):
                _sub = _types_dist.ModuleType("torch.distributions." + _mod_suffix)
                setattr(_sub, _cls_name, getattr(_dist, _cls_name))
                _modules["torch.distributions." + _mod_suffix] = _sub
                setattr(_dist, _mod_suffix, _sub)
        if hasattr(_dist, "RelaxedBernoulli") or hasattr(_dist, "LogitRelaxedBernoulli"):
            _relaxed_bernoulli = _types_dist.ModuleType("torch.distributions.relaxed_bernoulli")
            if hasattr(_dist, "RelaxedBernoulli"):
                _relaxed_bernoulli.RelaxedBernoulli = _dist.RelaxedBernoulli
            if hasattr(_dist, "LogitRelaxedBernoulli"):
                _relaxed_bernoulli.LogitRelaxedBernoulli = _dist.LogitRelaxedBernoulli
            _modules["torch.distributions.relaxed_bernoulli"] = _relaxed_bernoulli
            _dist.relaxed_bernoulli = _relaxed_bernoulli
        if hasattr(_dist, "RelaxedOneHotCategorical"):
            _relaxed_categorical = _types_dist.ModuleType("torch.distributions.relaxed_categorical")
            _relaxed_categorical.RelaxedOneHotCategorical = _dist.RelaxedOneHotCategorical
            _modules["torch.distributions.relaxed_categorical"] = _relaxed_categorical
            _dist.relaxed_categorical = _relaxed_categorical
        if hasattr(_dist, "kl_divergence"):
            _kl = _types_dist.ModuleType("torch.distributions.kl")
            _kl.kl_divergence = _dist.kl_divergence
            _kl.register_kl = getattr(_dist, "register_kl", lambda *a, **k: (lambda f: f))
            _modules["torch.distributions.kl"] = _kl
            _dist.kl = _kl

        class Gumbel:
            def __init__(self, loc, scale, validate_args=None):
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
            def rsample(self, sample_shape=None):
                u = jt.random(self._sample_shape(sample_shape, self.batch_shape))
                eps = 1e-6
                u = jt.clamp(u, eps, 1.0 - eps)
                loc = self.loc if isinstance(self.loc, jt.Var) else jt.array(self.loc)
                scale = self.scale if isinstance(self.scale, jt.Var) else jt.array(self.scale)
                return loc - scale * jt.log(-jt.log(u))
            def sample(self, sample_shape=None):
                return self.rsample(sample_shape).stop_grad()

        class RelaxedBernoulli:
            def __init__(self, temperature, probs=None, logits=None, validate_args=None):
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
            def sample(self, sample_shape=None):
                return self.rsample(sample_shape).stop_grad()

        class RelaxedOneHotCategorical:
            def __init__(self, temperature, probs=None, logits=None, validate_args=None):
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
            def sample(self, sample_shape=None):
                return self.rsample(sample_shape).stop_grad()

        _dist.Gumbel = getattr(_dist, "Gumbel", Gumbel)
        _dist.RelaxedBernoulli = getattr(_dist, "RelaxedBernoulli", RelaxedBernoulli)
        _dist.RelaxedOneHotCategorical = getattr(_dist, "RelaxedOneHotCategorical", RelaxedOneHotCategorical)
        for _cls_name, _mod_suffix in (
            ("Gumbel", "gumbel"),
            ("RelaxedBernoulli", "relaxed_bernoulli"),
            ("RelaxedOneHotCategorical", "relaxed_categorical"),
        ):
            _sub = _types_dist.ModuleType("torch.distributions." + _mod_suffix)
            setattr(_sub, _cls_name, getattr(_dist, _cls_name))
            _modules[_sub.__name__] = _sub
            setattr(_dist, _mod_suffix, _sub)
    _install_distribution_surface()


def install_parity(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    def module(name):
        return registry.ensure(name)
    distributions = getattr(g, "distributions", None)
    if distributions is not None and hasattr(distributions, "Geometric"):
        geometric = module("torch.distributions.geometric")
        geometric.Geometric = distributions.Geometric
        distributions.geometric = geometric
