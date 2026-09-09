"""Distribution installation binds stable implementation owners and module paths."""
import types as _types_dist
from ..distribution_api import CONSTRAINTS, _Constraint, Gumbel, RelaxedBernoulli, RelaxedOneHotCategorical, unsupported_register_kl
from ..fidelity import Fidelity, register_api_bindings


def install(ctx):
    _install_distribution_surface(ctx)


def _install_distribution_surface(ctx):
    g = ctx.jittor_module
    _modules = ctx.registry.module_map
    import jittor.distributions as _dist
    from ..distribution_frontend import make_distribution_frontend
    existing = vars(g).get("distributions")
    if (existing is None or
            getattr(existing, "_native_distribution_module", None) is not _dist):
        existing = make_distribution_frontend(_dist, g)
    _dist = existing
    _dist.__path__ = getattr(_dist, "__path__", [])
    if not hasattr(_dist, "constraints"):
        _constraints = _types_dist.ModuleType("torch.distributions.constraints")
        for _cn, _constraint in CONSTRAINTS.items():
            setattr(_constraints, _cn, _constraint)
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
            _sub = ctx.registry.ensure("torch.distributions." + _mod_suffix)
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
        _kl.register_kl = getattr(_dist, "register_kl", unsupported_register_kl)
        _modules["torch.distributions.kl"] = _kl
        _dist.kl = _kl
    _dist.Gumbel = getattr(_dist, "Gumbel", Gumbel)
    _dist.RelaxedBernoulli = getattr(_dist, "RelaxedBernoulli", RelaxedBernoulli)
    _dist.RelaxedOneHotCategorical = getattr(_dist, "RelaxedOneHotCategorical", RelaxedOneHotCategorical)
    for _cls_name, _mod_suffix in (
        ("Gumbel", "gumbel"),
        ("RelaxedBernoulli", "relaxed_bernoulli"),
        ("RelaxedOneHotCategorical", "relaxed_categorical"),
    ):
        _sub = ctx.registry.ensure("torch.distributions." + _mod_suffix)
        setattr(_sub, _cls_name, getattr(_dist, _cls_name))
        _modules[_sub.__name__] = _sub
        setattr(_dist, _mod_suffix, _sub)
    register_api_bindings(_dist, "torch.distributions",
        tuple(name for name in ("Gumbel", "RelaxedBernoulli", "RelaxedOneHotCategorical",
                                "Normal", "Uniform", "Distribution", "Bernoulli", "Beta",
                                "Categorical", "Dirichlet", "Exponential", "Gamma", "Geometric",
                                "Independent", "LogNormal", "LogisticNormal", "MultivariateNormal",
                                "OneHotCategorical", "Poisson") if hasattr(_dist, name)),
        Fidelity.APPROXIMATE,
        "Native distributions retain their backend limits; sampling fallbacks clamp random values and have limited validation")
    register_api_bindings(_dist.constraints, "torch.distributions.constraints", ("Constraint",),
        Fidelity.APPROXIMATE, "Legacy permissive constraint objects do not validate mathematical support")
    for class_name in (
        "Distribution", "Normal", "Uniform", "Bernoulli", "Beta", "Categorical",
        "Dirichlet", "Exponential", "Gamma", "Geometric", "Independent", "LogNormal",
        "LogisticNormal", "MultivariateNormal", "OneHotCategorical", "Poisson",
        "Gumbel", "RelaxedBernoulli", "RelaxedOneHotCategorical",
    ):
        distribution_type = getattr(_dist, class_name, None)
        if distribution_type is not None:
            register_api_bindings(distribution_type, "torch.distributions." + class_name,
                ("__init__", "sample", "rsample", "log_prob", "prob", "entropy", "cdf",
                 "icdf", "mean", "variance", "stddev", "loc", "scale", "expand",
                 "enumerate_support"), Fidelity.APPROXIMATE,
                "Module-owned adapter applies frontend dtype/sampling policy to native "
                "distribution mathematics; native validation and backend limitations remain")
    register_api_bindings(_dist, "torch.distributions",
        ("kl_divergence", "broadcast_all", "simple_presum"), Fidelity.APPROXIMATE,
        "Native distribution mathematics under the active Tensor frontend scope; "
        "native argument and distribution-pair restrictions remain")
    register_api_bindings(_dist_utils, "torch.distributions.utils", ("broadcast_all",),
        Fidelity.APPROXIMATE, "Shared native broadcasting with frontend Tensor results")
    if hasattr(_dist, "kl"):
        register_api_bindings(_dist.kl, "torch.distributions.kl", ("kl_divergence",),
            Fidelity.APPROXIMATE, "Shared native KL implementation and supported distribution pairs")


def install_parity(ctx):
    distributions = getattr(ctx.jittor_module, "distributions", None)
    if distributions is not None and hasattr(distributions, "Geometric"):
        geometric = ctx.registry.ensure("torch.distributions.geometric")
        geometric.Geometric = distributions.Geometric
        distributions.geometric = geometric
