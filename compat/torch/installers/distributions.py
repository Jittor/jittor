"""Distribution installation binds stable implementation owners and module paths."""
import types as _types_dist
from ..distribution_api import CONSTRAINTS, _Constraint, Gumbel, RelaxedBernoulli, RelaxedOneHotCategorical, unsupported_register_kl
from ..fidelity import Fidelity, register_api_bindings


class Wishart:
    """Wishart distribution (Bartlett decomposition), matching torch's public surface.

    peft imports ``torch.distributions.wishart.Wishart`` at module scope and calls
    ``_bartlett_sampling`` for MonteCLoRA; this implementation covers construction,
    sampling, mean and variance. ``log_prob`` uses the standard density with the
    multivariate gamma function when the backend provides ``lgamma``.
    """

    arg_constraints = {}

    def __init__(self, df, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        import jittor as _jt
        if scale_tril is None:
            if covariance_matrix is not None:
                scale_tril = _jt.linalg.cholesky(covariance_matrix)
            elif precision_matrix is not None:
                scale_tril = _jt.linalg.cholesky(_jt.linalg.inv(precision_matrix))
            else:
                raise ValueError("One of covariance_matrix, precision_matrix or scale_tril must be specified")
        self.df = float(df)
        self._unbroadcasted_scale_tril = scale_tril
        self._dim = scale_tril.shape[-1]
        self._batch_shape = tuple(scale_tril.shape[:-2])
        self._event_shape = (self._dim, self._dim)
        self._jt = _jt

    @property
    def mean(self):
        L = self._unbroadcasted_scale_tril
        return self.df * (L @ _transpose2(L))

    @property
    def variance(self):
        L = self._unbroadcasted_scale_tril
        V = L @ _transpose2(L)
        d = self._jt.diag_embed(self.df * self._jt.diagonal(V, dim1=-2, dim2=-1) ** 2) if hasattr(self._jt, "diag_embed") else None
        if d is None:
            raise NotImplementedError("Wishart.variance needs diag_embed")
        return 2 * self.df * V * V + d

    def _bartlett_sampling(self, sample_shape=()):
        jt = self._jt
        p = self._dim
        df = self.df
        shape = tuple(sample_shape) + self._batch_shape
        idx = jt.arange(p)
        lower = (idx.reshape(p, 1) > idx.reshape(1, p)).float()
        diag_mask = (idx.reshape(p, 1) == idx.reshape(1, p)).float()
        diag = []
        for i in range(p):
            concentration = jt.array([max(df - i, 1e-6) / 2.0], dtype="float32")
            rate = jt.array([1.0], dtype="float32")
            chi2 = 2.0 * jt.distributions.Gamma(concentration, rate).sample(shape + (1,))
            diag.append(chi2.sqrt())
        d = jt.concat(diag, dim=-1)
        normal = jt.randn(shape + (p, p))
        B = normal * lower + d.unsqueeze(-1) * diag_mask
        L = self._unbroadcasted_scale_tril
        X = L @ B
        return X @ _transpose2(X)

    def rsample(self, sample_shape=()):
        return self._bartlett_sampling(sample_shape)

    def sample(self, sample_shape=()):
        return self._bartlett_sampling(sample_shape)

    def log_prob(self, value):
        jt = self._jt
        if not hasattr(jt, "lgamma"):
            raise NotImplementedError("Wishart.log_prob needs lgamma in the backend")
        L = self._unbroadcasted_scale_tril
        p = self._dim
        df = self.df
        sign, logdet = jt.linalg.slogdet(value)
        V = L @ _transpose2(L)
        Vinv = jt.linalg.inv(V)
        trace = jt.trace(Vinv @ value)
        log_multigamma = (df * p / 2.0) * jt.log(2.0) + (p * (p - 1) / 4.0) * jt.log(jt.array(3.141592653589793)) \
            + sum(jt.lgamma(jt.array((df + 1 - j) / 2.0)) for j in range(1, p + 1))
        return 0.5 * ((df - p - 1) * logdet - trace) - log_multigamma


def _transpose2(x):
    return x.transpose(-1, -2)


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
    if not hasattr(_dist, "Wishart"):
        _dist.Wishart = Wishart
    _wishart = ctx.registry.ensure("torch.distributions.wishart")
    _wishart.Wishart = _dist.Wishart
    _modules["torch.distributions.wishart"] = _wishart
    _dist.wishart = _wishart
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
