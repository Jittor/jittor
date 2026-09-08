"""Public distribution objects survive the move to family-owned modules."""

import importlib
import pickle

import jittor as jt
import jittor.distributions as distributions


def test_distribution_family_exports_and_pickle_globals():
    families = {
        "base": ("Distribution", "Independent"),
        "discrete": ("OneHotCategorical", "Categorical", "Geometric", "Bernoulli", "Poisson"),
        "continuous": ("Normal", "Uniform", "GammaDistribution", "Exponential", "Beta", "Gamma", "LogNormal"),
        "relaxed": ("LogitRelaxedBernoulli", "RelaxedBernoulli", "ExpRelaxedCategorical", "RelaxedOneHotCategorical"),
        "multivariate": ("Dirichlet", "LogisticNormal", "MultivariateNormal"),
        "divergence": ("kl_divergence",),
        "_utils": ("broadcast_all", "simple_presum"),
    }
    assert jt.distributions is distributions
    for family, names in families.items():
        owner = importlib.import_module("jittor.distributions." + family)
        for name in names:
            public = getattr(distributions, name)
            assert public is getattr(owner, name)
            assert public.__module__ == owner.__name__
            assert pickle.loads(pickle.dumps(public)) is public
            legacy_global = ("cjittor.distributions\n" + name + "\n.").encode("ascii")
            assert pickle.loads(legacy_global) is public
            assert name in distributions.__all__


def test_distribution_instance_pickle_and_constraint_identity():
    original = distributions.Normal(2.0, 3.0)
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is distributions.Normal
    assert (restored.mu, restored.sigma, restored.batch_shape) == (2.0, 3.0, ())
    owner = importlib.import_module("jittor.distributions._constraints")
    assert distributions.constraints is owner.constraints
    assert distributions.constraints.Constraint is owner._Constraint
    assert distributions.constraints.interval(1, 3).check(2)
    assert not distributions.constraints.interval(1, 3).check(4)


def test_distribution_broadcast_helper_keeps_shape_contract():
    helpers = importlib.import_module("jittor.distributions._utils")
    assert distributions._full_shape is helpers._full_shape
    assert distributions._broadcast_two is helpers._broadcast_two
    assert helpers._full_shape((5,), (2, 3), (4,)) == (5, 2, 3, 4)
    assert helpers._broadcast_two((2, 1), (3,)) == (2, 3)
