"""Torch-grade SHAPE semantics for ``jittor.distributions`` (TASK #12).

torch's ``Distribution.sample(sample_shape)`` returns

    sample_shape + batch_shape + event_shape

with the *batch* dims (the broadcast of the parameters) preserved and ``sample_shape``
PREPENDED. This module locks that contract for every distribution, for

    {scalar params, batched params} x {sample_shape = (), (n,), (n, m)}

on BOTH CPU and CUDA, plus ``.log_prob`` / ``.rsample`` shapes and a couple of value /
gradient sanity checks (sampled mean ~ loc for Normal; rsample is differentiable;
sample() is detached).

The expected shapes here are the GROUND TRUTH read off REAL torch 2.12
(torch.distributions, run on the cscg104 oracle) for the SAME parameters -- see the
``_T`` tables below, each annotated with the torch shape it mirrors.

One representational caveat, asserted explicitly rather than hidden: jittor has NO 0-d
(scalar) Var -- ``jt.zeros(())`` / ``jt.randn(())`` / ``reshape(())`` are rejected at the
C++ level -- so a *scalar* parameter materializes as shape ``(1,)``. Hence wherever torch
would return ``()`` (scalar params, ``sample_shape=()``) jittor returns ``(1,)``; this is
encoded by ``_c()`` (collapse ``() -> (1,)``). For every batched-parameter case, and for
scalar params with a non-empty ``sample_shape``, the shapes match torch EXACTLY.

Run:  python -m jittor.test.test_torch_compat_distrib_shape
      python -m pytest python/jittor/test/test_torch_compat_distrib_shape.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
from jittor import distributions as D

# CPU always; CUDA when the build has it (NPU/ACL also reports has_cuda).
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _c(shape):
    """Collapse torch's 0-d () to jittor's (1,) (jittor has no 0-d Var)."""
    return (1,) if tuple(shape) == () else tuple(shape)


# Each entry: name -> (factory, [torch shape for sample(()), sample((4,)), sample((4,2)),
#                                 torch shape for log_prob(sample((4,)))]).
# The torch shapes are the REAL torch 2.12 ground truth for these exact params.
SAMPLE_SHAPES = [(), (4,), (4, 2)]


def _cases():
    A = jt.array
    return {
        # ---- Normal: event_shape () ----  torch sample(()) , (4,) , (4,2) ; log_prob
        # scalar params: log_prob(sample((4,))) is (4,) in torch (the VALUE is (4,),
        # broadcast against the scalar params) -- NOT ().
        "Normal scalar":       (lambda: D.Normal(0.0, 1.0),
                                [(), (4,), (4, 2)], (4,)),                    # torch (),(4,),(4,2)|(4,)
        "Normal batched(3)":   (lambda: D.Normal(A([1., 2., 3.]), A([.1, .2, .3])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),           # torch (3,),(4,3),(4,2,3)|(4,3)
        "Normal batched(2,3)": (lambda: D.Normal(jt.zeros(2, 3), jt.ones(2, 3)),
                                [(2, 3), (4, 2, 3), (4, 2, 2, 3)], (4, 2, 3)),

        # ---- Categorical: batch=probs.shape[:-1], event () ----
        "Categorical (3)":     (lambda: D.Categorical(probs=A([.2, .3, .5])),
                                [(), (4,), (4, 2)], (4,)),                    # torch (),(4,),(4,2)|(4,)
        "Categorical (2,3)":   (lambda: D.Categorical(probs=A([[.2, .3, .5], [.1, .1, .8]])),
                                [(2,), (4, 2), (4, 2, 2)], (4, 2)),

        # ---- Bernoulli: event () ----
        "Bernoulli scalar":    (lambda: D.Bernoulli(probs=0.3),
                                [(), (4,), (4, 2)], (4,)),
        "Bernoulli batched(3)":(lambda: D.Bernoulli(probs=A([.2, .5, .8])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- Uniform: event () ----
        "Uniform scalar":      (lambda: D.Uniform(0.0, 1.0),
                                [(), (4,), (4, 2)], (4,)),
        "Uniform batched(3)":  (lambda: D.Uniform(A([0., 1., 2.]), A([1., 2., 3.])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- Exponential: event () ----
        "Exponential scalar":  (lambda: D.Exponential(rate=1.0),
                                [(), (4,), (4, 2)], (4,)),
        "Exponential batched(3)":(lambda: D.Exponential(rate=A([1., 2., 3.])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- Poisson: event () ----
        "Poisson scalar":      (lambda: D.Poisson(rate=3.0),
                                [(), (4,), (4, 2)], (4,)),
        "Poisson batched(3)":  (lambda: D.Poisson(rate=A([1., 2., 3.])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- Gamma: event () ----
        "Gamma scalar":        (lambda: D.Gamma(2.0, 1.0),
                                [(), (4,), (4, 2)], (4,)),
        "Gamma batched(3)":    (lambda: D.Gamma(A([1., 2., 3.]), A([1., 1., 1.])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- Beta: event () ----
        "Beta scalar":         (lambda: D.Beta(2.0, 3.0),
                                [(), (4,), (4, 2)], (4,)),
        "Beta batched(3)":     (lambda: D.Beta(A([1., 2., 3.]), A([1., 1., 1.])),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- Dirichlet: event (k,) -> log_prob drops it ----
        "Dirichlet (3)":       (lambda: D.Dirichlet(A([1., 1., 1.])),
                                [(3,), (4, 3), (4, 2, 3)], (4,)),             # torch (3,),(4,3),(4,2,3)|(4,)
        "Dirichlet (2,3)":     (lambda: D.Dirichlet(A([[1., 1., 1.], [2., 2., 2.]])),
                                [(2, 3), (4, 2, 3), (4, 2, 2, 3)], (4, 2)),

        # ---- LogNormal: event () ----
        "LogNormal scalar":    (lambda: D.LogNormal(0.0, 1.0),
                                [(), (4,), (4, 2)], (4,)),
        "LogNormal batched(3)":(lambda: D.LogNormal(jt.zeros(3), jt.ones(3)),
                                [(3,), (4, 3), (4, 2, 3)], (4, 3)),

        # ---- OneHotCategorical: event (k,) -> log_prob drops it ----
        "OneHotCategorical (3)":(lambda: D.OneHotCategorical(probs=A([.2, .3, .5])),
                                [(3,), (4, 3), (4, 2, 3)], (4,)),
        "OneHotCategorical (2,3)":(lambda: D.OneHotCategorical(probs=A([[.2, .3, .5], [.1, .1, .8]])),
                                [(2, 3), (4, 2, 3), (4, 2, 2, 3)], (4, 2)),

        # ---- MultivariateNormal: event (k,) -> log_prob drops it ----
        "MVN (3)":             (lambda: D.MultivariateNormal(jt.zeros(3), jt.init.eye(3)),
                                [(3,), (4, 3), (4, 2, 3)], (4,)),
        "MVN (2,3)":           (lambda: D.MultivariateNormal(jt.zeros(2, 3), jt.init.eye(3)),
                                [(2, 3), (4, 2, 3), (4, 2, 2, 3)], (4, 2)),

        # ---- Independent(Normal(2,3),1): batch (2,), event (3,) ----
        "Independent":         (lambda: D.Independent(D.Normal(jt.zeros(2, 3), jt.ones(2, 3)), 1),
                                [(2, 3), (4, 2, 3), (4, 2, 2, 3)], (4, 2)),

        # ---- Geometric: event () (jittor asserts scalar p in (0,1)) ----
        "Geometric scalar":    (lambda: D.Geometric(p=0.3),
                                [(), (4,), (4, 2)], (4,)),
    }


class TestSampleShape(unittest.TestCase):
    """sample()/rsample()/log_prob() shapes == torch's sample_shape+batch+event."""

    def _check_one(self, name, mk, sample_specs, lp_spec, dev):
        for ss, exp in zip(SAMPLE_SHAPES, sample_specs):
            got = tuple(mk().sample(ss).shape)
            self.assertEqual(got, _c(exp),
                f"{name}.sample({ss}) [{dev}]: got {got}, torch->{exp} (jittor collapse {_c(exp)})")
            # rsample (where the distribution exposes it) must share sample's shape.
            if hasattr(mk(), "rsample"):
                gr = tuple(mk().rsample(ss).shape)
                self.assertEqual(gr, _c(exp),
                    f"{name}.rsample({ss}) [{dev}]: got {gr}, torch->{exp}")
        # log_prob on a sample drawn at sample_shape=(4,)
        s = mk().sample((4,))
        glp = tuple(mk().log_prob(s).shape)
        self.assertEqual(glp, _c(lp_spec),
            f"{name}.log_prob(sample((4,))) [{dev}]: got {glp}, torch->{lp_spec}")

    def test_all(self):
        cases = _cases()
        def body(dev):
            for name, (mk, sample_specs, lp_spec) in cases.items():
                self._check_one(name, mk, sample_specs, lp_spec, dev)
        both_devices(body)


class TestValueSanity(unittest.TestCase):
    """Shapes are right AND the numbers mean what they should."""

    def test_normal_mean_matches_loc(self):
        loc = np.array([1., 2., 3.], dtype="float32")
        def body(dev):
            d = D.Normal(jt.array(loc), jt.array([0.5, 0.5, 0.5]))
            s = d.sample((20000,))
            self.assertEqual(tuple(s.shape), (20000, 3), f"shape {dev}")
            np.testing.assert_allclose(s.mean(0).numpy(), loc, atol=0.05,
                                       err_msg=f"Normal sample mean ~ loc {dev}")
        both_devices(body)

    def test_beta_mean(self):
        a = np.array([2., 5.], dtype="float32"); b = np.array([2., 1.], dtype="float32")
        def body(dev):
            s = D.Beta(jt.array(a), jt.array(b)).sample((20000,))
            self.assertEqual(tuple(s.shape), (20000, 2), f"shape {dev}")
            np.testing.assert_allclose(s.mean(0).numpy(), a / (a + b), atol=0.03,
                                       err_msg=f"Beta sample mean {dev}")
        both_devices(body)

    def test_dirichlet_sums_to_one(self):
        def body(dev):
            s = D.Dirichlet(jt.array([1., 2., 3.])).sample((5000,))
            self.assertEqual(tuple(s.shape), (5000, 3), f"shape {dev}")
            np.testing.assert_allclose(s.sum(-1).numpy(), np.ones(5000), atol=1e-4,
                                       err_msg=f"Dirichlet rows sum to 1 {dev}")
        both_devices(body)

    def test_onehot_is_one_hot(self):
        def body(dev):
            s = D.OneHotCategorical(probs=jt.array([.2, .3, .5])).sample((1000,))
            self.assertEqual(tuple(s.shape), (1000, 3), f"shape {dev}")
            np.testing.assert_allclose(s.sum(-1).numpy(), np.ones(1000), atol=1e-6,
                                       err_msg=f"one-hot rows sum to 1 {dev}")
        both_devices(body)


class TestGradAndDetach(unittest.TestCase):
    """torch parity: rsample is differentiable to the params; sample() is detached."""

    def test_normal_rsample_is_differentiable(self):
        def body(dev):
            mu = jt.array([0., 0., 0.])
            d = D.Normal(mu, jt.array([1., 1., 1.]))
            r = d.rsample((100,))
            self.assertEqual(tuple(r.shape), (100, 3), f"rsample shape {dev}")
            g = jt.grad(r.mean(), [mu])[0]
            # d(mean over 100x3 / count)/dmu_i = 1/3 per parameter
            np.testing.assert_allclose(g.numpy(), np.full(3, 1.0 / 3), atol=1e-4,
                                       err_msg=f"rsample grad to mu {dev}")
        both_devices(body)

    def test_sample_is_detached(self):
        def body(dev):
            d = D.Normal(jt.array([0., 0.]), jt.array([1., 1.]))
            self.assertTrue(d.sample((5,)).is_stop_grad(),
                            f"Normal.sample() must be detached {dev}")
            self.assertFalse(d.rsample((5,)).is_stop_grad(),
                             f"Normal.rsample() must keep grad {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main()
