"""Torch-grade ``jittor.distributions`` semantics tests (sibling of test_torch_compat_ops.py).

Part of the torch-grade test-suite rewrite. Each distribution is checked against an
INDEPENDENT, scipy-FREE analytic reference (the closed-form log_prob / entropy / mean /
variance formulas), on BOTH CPU and CUDA when the build has it. This locks the
torch.distributions-facing *semantics* rather than jittor self-consistency.

Covers Normal / Bernoulli / Categorical / Uniform / Exponential:
  * log_prob / entropy vs analytic formulas,
  * .sample(shape) shape + value-range / dtype sanity,
  * .rsample() reparameterization is differentiable (grad flows to the parameters),
  * Categorical(logits=) == Categorical(probs=softmax(logits)) (the softmax fix).

API quirks pinned here (verified against jittor/distributions.py):
  * Normal is constructed POSITIONALLY as Normal(mu, sigma) -- NOT loc/scale.
  * entropy() is a METHOD on all of these; Normal/Categorical/Uniform/Exponential have
    NO mean/variance attribute (so mean/variance are checked via the formulas / samples).
  * Categorical: logits -> probs via SOFTMAX; self.logits is stored as log_softmax;
    .sample() takes a shape TUPLE and returns INTEGER class indices.
  * Bernoulli: logits -> probs via SIGMOID (contrast with Categorical); sample is float 0/1.
  * jittor has no 0-d scalar Var; scalar results are compared via .item().

Known-broken behaviour is @unittest.skip-ped with a ``BUG:``/``DIVERGENCE:`` reason
rather than asserted around (see the suspected-bug list returned with this rewrite).

Run:  python -m jittor.test.test_torch_compat_distributions
      python -m pytest python/jittor/test/test_torch_compat_distributions.py
"""
import math
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
from jittor import distributions as D

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def scalar(self, v):
        """Pull a python float out of a jittor Var / numpy scalar (no 0-d Var in jittor)."""
        a = np.asarray(v.numpy() if hasattr(v, "numpy") else v)
        return float(a.reshape(-1)[0]) if a.size == 1 else a


_LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------------- Normal

class TestNormal(Base):
    def setUp(self):
        self.mu = np.array([0.5, -1.0, 2.0], dtype="float32")
        self.sigma = np.array([2.0, 0.5, 1.5], dtype="float32")

    def _dist(self):
        return D.Normal(jt.array(self.mu), jt.array(self.sigma))

    def test_log_prob(self):
        x = np.array([1.3, -0.5, 2.2], dtype="float32")
        ref = -0.5 * ((x - self.mu) / self.sigma) ** 2 \
            - np.log(self.sigma) - 0.5 * _LOG2PI
        def body(dev):
            self.ac(self._dist().log_prob(jt.array(x)).numpy(), ref, atol=1e-5,
                    msg=f"Normal.log_prob {dev}")
        both_devices(body)

    def test_entropy(self):
        ref = 0.5 + 0.5 * _LOG2PI + np.log(self.sigma)
        def body(dev):
            self.ac(self._dist().entropy().numpy(), ref, atol=1e-5,
                    msg=f"Normal.entropy {dev}")
        both_devices(body)

    def test_sample_shape(self):
        def body(dev):
            # batched params: sample() with no shape == batch_shape (matches torch).
            self.assertEqual(tuple(self._dist().sample().shape), (3,),
                             f"Normal.sample() shape {dev}")
            # scalar params: sample(shape) returns exactly `shape`.
            d0 = D.Normal(jt.array(0.5), jt.array(2.0))
            self.assertEqual(tuple(d0.sample((100,)).shape), (100,),
                             f"Normal.sample((100,)) shape {dev}")
            self.assertEqual(tuple(d0.sample((10, 4)).shape), (10, 4),
                             f"Normal.sample((10,4)) shape {dev}")
        both_devices(body)

    @unittest.skip("DIVERGENCE: with BATCHED params (shape (3,)), Normal.sample((n,)) "
                   "raises a broadcast error instead of returning torch's (n,)+batch=(n,3)"
                   ". jittor passes sample_shape as the literal output shape (jt.normal "
                   "size=sample_shape) rather than prepending it to batch_shape; you must "
                   "pass the full (n,3). distributions.py:100.")
    def test_sample_prepends_to_batch_shape(self):
        def body(dev):
            self.assertEqual(tuple(self._dist().sample((100,)).shape), (100, 3),
                             f"Normal.sample((100,)) on batched params {dev}")
        both_devices(body)

    def test_sample_statistics(self):
        # mean/std of a large i.i.d. sample should match (mu, sigma); a coarse check
        # since these are no closed-form .mean/.variance attributes on Normal.
        mu, sigma = 1.5, 2.0
        def body(dev):
            d = D.Normal(jt.array(mu), jt.array(sigma))
            s = d.sample((50000,)).numpy()
            self.assertAlmostEqual(float(s.mean()), mu, delta=0.1, msg=f"Normal sample mean {dev}")
            self.assertAlmostEqual(float(s.std()), sigma, delta=0.1, msg=f"Normal sample std {dev}")
        both_devices(body)

    def test_rsample_is_differentiable(self):
        # reparameterized sample: gradient must flow to BOTH mu and sigma.
        def body(dev):
            mu_v = jt.array([0.5]); sigma_v = jt.array([2.0])
            d = D.Normal(mu_v, sigma_v)
            s = d.rsample((64,))
            self.assertEqual(tuple(s.shape), (64,), f"Normal.rsample shape {dev}")
            gmu, gsig = jt.grad(s.sum(), [mu_v, sigma_v])
            # d/dmu sum(mu + sigma*eps) = N (=64); d/dsigma = sum(eps) (random, ~nonzero)
            self.assertAlmostEqual(float(gmu.numpy().sum()), 64.0, delta=1e-3,
                                   msg=f"Normal.rsample grad mu {dev}")
            self.assertNotEqual(float(gsig.numpy().sum()), 0.0,
                                f"Normal.rsample grad sigma must be nonzero {dev}")
        both_devices(body)


# ----------------------------------------------------------------------- Bernoulli

class TestBernoulli(Base):
    def setUp(self):
        self.p = np.array([0.3, 0.7, 0.5], dtype="float32")

    def _dist(self):
        return D.Bernoulli(probs=jt.array(self.p))

    def test_log_prob(self):
        p = self.p
        def body(dev):
            d = self._dist()
            self.ac(d.log_prob(jt.ones((3,))).numpy(), np.log(p), atol=1e-5,
                    msg=f"Bernoulli.log_prob(1) {dev}")
            self.ac(d.log_prob(jt.zeros((3,))).numpy(), np.log(1 - p), atol=1e-5,
                    msg=f"Bernoulli.log_prob(0) {dev}")
        both_devices(body)

    def test_entropy(self):
        p = self.p
        ref = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        def body(dev):
            self.ac(self._dist().entropy().numpy(), ref, atol=1e-5,
                    msg=f"Bernoulli.entropy {dev}")
        both_devices(body)

    def test_sample_shape_and_values(self):
        def body(dev):
            # batched probs (3,): sample() (no shape) == batch_shape; for a leading
            # sample dim you must pass the full (n,3) (see broadcast divergence below).
            d = self._dist()
            self.assertEqual(tuple(d.sample().shape), (3,), f"Bernoulli.sample() shape {dev}")
            s = d.sample((1000, 3))
            self.assertEqual(tuple(s.shape), (1000, 3), f"Bernoulli.sample((1000,3)) shape {dev}")
            vals = set(np.unique(s.numpy()).tolist())
            self.assertTrue(vals <= {0.0, 1.0}, f"Bernoulli sample values {vals} {dev}")
        both_devices(body)

    def test_sample_mean(self):
        def body(dev):
            d = D.Bernoulli(probs=jt.array([0.3]))
            s = d.sample((20000,)).numpy()
            self.assertAlmostEqual(float(s.mean()), 0.3, delta=0.02,
                                   msg=f"Bernoulli sample mean ~ p {dev}")
        both_devices(body)

    def test_logits_use_sigmoid(self):
        # Bernoulli maps logits -> probs via SIGMOID (NOT softmax).
        logits = np.array([0.8, -1.2, 0.0], dtype="float32")
        ref_p = 1.0 / (1.0 + np.exp(-logits))
        def body(dev):
            d = D.Bernoulli(logits=jt.array(logits))
            self.ac(d.probs.numpy(), ref_p, atol=1e-5, msg=f"Bernoulli logits->sigmoid {dev}")
            # log_prob(1) == log(sigmoid(logit))
            self.ac(d.log_prob(jt.ones((3,))).numpy(), np.log(ref_p), atol=1e-5,
                    msg=f"Bernoulli logits log_prob {dev}")
        both_devices(body)


# --------------------------------------------------------------------- Categorical

class TestCategorical(Base):
    def setUp(self):
        self.probs = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

    def test_log_prob(self):
        p = self.probs
        def body(dev):
            d = D.Categorical(probs=jt.array(p))
            for k in range(4):
                lp = d.log_prob(jt.array([k])).numpy()
                self.ac(lp, np.array([np.log(p[k])]), atol=1e-5,
                        msg=f"Categorical.log_prob({k}) {dev}")
        both_devices(body)

    def test_entropy(self):
        p = self.probs
        ref = -(p * np.log(p)).sum()
        def body(dev):
            d = D.Categorical(probs=jt.array(p))
            self.assertAlmostEqual(self.scalar(d.entropy()), float(ref), delta=1e-5,
                                   msg=f"Categorical.entropy {dev}")
        both_devices(body)

    def test_probs_are_normalized(self):
        # unnormalized probs must be renormalized.
        raw = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        def body(dev):
            d = D.Categorical(probs=jt.array(raw))
            self.ac(d.probs.numpy(), raw / raw.sum(), atol=1e-6,
                    msg=f"Categorical probs normalized {dev}")
        both_devices(body)

    def test_sample_shape_and_dtype(self):
        p = self.probs
        def body(dev):
            d = D.Categorical(probs=jt.array(p))
            s = d.sample((5,))
            self.assertEqual(tuple(s.shape), (5,), f"Categorical.sample shape {dev}")
            arr = s.numpy()
            self.assertTrue(np.issubdtype(arr.dtype, np.integer),
                            f"Categorical sample must be integer indices, got {arr.dtype} {dev}")
            self.assertTrue(arr.min() >= 0 and arr.max() <= 3,
                            f"Categorical sample index range {dev}")
        both_devices(body)

    def test_sample_frequencies(self):
        p = self.probs
        def body(dev):
            d = D.Categorical(probs=jt.array(p))
            s = d.sample((40000,)).numpy()
            freq = np.bincount(s, minlength=4) / s.size
            self.ac(freq, p, atol=0.02, msg=f"Categorical sample frequencies {dev}")
        both_devices(body)

    def test_logits_use_softmax(self):
        # The fixed behaviour: logits -> probs via SOFTMAX (previously a sigmoid bug).
        logits = np.array([1.0, 2.0, 3.0, 0.5], dtype="float32")
        sm = np.exp(logits); sm = sm / sm.sum()
        def body(dev):
            d = D.Categorical(logits=jt.array(logits))
            self.ac(d.probs.numpy(), sm, atol=1e-5, msg=f"Categorical logits->softmax {dev}")
            # log_prob(k) == log(softmax(logits)[k])
            for k in range(4):
                self.ac(d.log_prob(jt.array([k])).numpy(), np.array([np.log(sm[k])]),
                        atol=1e-5, msg=f"Categorical logits log_prob({k}) {dev}")
            # entropy == -sum p log p of the softmax distribution
            self.assertAlmostEqual(self.scalar(d.entropy()),
                                   float(-(sm * np.log(sm)).sum()), delta=1e-5,
                                   msg=f"Categorical logits entropy {dev}")
        both_devices(body)

    def test_logits_equiv_probs(self):
        # Categorical(logits=L) must equal Categorical(probs=softmax(L)).
        logits = np.array([0.2, -0.5, 1.3, 0.7], dtype="float32")
        sm = np.exp(logits); sm = sm / sm.sum()
        def body(dev):
            dl = D.Categorical(logits=jt.array(logits))
            dp = D.Categorical(probs=jt.array(sm.astype("float32")))
            self.ac(dl.probs.numpy(), dp.probs.numpy(), atol=1e-6,
                    msg=f"Categorical logits==probs(softmax) probs {dev}")
            self.assertAlmostEqual(self.scalar(dl.entropy()), self.scalar(dp.entropy()),
                                   delta=1e-5, msg=f"Categorical logits==probs entropy {dev}")
        both_devices(body)


# ------------------------------------------------------------------------- Uniform

class TestUniform(Base):
    def test_log_prob_in_support(self):
        # in-support density is constant -log(high-low).
        low, high = 2.0, 5.0
        def body(dev):
            d = D.Uniform(low, high)
            self.assertAlmostEqual(self.scalar(d.log_prob(3.0)), -math.log(high - low),
                                   delta=1e-5, msg=f"Uniform.log_prob in-support {dev}")
        both_devices(body)

    def test_entropy(self):
        low, high = 2.0, 5.0
        def body(dev):
            d = D.Uniform(low, high)
            self.assertAlmostEqual(self.scalar(d.entropy()), math.log(high - low),
                                   delta=1e-5, msg=f"Uniform.entropy {dev}")
        both_devices(body)

    def test_sample(self):
        # fixed: Uniform.sample now maps jt.random(U[0,1)) to [low,high) (was jt.uniform,
        # which doesn't exist)
        low, high = 2.0, 5.0
        def body(dev):
            s = D.Uniform(low, high).sample((1000,)).numpy()
            self.assertEqual(s.shape, (1000,), f"Uniform.sample shape {dev}")
            self.assertTrue(s.min() >= low and s.max() < high, f"Uniform sample range {dev}")
        both_devices(body)

    def test_log_prob_out_of_support(self):
        # fixed: Uniform.log_prob now returns -inf (was +inf) outside [low,high)
        def body(dev):
            d = D.Uniform(2.0, 5.0)
            self.assertEqual(self.scalar(d.log_prob(10.0)), -math.inf,
                             f"Uniform.log_prob out-of-support should be -inf {dev}")
        both_devices(body)


# --------------------------------------------------------------------- Exponential

class TestExponential(Base):
    def setUp(self):
        self.rate = np.array([1.5, 0.5, 2.0], dtype="float32")

    def _dist(self):
        return D.Exponential(jt.array(self.rate))

    def test_log_prob(self):
        rate = self.rate
        x = np.array([0.7, 1.2, 0.3], dtype="float32")
        ref = np.log(rate) - rate * x          # log(rate) - rate*x  for x>=0
        def body(dev):
            self.ac(self._dist().log_prob(jt.array(x)).numpy(), ref, atol=1e-5,
                    msg=f"Exponential.log_prob {dev}")
        both_devices(body)

    def test_entropy(self):
        ref = 1.0 - np.log(self.rate)
        def body(dev):
            self.ac(self._dist().entropy().numpy(), ref, atol=1e-5,
                    msg=f"Exponential.entropy {dev}")
        both_devices(body)

    def test_sample_shape_and_nonneg(self):
        def body(dev):
            # batched rate (3,): sample() (no shape) == batch_shape (3,).
            d = self._dist()
            self.assertEqual(tuple(d.sample().shape), (3,), f"Exponential.sample() shape {dev}")
            # scalar rate: sample(shape) returns exactly `shape`.
            d0 = D.Exponential(jt.array([2.0]))
            s = d0.sample((2000,))
            self.assertEqual(tuple(s.shape), (2000,), f"Exponential.sample((2000,)) shape {dev}")
            self.assertTrue(float(s.min().item()) >= 0.0,
                            f"Exponential samples must be >= 0 {dev}")
        both_devices(body)

    def test_sample_mean(self):
        # E[X] = 1/rate ; coarse statistical check.
        rate = 2.0
        def body(dev):
            d = D.Exponential(jt.array([rate]))
            s = d.sample((40000,)).numpy()
            self.assertAlmostEqual(float(s.mean()), 1.0 / rate, delta=0.05,
                                   msg=f"Exponential sample mean ~ 1/rate {dev}")
        both_devices(body)

    def test_rsample_differentiable(self):
        # inverse-CDF sample -safe_log(1-u)/rate keeps a pathwise grad to `rate`,
        # so rsample (which falls back to sample here) is still differentiable w.r.t rate.
        def body(dev):
            rate_v = jt.array([1.5])
            d = D.Exponential(rate_v)
            s = d.rsample((128,))
            self.assertEqual(tuple(s.shape), (128,), f"Exponential.rsample shape {dev}")
            g = jt.grad(s.sum(), [rate_v])[0]
            self.assertNotEqual(float(g.numpy().sum()), 0.0,
                                f"Exponential.rsample grad rate must be nonzero {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
