# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import math
import unittest
import jittor as jt
import numpy as np
import jittor.distributions as jd
from _helpers.torch_runtime import import_torch_modules, modules_available

skip_this_test = not modules_available("torch")
torch = None


def setUpModule():
    global torch
    if not skip_this_test:
        (torch,) = import_torch_modules("torch")


class TestOneHot(unittest.TestCase):
    def test_presum(self):
        a = jt.array([[1,2,3,4]])
        b = jd.simple_presum(a)
        assert (b.data == [[0,1,3,6,10]]).all()

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_one_hot(self):
        a = jd.OneHotCategorical(jt.array([0.25, 0.25, 0.25, 0.25]))
        x = a.sample().numpy()
        for i in range(1000):
            x += a.sample().numpy()
        assert (x > 200).all()
        y = a.sample([2,3])
        y.sync()
        assert y.shape == [2,3,4]
        probs,probs2 = np.random.uniform(0,1,(10)), np.random.uniform(0,1,(10))
        probs,probs2 = probs / probs.sum(),probs2 / probs2.sum()

        jc, jc2 = jd.OneHotCategorical(jt.array(probs)),jd.OneHotCategorical(jt.array(probs2))
        tc, tc2 = torch.distributions.OneHotCategorical(torch.tensor(probs)),torch.distributions.OneHotCategorical(torch.tensor(probs2))
        assert np.allclose(jc.entropy().data,tc.entropy().numpy())
        x = np.zeros((4,10))
        for _ in range(4):
            nx = np.random.randint(0,9)    
            x[_,nx] = 1
        np.testing.assert_allclose(jc.log_prob(jt.array(x)),tc.log_prob(torch.tensor(x)), atol=1e-5)
        assert np.allclose(jd.kl_divergence(jc,jc2),torch.distributions.kl_divergence(tc,tc2))

    def test_cate(self):
        a = jd.Categorical(jt.array([0.25, 0.25, 0.25, 0.25]))
        x =np.array([0,0,0,0])
        for i in range(1000):
            x[a.sample().item()]+=1
        assert (x > 200).all()
        y = a.sample([2,3])
        y.sync()
        assert y.shape == [2,3]
        
    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_normal(self):
        for _ in range(4):
            mu = np.random.uniform(-1,1)
            sigma = np.random.uniform(0,2)
            jn = jd.Normal(mu,sigma)
            tn = torch.distributions.Normal(mu,sigma)
            assert np.allclose(jn.entropy().data,tn.entropy().numpy())
            x = np.random.uniform(-1,1)
            # jittor computes in float32, torch's ref in float64; the default
            # assert_allclose(rtol=1e-7, atol=0) is below float32 precision (~1e-6) and
            # flaked when log_prob landed near 0 / sigma near 0. Use a float32-appropriate
            # tolerance (this is the underlying math, not a jittor bug).
            np.testing.assert_allclose(jn.log_prob(x),tn.log_prob(torch.tensor(x)), rtol=1e-4, atol=1e-5)
            mu2 = np.random.uniform(-1,1)
            sigma2 = np.random.uniform(0,2)
            jn2 = jd.Normal(mu2,sigma2)
            tn2 = torch.distributions.Normal(mu2,sigma2)
            assert np.allclose(jd.kl_divergence(jn,jn2).data,torch.distributions.kl_divergence(tn,tn2).numpy())

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_categorical1(self):
        for _ in range(4):
            probs,probs2 = np.random.uniform(0,1,(10)), np.random.uniform(0,1,(10))
            probs,probs2 = probs / probs.sum(),probs2 / probs2.sum()
            jc, jc2 = jd.Categorical(jt.array(probs)),jd.Categorical(jt.array(probs2))
            tc, tc2 = torch.distributions.Categorical(torch.tensor(probs)),torch.distributions.Categorical(torch.tensor(probs2))
            assert np.allclose(jc.entropy().data, tc.entropy().numpy()), (jc.entropy().data, tc.entropy().numpy())
            x = np.random.randint(0,10,(4))
            np.testing.assert_allclose(jc.log_prob(x), tc.log_prob(torch.tensor(x)), atol=1e-5)
            assert np.allclose(jd.kl_divergence(jc,jc2),torch.distributions.kl_divergence(tc,tc2))

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_categorical2(self):
        def check(prob_shape, sample_shape):
            for _ in range(4):
                probs,probs2 = np.random.uniform(0,1,prob_shape), np.random.uniform(0,1, prob_shape)

                jc, jc2 = jd.Categorical(jt.array(probs)),jd.Categorical(jt.array(probs2))
                tc, tc2 = torch.distributions.Categorical(torch.tensor(probs)),torch.distributions.Categorical(torch.tensor(probs2))
                assert np.allclose(jc.entropy().data, tc.entropy().numpy()), (jc.entropy().data, tc.entropy().numpy())
                x1 = jc.sample(sample_shape)
                x2 = tc.sample(sample_shape)
                assert tuple(x1.shape) == tuple(x2.shape)
                x = np.random.randint(0,prob_shape[-1], tuple(x1.shape))
                np.testing.assert_allclose(jc.log_prob(x), tc.log_prob(torch.tensor(x)), atol=1e-5)
                np.testing.assert_allclose(jd.kl_divergence(jc,jc2), torch.distributions.kl_divergence(tc,tc2), atol=1e-5)
        check((10,), (4,))
        check((2,3), (4,))
        check((3,4,5,6), (2,))

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_one_hot_categorical2(self):
        def check(prob_shape, sample_shape):
            for _ in range(4):
                probs,probs2 = np.random.uniform(0,1,prob_shape), np.random.uniform(0,1, prob_shape)

                jc, jc2 = jd.OneHotCategorical(jt.array(probs)),jd.OneHotCategorical(jt.array(probs2))
                tc, tc2 = torch.distributions.OneHotCategorical(torch.tensor(probs)),torch.distributions.OneHotCategorical(torch.tensor(probs2))
                assert np.allclose(jc.entropy().data, tc.entropy().numpy()), (jc.entropy().data, tc.entropy().numpy())
                x1 = jc.sample(sample_shape)
                x2 = tc.sample(sample_shape)
                assert tuple(x1.shape) == tuple(x2.shape)
                indexes = np.random.randint(0, prob_shape[-1], tuple(x1.shape[:-1]))
                x = np.eye(prob_shape[-1], dtype="float32")[indexes]
                np.testing.assert_allclose(jc.log_prob(x), tc.log_prob(torch.tensor(x)), atol=1e-5)
                np.testing.assert_allclose(jd.kl_divergence(jc,jc2), torch.distributions.kl_divergence(tc,tc2), atol=1e-5)
        check((10,), (4,))
        check((2,3), (4,))
        check((3,4,5,6), (2,))

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_categorical_logits(self):
        # The logits= path was broken (sigmoid+renorm instead of softmax; raw logits
        # stored for entropy) -> wrong log_prob/entropy, which breaks PPO/RLHF. It must
        # match torch AND be differentiable wrt the logits (policy gradient).
        for shape in [(10,), (2, 3), (3, 4, 5)]:
            logits = np.random.randn(*shape).astype('float32')
            jc = jd.Categorical(logits=jt.array(logits))
            tc = torch.distributions.Categorical(logits=torch.tensor(logits))
            np.testing.assert_allclose(jc.entropy().data, tc.entropy().numpy(), atol=1e-5)
            x = np.random.randint(0, shape[-1], shape[:-1] if len(shape) > 1 else (4,))
            np.testing.assert_allclose(jc.log_prob(x), tc.log_prob(torch.tensor(x)), atol=1e-5)
        # differentiability (PPO needs grad through log_prob + entropy)
        L = jt.array(np.random.randn(3, 5).astype('float32'))
        dc = jd.Categorical(logits=L)
        loss = dc.log_prob(jt.array(np.array([0, 2, 4], dtype='int64'))).sum() + dc.entropy().sum()
        g = jt.grad(loss, [L])[0]
        assert bool(jt.isfinite(g).all().item()) and float(jt.abs(g).sum().item()) > 0, \
            "Categorical log_prob/entropy must be differentiable wrt logits"

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_bernoulli_exponential_independent(self):
        # newly added distributions (Bernoulli/Exponential/Independent/Distribution base).
        # Bernoulli's logits->probs map is sigmoid (correct, unlike Categorical's softmax).
        logits = np.random.randn(5).astype('float32')
        jb, tb = jd.Bernoulli(logits=jt.array(logits)), torch.distributions.Bernoulli(logits=torch.tensor(logits))
        xb = (np.random.rand(5) > 0.5).astype('float32')
        np.testing.assert_allclose(jb.log_prob(jt.array(xb)).data, tb.log_prob(torch.tensor(xb)).numpy(), atol=1e-5)
        np.testing.assert_allclose(jb.entropy().data, tb.entropy().numpy(), atol=1e-5)
        self.assertTrue(isinstance(jb, jd.Distribution))
        rate = (np.abs(np.random.randn(4)) + 0.2).astype('float32')
        je, te = jd.Exponential(jt.array(rate)), torch.distributions.Exponential(torch.tensor(rate))
        xe = (np.abs(np.random.randn(4)) + 0.1).astype('float32')
        np.testing.assert_allclose(je.log_prob(jt.array(xe)).data, te.log_prob(torch.tensor(xe)).numpy(), atol=1e-5)
        np.testing.assert_allclose(je.entropy().data, te.entropy().numpy(), atol=1e-5)
        mu, sig = np.random.randn(3).astype('float32'), (np.abs(np.random.randn(3)) + 0.2).astype('float32')
        ji = jd.Independent(jd.Normal(jt.array(mu), jt.array(sig)), 1)
        ti = torch.distributions.Independent(torch.distributions.Normal(torch.tensor(mu), torch.tensor(sig)), 1)
        xv = np.random.randn(3).astype('float32')
        np.testing.assert_allclose(ji.log_prob(jt.array(xv)).data, ti.log_prob(torch.tensor(xv)).numpy(), atol=1e-5)

    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_uniform(self):
        for _ in range(4):
            low, low2 = np.random.randint(-1,2), np.random.randint(-1,2)
            leng, leng2 = np.random.uniform(0,2), np.random.uniform(0,2)
            high, high2 = low + leng, low2 + leng2
            ju, ju2 = jd.Uniform(low,high),jd.Uniform(low2,high2)
            tu, tu2 = torch.distributions.Uniform(low,high),torch.distributions.Uniform(low2,high2)
            assert np.allclose(ju.entropy().data,tu.entropy().numpy())
            x = np.random.uniform(low,high)
            assert np.allclose(ju.log_prob(x),tu.log_prob(torch.tensor(x)))
            assert np.allclose(jd.kl_divergence(ju,ju2),torch.distributions.kl_divergence(tu,tu2))
    
    @unittest.skipIf(skip_this_test, "No Torch Found")
    def test_geometric(self):
        cases = ((0.1, 0.7, 0), (0.25, 0.6, 2), (0.6, 0.2, 5), (0.9, 0.4, 8))
        for prob, prob2, x in cases:
            jg, jg2 = jd.Geometric(prob),jd.Geometric(prob2)
            tg, tg2 = torch.distributions.Geometric(prob),torch.distributions.Geometric(prob2)
            np.testing.assert_allclose(jg.entropy().data,tg.entropy().numpy(), atol=1e-5)
            np.testing.assert_allclose(jg.log_prob(x),tg.log_prob(torch.tensor(x)), atol=1e-5)
            np.testing.assert_allclose(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2), atol=1e-5)

# Reference values captured from REAL PyTorch (torch.distributions) in a clean env.
# Hardcoded on purpose: in a jittor-as-torch deployment the in-process `torch` is the
# jittor shim (torch.distributions IS jittor.distributions, torch.lgamma IS jittor's),
# so comparing against it would be a tautology. These constants make this a TRUE oracle.
_REF = {
    "beta_lp": [0.5675839185714722, 0.40157508850097656],
    "beta_ent": [-0.23490619659423828, -0.27300751209259033],
    "beta_mean": [0.4000000059604645, 0.6666666865348816],
    "beta_var": [0.03999999910593033, 0.04040404036641121],
    "gamma_lp": [-1.0945348739624023, -0.9397292137145996],
    "gamma_ent": [1.5772157907485962, 1.4604358673095703],
    "gamma_mean": [2.0, 2.5], "gamma_var": [2.0, 1.25],
    "pois_lp": [-1.3068528175354004, -1.7403020858764648],
    "pois_mean": [2.0, 5.0], "pois_var": [2.0, 5.0],
    "dir_lp": [1.504077434539795, 1.4632554054260254],
    "dir_ent": [-1.2443442344665527, -0.9374915361404419],
    "dir_mean": [[0.16666667, 0.33333334, 0.5], [0.33333334, 0.33333334, 0.33333334]],
    "ln_lp": [-0.9189385175704956, -1.2934778928756714],
    "ln_ent": [1.4189385175704956, 1.5622634887695312],
    "ln_mean": [1.6487212181091309, 2.1064414978027344],
    "ln_var": [4.670774459838867, 2.805647373199463],
    "mvn_lp": -2.828968048095703, "mvn_ent": 3.161428689956665,
    "mvn_lp_batch": [-2.828968048095703, -2.161428689956665],
    "mvn_mean": [1.0, -1.0], "mvn_var": [1.9999998807907104, 0.9999999403953552],
    "lgamma_grad": [0.03648991510272026, 0.7031567692756653, 1.1031566858291626],
    "beta_ent_grad": [0.019034862518310547, -0.16805529594421387],
}


def _grad1(out, t):
    r = jt.grad(out, t)
    return r[0] if isinstance(r, (list, tuple)) else r


class TestMoreDistributions(unittest.TestCase):
    ''' Beta / Gamma / Poisson / Dirichlet / LogNormal / MultivariateNormal parity vs
    real torch.distributions (log_prob / entropy / mean / variance), plus the lgamma
    backward fix that makes their parameter-gradients differentiable. Asserts against
    hardcoded real-torch oracle constants (_REF) -- independent of the in-env torch. '''

    def _ac(self, got, key, atol=1e-5):
        np.testing.assert_allclose(np.asarray(got.data), np.asarray(_REF[key]), atol=atol)

    def test_beta(self):
        j = jd.Beta([2.0, 3.0], [3.0, 1.5])
        self._ac(j.log_prob(jt.array([0.3, 0.6])), "beta_lp")
        self._ac(j.entropy(), "beta_ent")
        self._ac(j.mean, "beta_mean"); self._ac(j.variance, "beta_var")

    def test_gamma(self):
        j = jd.Gamma([2.0, 5.0], [1.0, 2.0])
        self._ac(j.log_prob(jt.array([1.5, 2.0])), "gamma_lp")
        self._ac(j.entropy(), "gamma_ent")
        self._ac(j.mean, "gamma_mean"); self._ac(j.variance, "gamma_var")

    def test_poisson(self):
        j = jd.Poisson([2.0, 5.0])
        self._ac(j.log_prob(jt.array([1.0, 4.0])), "pois_lp")
        self._ac(j.mean, "pois_mean"); self._ac(j.variance, "pois_var")

    def test_dirichlet(self):
        j = jd.Dirichlet([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        self._ac(j.log_prob(jt.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])), "dir_lp")
        self._ac(j.entropy(), "dir_ent"); self._ac(j.mean, "dir_mean")

    def test_lognormal(self):
        j = jd.LogNormal([0.0, 0.5], [1.0, 0.7])
        self._ac(j.log_prob(jt.array([1.0, 2.0])), "ln_lp")
        self._ac(j.entropy(), "ln_ent")
        self._ac(j.mean, "ln_mean"); self._ac(j.variance, "ln_var", atol=1e-4)

    def test_multivariate_normal(self):
        j = jd.MultivariateNormal([1.0, -1.0], [[2.0, 0.3], [0.3, 1.0]])
        # jittor has no 0-d scalar -> (1,) vs torch (); assert_allclose broadcasts
        self._ac(j.log_prob(jt.array([0.5, 0.0])), "mvn_lp")
        self._ac(j.entropy(), "mvn_ent")
        self._ac(j.log_prob(jt.array([[0.5, 0.0], [1.0, -1.0]])), "mvn_lp_batch")
        self._ac(j.mean, "mvn_mean"); self._ac(j.variance, "mvn_var")

    def test_lgamma_backward_parity(self):
        # jittor's lgamma now defines a backward (= digamma), matching torch -- this is
        # what makes Gamma/Beta/Dirichlet entropy & log_prob differentiable w.r.t. the
        # concentration parameters (RL / variational inference need this).
        jx = jt.array([1.5, 2.5, 3.5])
        self._ac(_grad1(jt.lgamma.apply(jx).sum(), jx), "lgamma_grad")
        ja = jt.array([2.0, 3.0])
        self._ac(_grad1(jd.Beta(ja, jt.array([3.0, 1.5])).entropy().sum(), ja), "beta_ent_grad")


def _np_softplus(z):
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _np_log_softmax(z):
    m = z.max(axis=-1, keepdims=True)
    shifted = z - m
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class TestRelaxedDistributions(unittest.TestCase):
    """LogitRelaxedBernoulli / RelaxedBernoulli / ExpRelaxedCategorical /
    RelaxedOneHotCategorical against torch's formulas.

    ``LogitRelaxedBernoulli`` used to be a plain alias of ``RelaxedBernoulli``
    (so its samples came back squashed through a sigmoid and its log_prob was
    the wrong density), and ``RelaxedOneHotCategorical`` inherited the discrete
    ``OneHotCategorical.log_prob`` while pointing ``base_dist`` at itself.

    The numpy references below are torch's own formulas; the expected numbers
    are what torch 2.12.1 prints for the same inputs, so the references are
    pinned to the real library, not to a re-derivation.
    """

    TOL = 2e-4

    T = 0.5
    LOGITS = np.array([[-1.0, 0.5], [2.0, -0.25]], dtype=np.float64)
    LOGIT_VALUES = np.array([[0.3, -1.2], [0.75, 2.5]], dtype=np.float64)
    UNIT_VALUES = np.array([[0.2, 0.7], [0.55, 0.9]], dtype=np.float64)
    CLOGITS = np.array([[0.1, -0.4, 1.2], [-2.0, 0.5, 0.3]], dtype=np.float64)
    SIMPLEX_VALUES = np.array([[0.5, 0.2, 0.3], [0.1, 0.6, 0.3]], dtype=np.float64)

    # torch 2.12.1 reference output
    TORCH_LOGIT_RELAXED_BERNOULLI_LOG_PROB = np.array(
        [[-2.3933083469327423, -2.367817830790807],
         [-2.6776364513312636, -2.5959737365254503]])
    TORCH_RELAXED_BERNOULLI_LOG_PROB = np.array(
        [[-0.27030796411579217, -0.5202508110292254],
         [-1.4753279283138316, -0.0954026955206726]])
    TORCH_EXP_RELAXED_CATEGORICAL_LOG_PROB = np.array(
        [-4.672430492080289, -4.725313779988706])
    TORCH_RELAXED_ONE_HOT_LOG_PROB = np.array(
        [-1.165872594760307, -0.7079302589027341])

    # ---- numpy references (torch's formulas) ------------------------------
    def _ref_logit_relaxed_bernoulli_log_prob(self, value):
        diff = self.LOGITS - value * self.T
        return np.log(self.T) + diff - 2 * _np_softplus(diff)

    def _ref_relaxed_bernoulli_log_prob(self, value):
        x = np.log(value) - np.log1p(-value)
        return (self._ref_logit_relaxed_bernoulli_log_prob(x)
                + _np_softplus(x) + _np_softplus(-x))

    def _ref_exp_relaxed_categorical_log_prob(self, log_value):
        K = self.CLOGITS.shape[-1]
        log_scale = math.lgamma(K) + (K - 1) * np.log(self.T)
        score = self.CLOGITS - log_value * self.T
        return _np_log_softmax(score).sum(-1) + log_scale

    def _ref_relaxed_one_hot_log_prob(self, value):
        log_value = np.log(value)
        return (self._ref_exp_relaxed_categorical_log_prob(log_value)
                - log_value.sum(-1))

    def test_numpy_references_match_torch(self):
        np.testing.assert_allclose(
            self._ref_logit_relaxed_bernoulli_log_prob(self.LOGIT_VALUES),
            self.TORCH_LOGIT_RELAXED_BERNOULLI_LOG_PROB, rtol=1e-12)
        np.testing.assert_allclose(
            self._ref_relaxed_bernoulli_log_prob(self.UNIT_VALUES),
            self.TORCH_RELAXED_BERNOULLI_LOG_PROB, rtol=1e-12)
        np.testing.assert_allclose(
            self._ref_exp_relaxed_categorical_log_prob(np.log(self.SIMPLEX_VALUES)),
            self.TORCH_EXP_RELAXED_CATEGORICAL_LOG_PROB, rtol=1e-12)
        np.testing.assert_allclose(
            self._ref_relaxed_one_hot_log_prob(self.SIMPLEX_VALUES),
            self.TORCH_RELAXED_ONE_HOT_LOG_PROB, rtol=1e-12)

    # ---- jittor ------------------------------------------------------------
    def _v(self, arr):
        return jt.array(arr.astype("float32"), dtype="float32")

    def test_logit_relaxed_bernoulli_is_its_own_distribution(self):
        assert jd.LogitRelaxedBernoulli is not jd.RelaxedBernoulli
        d = jd.LogitRelaxedBernoulli(self.T, logits=self._v(self.LOGITS))
        np.testing.assert_allclose(
            d.log_prob(self._v(self.LOGIT_VALUES)).numpy(),
            self.TORCH_LOGIT_RELAXED_BERNOULLI_LOG_PROB,
            atol=self.TOL, rtol=self.TOL)

    def test_logit_relaxed_bernoulli_samples_are_unbounded(self):
        jt.set_global_seed(1234)
        d = jd.LogitRelaxedBernoulli(self.T, logits=self._v(self.LOGITS))
        x = d.rsample((4096,)).numpy()
        self.assertEqual(x.shape, (4096, 2, 2))
        # a logit-space sample must leave (0, 1) -- the old alias returned
        # sigmoid(...) and never did.
        assert (x < 0).any() and (x > 1).any(), (x.min(), x.max())
        # and sigmoid of it lands in RelaxedBernoulli's support
        y = 1.0 / (1.0 + np.exp(-x))
        assert ((y >= 0) & (y <= 1)).all()

    def test_relaxed_bernoulli_log_prob_and_base_dist(self):
        d = jd.RelaxedBernoulli(self.T, logits=self._v(self.LOGITS))
        assert isinstance(d.base_dist, jd.LogitRelaxedBernoulli)
        np.testing.assert_allclose(
            d.log_prob(self._v(self.UNIT_VALUES)).numpy(),
            self.TORCH_RELAXED_BERNOULLI_LOG_PROB,
            atol=self.TOL, rtol=self.TOL)
        jt.set_global_seed(4321)
        y = d.rsample((512,)).numpy()
        self.assertEqual(y.shape, (512, 2, 2))
        assert ((y >= 0) & (y <= 1)).all()

    def test_exp_relaxed_categorical_log_prob(self):
        d = jd.ExpRelaxedCategorical(self.T, logits=self._v(self.CLOGITS))
        np.testing.assert_allclose(
            d.log_prob(self._v(np.log(self.SIMPLEX_VALUES))).numpy(),
            self.TORCH_EXP_RELAXED_CATEGORICAL_LOG_PROB,
            atol=self.TOL, rtol=self.TOL)
        jt.set_global_seed(99)
        x = d.rsample((16,)).numpy()
        self.assertEqual(x.shape, (16, 2, 3))
        # samples are log-probability vectors: exponentiate to the simplex
        np.testing.assert_allclose(np.exp(x).sum(-1), 1.0, atol=1e-5)

    def test_relaxed_one_hot_log_prob_and_base_dist(self):
        d = jd.RelaxedOneHotCategorical(self.T, logits=self._v(self.CLOGITS))
        assert isinstance(d.base_dist, jd.ExpRelaxedCategorical)
        assert d.base_dist is not d
        np.testing.assert_allclose(
            d.log_prob(self._v(self.SIMPLEX_VALUES)).numpy(),
            self.TORCH_RELAXED_ONE_HOT_LOG_PROB,
            atol=self.TOL, rtol=self.TOL)
        jt.set_global_seed(7)
        y = d.rsample((16,)).numpy()
        self.assertEqual(y.shape, (16, 2, 3))
        np.testing.assert_allclose(y.sum(-1), 1.0, atol=1e-5)

    def test_relaxed_log_prob_differs_from_the_discrete_parent(self):
        # the whole point: the inherited discrete log_prob answers a different
        # question, so the two must not agree on a relaxed sample.
        d = jd.RelaxedOneHotCategorical(self.T, logits=self._v(self.CLOGITS))
        discrete = jd.OneHotCategorical(logits=self._v(self.CLOGITS))
        relaxed = d.log_prob(self._v(self.SIMPLEX_VALUES)).numpy()
        assert not np.allclose(relaxed,
                               discrete.log_prob(self._v(self.SIMPLEX_VALUES)).numpy(),
                               atol=1e-3)

    def test_no_closed_form_moments(self):
        # torch raises NotImplementedError for all of these; inheriting the
        # discrete answer would be silently wrong.
        rb = jd.RelaxedBernoulli(self.T, logits=self._v(self.LOGITS))
        roc = jd.RelaxedOneHotCategorical(self.T, logits=self._v(self.CLOGITS))
        lrb = jd.LogitRelaxedBernoulli(self.T, logits=self._v(self.LOGITS))
        for dist, names in ((rb, ("entropy", "mean", "mode")),
                            (roc, ("entropy", "mode")),
                            (lrb, ("entropy",))):
            for name in names:
                with self.assertRaises(NotImplementedError):
                    attr = getattr(dist, name)
                    attr() if callable(attr) else attr


if __name__ == "__main__":
    unittest.main()
