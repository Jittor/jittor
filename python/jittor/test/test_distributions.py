# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import jittor.distributions as jd

skip_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
except:
    torch = None
    skip_this_test = True


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
                x = np.random.randint(0,prob_shape[-1], tuple(x1.shape))
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
        for _ in range(4):
            prob, prob2 = np.random.uniform(0,1), np.random.uniform(0,1)
            jg, jg2 = jd.Geometric(prob),jd.Geometric(prob2)
            tg, tg2 = torch.distributions.Geometric(prob),torch.distributions.Geometric(prob2)
            np.testing.assert_allclose(jg.entropy().data,tg.entropy().numpy(), atol=1e-4)
            x = np.random.randint(1,10)
            np.testing.assert_allclose(jg.log_prob(x),tg.log_prob(torch.tensor(x)), atol=1e-4)
            # print(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2))
            np.testing.assert_allclose(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2), atol=1e-4)

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


if __name__ == "__main__":
    unittest.main()