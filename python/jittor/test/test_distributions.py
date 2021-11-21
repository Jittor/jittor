# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
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
            np.testing.assert_allclose(jn.log_prob(x),tn.log_prob(torch.tensor(x)))
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

if __name__ == "__main__":
    unittest.main()