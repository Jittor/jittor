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


class TestOneHot(unittest.TestCase):
    def test_presum(self):
        a = jt.array([[1,2,3,4]])
        b = jd.simple_presum(a)
        assert (b.data == [[0,1,3,6,10]]).all()
    
    def test_lgamma(self):
        import torch
        ta = np.random.uniform(2,3,(1))
        a = jt.array(ta).float32()
        assert np.allclose(jd.lgamma(a).data, torch.lgamma(torch.tensor(ta)).numpy()),(jd.lgamma(a).data, torch.lgamma(torch.tensor(ta)).numpy())

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
        import torch
        tc, tc2 = torch.distributions.OneHotCategorical(torch.tensor(probs).to(torch.float32)),torch.distributions.OneHotCategorical(torch.tensor(probs2).to(torch.float32))
        jc, jc2 = jd.OneHotCategorical(jt.array(probs).reshape(1,-1)),jd.OneHotCategorical(jt.array(probs2).reshape(1,-1))
        # print(jc.probs,tc.probs)
        # print(jc.logits,tc.logits)
        assert np.allclose(jc.entropy().data,tc.entropy().numpy()), (jc.entropy().data, tc.entropy().numpy())
        x = np.zeros((4,10))
        for _ in range(4):
            nx = np.random.randint(0,9)    
            x[_,nx] = 1
        assert np.allclose(tc.log_prob(torch.tensor(x).to(torch.float32)),jc.log_prob(jt.array(x)))
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
        
    def test_normal(self):
        import torch
        for _ in range(4):
            mu = np.random.uniform(-1,1)
            sigma = np.random.uniform(0,2)
            jn = jd.Normal(mu,sigma)
            tn = torch.distributions.Normal(mu,sigma)
            assert np.allclose(jn.entropy().data,tn.entropy().numpy())
            x = np.random.uniform(-1,1)
            assert np.allclose(jn.log_prob(x),tn.log_prob(torch.tensor(x)))
            mu2 = np.random.uniform(-1,1)
            sigma2 = np.random.uniform(0,2)
            jn2 = jd.Normal(mu2,sigma2)
            tn2 = torch.distributions.Normal(mu2,sigma2)
            assert np.allclose(jd.kl_divergence(jn,jn2).data,torch.distributions.kl_divergence(tn,tn2).numpy())

    def test_categorical(self):
        import torch
        for _ in range(4):
            probs,probs2 = np.random.uniform(0,1,(10)), np.random.uniform(0,1,(10))
            probs,probs2 = probs / probs.sum(),probs2 / probs2.sum()
            tc, tc2 = torch.distributions.Categorical(torch.tensor(probs)),torch.distributions.Categorical(torch.tensor(probs2))
            jc, jc2 = jd.Categorical(jt.array(probs).reshape(1,-1)),jd.Categorical(jt.array(probs2).reshape(1,-1))
            assert np.allclose(jc.entropy().data, tc.entropy().numpy()), (jc.entropy().data, tc.entropy().numpy())
            x = np.random.randint(0,10,(4))
            assert np.allclose(jc.log_prob(x), tc.log_prob(torch.tensor(x)))
            assert np.allclose(jd.kl_divergence(jc,jc2),torch.distributions.kl_divergence(tc,tc2))
            
    def test_uniform(self):
        import torch
        for _ in range(4):
            low, low2 = np.random.randint(-1,2), np.random.randint(-1,2)
            leng, leng2 = np.random.uniform(0,2), np.random.uniform(0,2)
            high, high2 = low + leng, low2 + leng2
            ju, ju2 = jd.Uniform(low,high),jd.Uniform(low2,high2)
            tu, tu2 = torch.distributions.Uniform(low,high),torch.distributions.Uniform(low2,high2)
            assert np.allclose(ju.entropy().data,tu.entropy().numpy()),(ju.entropy().data,tu.entropy().numpy())
            x = np.random.uniform(low,high)
            assert np.allclose(ju.log_prob(x),tu.log_prob(torch.tensor(x)))
            assert np.allclose(jd.kl_divergence(ju,ju2),torch.distributions.kl_divergence(tu,tu2))
    
    def test_geometric(self):
        import torch
        for _ in range(4):
            prob, prob2 = np.random.uniform(0,1), np.random.uniform(0,1)
            jg, jg2 = jd.Geometric(prob),jd.Geometric(prob2)
            tg, tg2 = torch.distributions.Geometric(prob),torch.distributions.Geometric(prob2)
            assert np.allclose(jg.entropy().data,tg.entropy().numpy()),(jg.entropy().data,tg.entropy().numpy())
            x = np.random.randint(1,10)
            assert np.allclose(jg.log_prob(jt.array(x)),tg.log_prob(torch.tensor(x)))
            # print(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2))
            assert np.allclose(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2)),(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2))
    
    def test_poisson(self):
        import torch
        for _ in range(4):
            prob, prob2 = np.random.uniform(0,1), np.random.uniform(0,1)
            jp, jp2 = jd.Poisson(prob),jd.Poisson(prob2)
            tp, tp2 = torch.distributions.Poisson(prob),torch.distributions.Poisson(prob2)
            x = np.random.randint(1,10)
            assert np.allclose(jp.log_prob(jt.array(x).float32()),tp.log_prob(torch.tensor(x)))
            # print(jd.kl_divergence(jg,jg2),torch.distributions.kl_divergence(tg,tg2))
            assert np.allclose(jd.kl_divergence(jp,jp2),torch.distributions.kl_divergence(tp,tp2)),(jd.kl_divergence(jp,jp2),torch.distributions.kl_divergence(tp,tp2))


if __name__ == "__main__":
    unittest.main()
