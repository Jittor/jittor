
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

    def test_one_hot(self):
        a = jd.OneHotCategorical(jt.array([0.25, 0.25, 0.25, 0.25]))
        x = a.sample().numpy()
        for i in range(1000):
            x += a.sample().numpy()
        print(x)
        assert (x > 200).all()
        y = a.sample([2,3])
        y.sync()
        assert y.shape == [2,3,4]

    def test_cate(self):
        a = jd.Categorical(jt.array([0.25, 0.25, 0.25, 0.25]))
        x =np.array([0,0,0,0])
        for i in range(1000):
            x[a.sample().item()]+=1
        assert (x > 200).all()
        y = a.sample([2,3])
        y.sync()
        assert y.shape == [2,3]



if __name__ == "__main__":
    unittest.main()