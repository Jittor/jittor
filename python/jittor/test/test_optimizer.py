# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#    Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from jittor import nn

class TestOptimizer(unittest.TestCase):
    def test_param_groups(self):
        pa = jt.ones((1,))
        pb = jt.ones((1,))
        data = jt.ones((1,))
        opt = nn.SGD([
            {"params":[pa], "lr":0.1}, 
            {"params":[pb]}, 
        ], 1)
        opt.step(pa*data+pb*data)
        assert pa.data == 0.9 and pb.data == 0, (pa, pb)

    def test_clip_grad_norm(self):
        a = jt.ones(2)
        opt = jt.optim.SGD([a], 0.1)

        loss = a*a
        opt.zero_grad()
        opt.backward(loss)
        opt.clip_grad_norm(0.01, 2)
        assert np.allclose(opt.param_groups[0]['grads'][0].norm(), 0.01)
        opt.step()

    def test_state_dict(self):
        a = jt.ones(2)
        opt = jt.optim.SGD([a], 0.1)
        s = opt.state_dict()
        # print(s)
        opt.load_state_dict(s)

    def test_opt_grad(self):
        a = jt.ones(2)
        opt = jt.optim.SGD([a], 0.1)
        opt.backward(a**2)
        g = a.opt_grad(opt)
        np.testing.assert_allclose(g.data, 2)


        

if __name__ == "__main__":
    unittest.main()