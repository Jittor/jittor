
# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import random

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
except:
    torch = None
    skip_this_test = True
    
def check_equal(q,k,v,tatt,jatt):
    tq=torch.from_numpy(q)
    jq=jt.array(q)
    tk=torch.from_numpy(k)
    jk=jt.array(k)
    tv=torch.from_numpy(v)
    jv=jt.array(v)

    jatt.load_parameters(tatt.state_dict())
    ty, tw = tatt(tq, tk, tv)
    jy, jw = jatt(jq, jk, jv)
    assert np.allclose(ty.detach().numpy(), jy.numpy(), rtol=1e-3)
    assert np.allclose(tw.detach().numpy(), jw.numpy(), rtol=1e-3)

@unittest.skipIf(skip_this_test, "No Torch found")
class TestAttention(unittest.TestCase):
    def test_attention(self):
        j_opt = jt.optim.SGD([jt.array([1])], 1.0)
        t_opt = torch.optim.SGD([torch.ones([1])], 1.0)
        j_scheduler = jt.lr_scheduler.ReduceLROnPlateau(j_opt)
        t_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(t_opt)
        for i in range(100):
            loss=random.random()
            j_scheduler.step(loss)
            t_scheduler.step(loss)
            assert j_opt.lr == t_opt.state_dict()['param_groups'][0]['lr']

if __name__ == "__main__":
    unittest.main()