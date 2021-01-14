
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
import jittor.attention as jtatt
import numpy as np

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
    import fairseq
except:
    torch = None
    tnn = None
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
        q=np.random.rand(4,8,16).astype(np.float32)
        k=np.random.rand(4,8,16).astype(np.float32)
        v=np.random.rand(4,8,16).astype(np.float32)

        tatt=fairseq.modules.multihead_attention.MultiheadAttention(16,1)
        jatt=jt.attention.MultiheadAttention(16,1)
        check_equal(q,k,v,tatt,jatt)
        
        tatt=fairseq.modules.multihead_attention.MultiheadAttention(16,4)
        jatt=jt.attention.MultiheadAttention(16,4)
        check_equal(q,k,v,tatt,jatt)
        
        tatt=fairseq.modules.multihead_attention.MultiheadAttention(16,1,self_attention=True)
        jatt=jt.attention.MultiheadAttention(16,1,self_attention=True)
        check_equal(q,q,q,tatt,jatt)
        
        tatt=fairseq.modules.multihead_attention.MultiheadAttention(16,4,self_attention=True)
        jatt=jt.attention.MultiheadAttention(16,4,self_attention=True)
        check_equal(q,q,q,tatt,jatt)

if __name__ == "__main__":
    unittest.main()