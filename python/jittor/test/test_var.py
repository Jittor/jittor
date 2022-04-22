# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
#     Zheng-Ning Liu <lzhengning@gmail.com>
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import jittor.nn as jnn

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    skip_this_test = True

@unittest.skipIf(skip_this_test, "No Torch found")
class TestVarFunctions(unittest.TestCase):
    def test_var(self):
        x = np.random.randn(100, 1000).astype(np.float32)

        jt_x = jt.array(x)
        tc_x = torch.from_numpy(x)
        np.testing.assert_allclose(jt_x.var().numpy(), tc_x.var().numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(jt_x.var(dim=1).numpy(), tc_x.var(dim=1).numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(jt_x.var(dim=0, unbiased=True).numpy(), tc_x.var(dim=0, unbiased=True).numpy(), rtol=1e-3, atol=1e-4)
        
    def test_std(self):
        x=np.random.randn(100, 1000).astype(np.float32)
        jt_x = jt.array(x)
        tc_x = torch.from_numpy(x)
        np.testing.assert_allclose(jt_x.std().numpy(), tc_x.std().numpy(), 1e-4)

    def test_norm(self):
        x = np.random.randn(100, 1000).astype(np.float32)
        jt_x = jt.array(x)
        tc_x = torch.from_numpy(x)
        np.testing.assert_allclose(jt_x.norm(1,1).numpy(), tc_x.norm(1,1).numpy(), atol=1e-6)
        np.testing.assert_allclose(jt_x.norm(1,0).numpy(), tc_x.norm(1,0).numpy(), atol=1e-6)
        np.testing.assert_allclose(jt_x.norm(2,1).numpy(), tc_x.norm(2,1).numpy(), atol=1e-6)
        np.testing.assert_allclose(jt_x.norm(2,0).numpy(), tc_x.norm(2,0).numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
