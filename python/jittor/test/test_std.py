# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import os
import numpy as np
import jittor.nn as jnn

from jittor.test.test_log import find_log_with_re
skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    skip_this_test = True

@unittest.skipIf(skip_this_test, "No Torch found")
class TestStd(unittest.TestCase):
    def test_std(self):
        x=np.random.randn(100,1000).astype(np.float32)
        jt_x=jt.array(x)
        tc_x=torch.from_numpy(x)
        assert np.allclose(jt_x.std().numpy(), tc_x.std().numpy(), 1e-4) ,(x, jt_x.std().numpy(), tc_x.std().numpy())

    def test_norm(self):
        x=np.random.randn(100,1000).astype(np.float32)
        jt_x=jt.array(x)
        tc_x=torch.from_numpy(x)
        assert np.allclose(jt_x.norm(1,1).numpy(), tc_x.norm(1,1).numpy())
        assert np.allclose(jt_x.norm(1,0).numpy(), tc_x.norm(1,0).numpy())
        assert np.allclose(jt_x.norm(2,1).numpy(), tc_x.norm(2,1).numpy())
        assert np.allclose(jt_x.norm(2,0).numpy(), tc_x.norm(2,0).numpy())
        
if __name__ == "__main__":
    unittest.main()
