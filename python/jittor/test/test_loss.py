# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
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
class TestLoss(unittest.TestCase):
    def test_l1_loss(self):
        jt_loss=jnn.L1Loss()
        tc_loss=tnn.L1Loss()
        output=np.random.randn(10,100).astype(np.float32)
        target=np.random.randn(10,100).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
    def test_mse_loss(self):
        jt_loss=jnn.MSELoss()
        tc_loss=tnn.MSELoss()
        output=np.random.randn(10,100).astype(np.float32)
        target=np.random.randn(10,100).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_loss(self):
        jt_loss=jnn.CrossEntropyLoss()
        tc_loss=tnn.CrossEntropyLoss()
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_bce_loss(self):
        jt_loss=jnn.BCELoss()
        tc_loss=tnn.BCELoss()
        jt_sig = jnn.Sigmoid()
        tc_sig = tnn.Sigmoid()
        output=np.random.randn(100).astype(np.float32)
        target=np.random.randint(2, size=(100)).astype(np.float32)
        jt_y=jt_loss(jt_sig(jt.array(output)), jt.array(target))
        tc_y=tc_loss(tc_sig(torch.from_numpy(output)), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

        weight=np.random.randn(100).astype(np.float32)
        jt_loss=jnn.BCELoss(weight=jt.array(weight), size_average=False)
        tc_loss=tnn.BCELoss(weight=torch.Tensor(weight), size_average=False)
        jt_y=jt_loss(jt_sig(jt.array(output)), jt.array(target))
        tc_y=tc_loss(tc_sig(torch.from_numpy(output)), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
    def test_bce_with_logits_loss(self):
        jt_loss=jnn.BCEWithLogitsLoss()
        tc_loss=tnn.BCEWithLogitsLoss()
        output=np.random.randn(100).astype(np.float32)
        target=np.random.randint(2, size=(100)).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        
if __name__ == "__main__":
    unittest.main()
