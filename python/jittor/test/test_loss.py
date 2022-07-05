# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
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
        
    def test_nll_loss(self):
        tc_loss = tnn.functional.nll_loss
        jt_loss = jnn.nll_loss
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        jt_y=jt_loss(jt.array(output), jt.array(target),reduction='mean')
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target),reduction='mean')
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        weight=np.random.randn(10,).astype(np.float32)
        jt_y=jt_loss(jt.array(output), jt.array(target),jt.array(weight),reduction='mean')
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target),torch.from_numpy(weight),reduction='mean')
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_loss(self):
        jt_loss=jnn.CrossEntropyLoss()
        tc_loss=tnn.CrossEntropyLoss()
        output=np.random.randn(10,10).astype(np.float32)
        target=np.random.randint(10, size=(10))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())
    
    def test_cross_entropy_loss_v2(self):
        B = 100
        C = 5
        for shape in [[100,1],[],[100,20]]:
            s1 = [B,C]+shape
            s2 = [B]+shape
            a = np.random.randn(*s1).astype(np.float32)
            b = np.random.randint(0,C,size=s2).astype(np.int32)
            weight = np.random.randn(C).astype(np.float32)

            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight),reduction=r)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight),reduction=r)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),reduction=r)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),reduction=r)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)))
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b))
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)

            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight))
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight))
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)

            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight),reduction=r,ignore_index=C//2)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight),reduction=r,ignore_index=C//2)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            for r in ['mean','sum','none']:
                r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),reduction=r,ignore_index=C//2)
                r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),reduction=r,ignore_index=C//2)
                np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)
            
            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),ignore_index=C//2)
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),ignore_index=C//2)
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)

            r1 = torch.nn.functional.cross_entropy(torch.tensor(a),torch.tensor(b.astype(np.int64)),weight=torch.tensor(weight),ignore_index=C//2)
            r2 = jnn.cross_entropy_loss(jt.array(a),jt.array(b),weight=jt.array(weight),ignore_index=C//2)
            np.testing.assert_allclose(r1.numpy(),r2.numpy(),rtol=1e-3, atol=1e-3)


    def test_cross_entropy_ignore_index(self):
        ignore_index = np.random.randint(0, 10)
        jt_loss = jnn.CrossEntropyLoss(ignore_index=ignore_index)
        tc_loss = tnn.CrossEntropyLoss(ignore_index=ignore_index)
        output = np.random.rand(100, 10).astype(np.float32)
        target = np.random.randint(10, size=(100))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_weight(self):
        weight = np.random.rand(10).astype('float32')
        jt_loss = jnn.CrossEntropyLoss(weight=jt.array(weight))
        tc_loss = tnn.CrossEntropyLoss(weight=torch.from_numpy(weight))
        output = np.random.rand(100, 10).astype(np.float32)
        target = np.random.randint(10, size=(100))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        assert np.allclose(jt_y.numpy(), tc_y.numpy())

    def test_cross_entropy_weight_ignore(self):
        weight = np.random.rand(4).astype('float32')
        jt_loss = jnn.CrossEntropyLoss(weight=jt.array(weight), ignore_index=1)
        tc_loss = tnn.CrossEntropyLoss(weight=torch.from_numpy(weight), ignore_index=1)
        output = np.random.rand(3, 4, 2,2).astype(np.float32)
        target = np.random.randint(4, size=(3, 2,2))
        jt_y=jt_loss(jt.array(output), jt.array(target))
        tc_y=tc_loss(torch.from_numpy(output), torch.from_numpy(target))
        np.testing.assert_allclose(jt_y.numpy(), tc_y.numpy())


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
