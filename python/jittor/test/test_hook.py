
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
import jittor.nn as jnn

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
    import torchvision
except:
    torch = None
    tnn = None
    torchvision = None
    skip_this_test = True

@unittest.skipIf(skip_this_test, "No Torch found")
class TestHook(unittest.TestCase):
    def test_bhook(self):
        a = jnn.ReLU()
        hooked = False
        def hook(mod, grad_input, grad_output):
            nonlocal hooked
            hooked = True
            assert len(grad_input) == 1
            assert len(grad_output) == 1
            np.testing.assert_allclose(grad_input[0].numpy(), [0, 1])
            np.testing.assert_allclose(grad_output[0].numpy(), [1, 1])
            return (jt.array([-1.0, -2.0]), )
        a.register_backward_hook(hook)
        x = jt.array([-1.0,2])
        y = a(x)
        dx = jt.grad(y, x)
        assert hooked
        np.testing.assert_allclose(dx.numpy(), [-1.0, -2.0])

    def test_register_hook(self):
        x = jt.array([0.0, 0.0])
        y = x * [1,2]
        y.register_hook(lambda g: g*2)
        dx = jt.grad(y, x)
        np.testing.assert_allclose(dx.data, [2,4])

    def test_requires_grads_(self):
        class Mod(jt.nn.Module):
            def execute(self, x):
                return x*2
        x = jt.random((100,))
        mod = Mod()
        mod.requires_grad_(True)
        y = mod(x)
        y = y*10
        dx = jt.grad(y, x)
        np.testing.assert_allclose(dx.data, 20)

        mod.requires_grad_(False)
        y = mod(x)
        y = y*10
        dx = jt.grad(y, x)
        np.testing.assert_allclose(dx.data, 0)

if __name__ == "__main__":
    unittest.main()