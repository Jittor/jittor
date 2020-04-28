
# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import jittor.nn as jnn

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    torch = None
    tnn = None

def check_equal(a, b):
    eps = 1e-1 # icc error almost reaches 1e-1
    relative_error = (abs(a - b) / abs(b + 1)).mean()
    print(f"relative_error: {relative_error}")
    return relative_error < eps

class TestRelu(unittest.TestCase):
    def test_relu(self):
        # ***************************************************************
        # Define jittor & pytorch array
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        jittor_arr = jt.array(arr)
        pytorch_arr = torch.Tensor(arr)
        # ***************************************************************
        # Test PReLU Layer
        # ***************************************************************
        pytorch_result = tnn.PReLU(10, 2)(pytorch_arr)
        jittor_result = jnn.PReLU(10, 2)(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.PReLU(10, -0.2)(pytorch_arr)
        jittor_result = jnn.PReLU(10, -0.2)(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.PReLU(10, 99.9)(pytorch_arr)
        jittor_result = jnn.PReLU(10, 99.9)(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        
        # ***************************************************************
        # Test ReLU6 Layer
        # ***************************************************************
        pytorch_result = tnn.ReLU6()(pytorch_arr)
        jittor_result = jnn.ReLU6()(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"

        # ***************************************************************
        # Test LeakyReLU Layer
        # ***************************************************************
        pytorch_result = tnn.LeakyReLU(2)(pytorch_arr)
        jittor_result = jnn.LeakyReLU(2)(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.LeakyReLU()(pytorch_arr)
        jittor_result = jnn.LeakyReLU()(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.LeakyReLU(99.9)(pytorch_arr)
        jittor_result = jnn.LeakyReLU(99.9)(jittor_arr)
        assert check_equal(pytorch_result.detach().numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"

if __name__ == "__main__":
    unittest.main()