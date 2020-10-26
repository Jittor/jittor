
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

skip_this_test = False

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torch.nn as tnn
except:
    torch = None
    tnn = None
    skip_this_test = True

def check_equal(arr, j_layer, p_layer):
    jittor_arr = jt.array(arr)
    pytorch_arr = torch.Tensor(arr)
    jittor_result = j_layer(jittor_arr)
    pytorch_result = p_layer(pytorch_arr)
    assert np.allclose(pytorch_result.detach().numpy(), jittor_result.numpy())

@unittest.skipIf(skip_this_test, "No Torch found")
class TestRelu(unittest.TestCase):
    def test_relu(self):
        # ***************************************************************
        # Test ReLU Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal(arr, jnn.ReLU(), tnn.ReLU())

        # ***************************************************************
        # Test PReLU Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal(arr, jnn.PReLU(), tnn.PReLU())
        check_equal(arr, jnn.PReLU(10, 99.9), tnn.PReLU(10, 99.9))
        check_equal(arr, jnn.PReLU(10, 2), tnn.PReLU(10, 2))
        check_equal(arr, jnn.PReLU(10, -0.2), tnn.PReLU(10, -0.2))
        
        # ***************************************************************
        # Test ReLU6 Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal(arr, jnn.ReLU6(), tnn.ReLU6())

        # ***************************************************************
        # Test LeakyReLU Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal(arr, jnn.LeakyReLU(), tnn.LeakyReLU())
        check_equal(arr, jnn.LeakyReLU(2), tnn.LeakyReLU(2))
        check_equal(arr, jnn.LeakyReLU(99.9), tnn.LeakyReLU(99.9))

        # ***************************************************************
        # Test Softplus  Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal(arr, jnn.Softplus (), tnn.Softplus ())
        check_equal(arr, jnn.Softplus (2), tnn.Softplus (2))
        check_equal(arr, jnn.Softplus (2, 99.9), tnn.Softplus (2, 99.9))
        
if __name__ == "__main__":
    unittest.main()