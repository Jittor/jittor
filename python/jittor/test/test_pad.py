
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
class TestPad(unittest.TestCase):
    def test_pad(self):
        # ***************************************************************
        # Test ReplicationPad2d Layer
        # ***************************************************************
        arr = np.random.randn(16,3,224,224)
        check_equal(arr, jnn.ReplicationPad2d(10), tnn.ReplicationPad2d(10))
        check_equal(arr, jnn.ReplicationPad2d((1,23,4,5)), tnn.ReplicationPad2d((1,23,4,5)))
        check_equal(arr, jnn.ReplicationPad2d((1,0,1,5)), tnn.ReplicationPad2d((1,0,1,5)))
        check_equal(arr, jnn.ReplicationPad2d((100)), tnn.ReplicationPad2d((100)))

        # ***************************************************************
        # Test ConstantPad2d Layer
        # ***************************************************************
        arr = np.random.randn(16,3,224,224)
        check_equal(arr, jnn.ConstantPad2d(10,-2), tnn.ConstantPad2d(10,-2))
        check_equal(arr, jnn.ConstantPad2d((2,3,34,1),10.2), tnn.ConstantPad2d((2,3,34,1),10.2))

        arr = np.random.randn(16,3,224,10,10)
        check_equal(arr, jnn.ConstantPad2d(10,-2), tnn.ConstantPad2d(10,-2))
        check_equal(arr, jnn.ConstantPad2d((2,3,34,1),10.2), tnn.ConstantPad2d((2,3,34,1),10.2))

        # ***************************************************************
        # Test ZeroPad2d Layer
        # ***************************************************************
        arr = np.random.randn(16,3,224,224)
        check_equal(arr, jnn.ZeroPad2d(1), tnn.ZeroPad2d(1))
        check_equal(arr, jnn.ZeroPad2d((2,3,34,1)), tnn.ZeroPad2d((2,3,34,1)))

        # ***************************************************************
        # Test ReflectionPad2d Layer
        # ***************************************************************
        arr = np.random.randn(16,3,224,224)
        check_equal(arr, jnn.ReflectionPad2d(20), tnn.ReflectionPad2d(20))
        check_equal(arr, jnn.ReflectionPad2d((2,3,34,1)), tnn.ReflectionPad2d((2,3,34,1)))
        check_equal(arr, jnn.ReflectionPad2d((10,123,34,1)), tnn.ReflectionPad2d((10,123,34,1)))
        check_equal(arr, jnn.ReflectionPad2d((100)), tnn.ReflectionPad2d((100)))

        # ***************************************************************
        # Test function pad
        # ***************************************************************
        arr = np.random.randn(16,3,224,224)
        padding = (10,11,2,3)
        for mode in ['constant','replicate','reflect','circular']:
            j_data = jt.array(arr)
            t_data = torch.tensor(arr)
            t_output = tnn.functional.pad(t_data,padding,mode=mode).detach().numpy()
            j_output = jnn.pad(j_data,padding,mode).numpy()
            assert np.allclose(t_output,j_output)


if __name__ == "__main__":
    unittest.main()