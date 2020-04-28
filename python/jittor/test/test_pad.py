
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

class TestPad(unittest.TestCase):
    def test_pad(self):
        # ***************************************************************
        # Define jittor & pytorch array
        # ***************************************************************
        arr = np.random.randn(16,3,224,224)
        jittor_arr = jt.array(arr)
        pytorch_arr = torch.Tensor(arr)
        # ***************************************************************
        # Test ReplicationPad2d Layer
        # ***************************************************************
        pytorch_result = tnn.ReplicationPad2d(10)(pytorch_arr)
        jittor_result = jnn.ReplicationPad2d(10)(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.ReplicationPad2d((1,23,4,5))(pytorch_arr)
        jittor_result = jnn.ReplicationPad2d((1,23,4,5))(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        # ***************************************************************
        # Test ConstantPad2d Layer
        # ***************************************************************
        pytorch_result = tnn.ConstantPad2d(10,-2)(pytorch_arr)
        jittor_result = jnn.ConstantPad2d(10,-2)(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.ConstantPad2d((2,3,34,1),10.2)(pytorch_arr)
        jittor_result = jnn.ConstantPad2d((2,3,34,1),10.2)(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        # ***************************************************************
        # Test ZeroPad2d Layer
        # ***************************************************************
        pytorch_result = tnn.ZeroPad2d(1)(pytorch_arr)
        jittor_result = jnn.ZeroPad2d(1)(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.ZeroPad2d((2,3,34,1))(pytorch_arr)
        jittor_result = jnn.ZeroPad2d((2,3,34,1))(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        # ***************************************************************
        # Test ReflectionPad2d Layer
        # ***************************************************************
        pytorch_result = tnn.ReflectionPad2d(20)(pytorch_arr)
        jittor_result = jnn.ReflectionPad2d(20)(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"
        pytorch_result = tnn.ReflectionPad2d((2,3,34,1))(pytorch_arr)
        jittor_result = jnn.ReflectionPad2d((2,3,34,1))(jittor_arr)
        assert check_equal(pytorch_result.numpy(), jittor_result.numpy()), f"{pytorch_result.mean()} || {jittor_result.mean()}"

if __name__ == "__main__":
    unittest.main()