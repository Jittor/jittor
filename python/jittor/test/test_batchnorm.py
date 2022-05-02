
# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
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

def check_equal_with_istrain(arr, j_layer, p_layer, is_train=True, has_running=True, threshold=1e-5):
    jittor_arr = jt.array(arr)
    pytorch_arr = torch.Tensor(arr)
    if has_running:
        if is_train:
            assert np.allclose(p_layer.running_mean.detach().numpy(), j_layer.running_mean.numpy(), threshold)
        else:
            assert np.allclose(p_layer.layer.running_mean.detach().numpy(), j_layer.running_mean.numpy(), threshold)
    jittor_result = j_layer(jittor_arr)
    pytorch_result = p_layer(pytorch_arr)
    if has_running:
        if is_train:
            assert np.allclose(p_layer.running_mean.detach().numpy(), j_layer.running_mean.numpy(), threshold)
        else:
            assert np.allclose(p_layer.layer.running_mean.detach().numpy(), j_layer.running_mean.numpy(), threshold)
    assert np.allclose(pytorch_result.detach().numpy(), jittor_result.numpy(), 1e-2, threshold), \
        ( np.abs(pytorch_result.detach().numpy() - jittor_result.numpy()).max() )

def check_equal_without_istrain(arr, j_layer, p_layer, threshold=1e-5):
    jittor_arr = jt.array(arr)
    pytorch_arr = torch.Tensor(arr)
    jittor_result = j_layer(jittor_arr)
    pytorch_result = p_layer(pytorch_arr)
    assert np.allclose(pytorch_result.detach().numpy(), jittor_result.numpy(), threshold)

@unittest.skipIf(skip_this_test, "No Torch found")
class TestBatchNorm(unittest.TestCase):
    @jt.flag_scope(auto_convert_64_to_32=0)
    def test_batchnorm(self):
        # ***************************************************************
        # Test BatchNorm Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal_with_istrain(arr, jnn.BatchNorm(10, is_train=True), tnn.BatchNorm2d(10))

        class Model(tnn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = tnn.BatchNorm2d(10)
            def forward(self, x):
                return self.layer(x)
        model = Model()
        model.eval()
        check_equal_with_istrain(arr, jnn.BatchNorm(10, is_train=False), model, False)

        # ***************************************************************
        # Test InstanceNorm2d Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)
        check_equal_without_istrain(arr, jnn.InstanceNorm2d(10, is_train=True), tnn.InstanceNorm2d(10))

        class Model(tnn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = tnn.InstanceNorm2d(10)
            def forward(self, x):
                return self.layer(x)
        model = Model()
        model.eval()
        check_equal_without_istrain(arr, jnn.InstanceNorm2d(10, is_train=False), model)

        # ***************************************************************
        # Test BatchNorm1d Layer
        # ***************************************************************
        arr = np.random.randn(16,10)
        check_equal_with_istrain(arr, jnn.BatchNorm1d(10, is_train=True), tnn.BatchNorm1d(10), 1e-3)

        class Model(tnn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = tnn.BatchNorm1d(10)
            def forward(self, x):
                return self.layer(x)
        model = Model()
        model.eval()
        check_equal_with_istrain(arr, jnn.BatchNorm1d(10, is_train=False), model, False)

        # ***************************************************************
        # Test GroupNorm Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)

        class Model(tnn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = tnn.GroupNorm(2, 10)
            def forward(self, x):
                return self.layer(x)
        model = Model()
        model.eval()
        check_equal_with_istrain(arr, jnn.GroupNorm(2, 10, is_train=False), model, False, False)

        # ***************************************************************
        # Test LayerNorm Layer
        # ***************************************************************
        arr = np.random.randn(16,10,224,224)

        class Model(tnn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = tnn.LayerNorm(224)
            def forward(self, x):
                return self.layer(x)
        model = Model()
        model.eval()
        check_equal_with_istrain(arr, jnn.LayerNorm(224), model, False, False)

if __name__ == "__main__":
    unittest.main()