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
import jittor.models as jtmodels

skip_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torchvision.models as tcmodels
    from torch import nn
except:
    torch = None
    skip_this_test = True

@unittest.skipIf(skip_this_test, "skip_this_test")
class test_models(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.models = [
            'squeezenet1_0',
            'squeezenet1_1',
            'alexnet',
            'resnet18',
            'resnet34',
            'resnet50',
            'resnet101',
            'resnet152',
            'resnext50_32x4d',
            'resnext101_32x8d',
            'vgg11',
            'vgg11_bn',
            'vgg13',
            'vgg13_bn',
            'vgg16',
            'vgg16_bn',
            'vgg19',
            'vgg19_bn',
            'wide_resnet50_2',
            'wide_resnet101_2',
            'googlenet',
            'mobilenet_v2',
            'mnasnet0_5',
            'mnasnet0_75',
            'mnasnet1_0',
            'mnasnet1_3',
            'shufflenet_v2_x0_5',
            'shufflenet_v2_x1_0',
            'shufflenet_v2_x1_5',
            'shufflenet_v2_x2_0',
            "densenet121",
            "densenet161",
            "densenet169",
            'inception_v3',
        ]

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1)
    def test_models(self):
        with torch.no_grad():
            self.run_models()

    def run_models(self):
        def to_cuda(x):
            if jt.has_cuda:
                return x.cuda()
            return x
        threshold = 1e-2
        # Define numpy input image
        bs = 1
        test_img = np.random.random((bs,3,224,224)).astype('float32')
        # test_img = np.random.random((bs,3,280,280)).astype('float32')
        # Define pytorch & jittor input image
        pytorch_test_img = to_cuda(torch.Tensor(test_img))
        jittor_test_img = jt.array(test_img)
        for test_model in self.models:
            if test_model == "inception_v3":
                test_img = np.random.random((bs,3,300,300)).astype('float32')
                pytorch_test_img = to_cuda(torch.Tensor(test_img))
                jittor_test_img = jt.array(test_img)
            # Define pytorch & jittor model
            pytorch_model = to_cuda(tcmodels.__dict__[test_model]())
            jittor_model = jtmodels.__dict__[test_model]()
            # Set eval to avoid dropout layer
            pytorch_model.eval()
            jittor_model.eval()
            # Jittor loads pytorch parameters to ensure forward alignment
            jittor_model.load_parameters(pytorch_model.state_dict())
            # Judge pytorch & jittor forward relative error. If the differece is lower than threshold, this test passes.
            pytorch_result = pytorch_model(pytorch_test_img)
            jittor_result = jittor_model(jittor_test_img)
            x = pytorch_result.detach().cpu().numpy() + 1
            y = jittor_result.data + 1
            relative_error = abs(x - y) / abs(y)
            diff = relative_error.mean()
            assert diff < threshold, f"[*] {test_model} forward fails..., Relative Error: {diff}"
            print(f"[*] {test_model} forword passes with Relative Error {diff}")
            jt.clean()
            jt.gc()
            torch.cuda.empty_cache()
        print('all models pass test.')
        
if __name__ == "__main__":
    unittest.main()
