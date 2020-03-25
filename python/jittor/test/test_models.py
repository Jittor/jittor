# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import numpy as np
import unittest
import jittor as jt
import numpy as np
import jittor.models as jtmodels

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torchvision.models as tcmodels
    from torch import nn
except:
    torch = None
    

skip_this_test = False


@unittest.skipIf(skip_this_test, "skip_this_test")
class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.testmodels = [
            ['squeezenet1_0', 'squeezenet1_0'],
            ['squeezenet1_1', 'squeezenet1_1'],
            ['alexnet', 'alexnet'],
            ['resnet18', 'Resnet18'],
            ['resnet34', 'Resnet34'],
            ['resnet50', 'Resnet50'],
            ['resnet101', 'Resnet101'],
            ['vgg11', 'VGG11'],
            ['vgg11_bn', 'VGG11_bn'],
            ['vgg13', 'VGG13'],
            ['vgg13_bn', 'VGG13_bn'],
            ['vgg16', 'VGG16'],
            ['vgg16_bn', 'VGG16_bn'],
            ['vgg19', 'VGG19'],
            ['vgg19_bn', 'VGG19_bn'],
            ['wide_resnet50_2', 'wide_resnet50_2'],
            ['wide_resnet101_2', 'wide_resnet101_2']
        ]

    @unittest.skipIf(not jt.has_cuda, "Cuda not found")
    @jt.flag_scope(use_cuda=1, use_stat_allocator=1)
    def test_models(self):
        threshold = 1e-3
        # Define numpy input image
        bs = 1
        turns = 100
        testimg = np.random.random((bs,3,224,224)).astype('float32')
        # Define pytorch & jittor input image
        pytorchtestimg = torch.Tensor(testimg).cuda()
        jittortestimg = jt.array(testimg)
        for testmodel in self.testmodels:
            # Define pytorch & jittor model
            pytorchmodel = tcmodels.__dict__[testmodel[0]]().cuda()
            if 'resnet' in testmodel[0]:
                jittormodel = jtmodels.resnet.__dict__[testmodel[1]]()
            elif 'vgg' in testmodel[0]:
                jittormodel = jtmodels.vgg.__dict__[testmodel[1]]()
            elif 'alexnet' in testmodel[0]:
                jittormodel = jtmodels.alexnet.__dict__[testmodel[1]]()
            elif 'squeezenet' in testmodel[0]:
                jittormodel = jtmodels.squeezenet.__dict__[testmodel[1]]()
            # Set eval to avoid dropout layer
            pytorchmodel.eval()
            jittormodel.eval()
            # Jittor loads pytorch parameters to ensure forward alignment
            jittormodel.load_parameters(pytorchmodel.state_dict())
            # Judge pytorch & jittor forward relative error. If the differece is lower than threshold, this test passes.
            pytorchresult = pytorchmodel(pytorchtestimg)
            jittorresult = jittormodel(jittortestimg)
            x = pytorchresult.detach().cpu().numpy() + 1
            y = jittorresult.data + 1
            relative_error = abs(x - y) / abs(y)
            diff = relative_error.mean()
            assert diff < threshold, f"[*] {testmodel[1]} forward fails..., Relative Error: {diff}"
            print(f"[*] {testmodel[1]} forword passes with Relative Error {diff}")
        print('all models pass test.')
        
if __name__ == "__main__":
    unittest.main()
