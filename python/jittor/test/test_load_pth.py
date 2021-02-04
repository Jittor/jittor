# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn
from jittor.models import resnet
import numpy as np
import sys, os
import random
import math
import unittest
from .test_reorder_tuner import simple_parser
from .test_log import find_log_with_re

model_test = os.environ.get("model_test", "") == "1"
skip_model_test = not model_test

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torchvision as tv
except:
    skip_model_test = True

@unittest.skipIf(skip_model_test, "Skip model test")
class TestLoadPth(unittest.TestCase):
    def test_load_pth(self):
        # TODO: load torch model params
        # define input img
        img = np.random.random((1,3,224,224)).astype("float32")
        jt_img = jt.array(img)
        torch_img = torch.Tensor(img)
        # define pytorch and jittor pretrained model
        torch_model = tv.models.resnet18(True)

        jt_model = resnet.Resnet18()
        jt_model.load_parameters(torch_model.state_dict())
        # todo: model.train() model.eval()

        # output
        jt_out = jt_model(jt_img)
        torch_out = torch_model(torch_img)
        print(np.max(np.abs(jt_out.fetch_sync() - torch_out.detach().numpy())))
        assert np.max(np.abs(jt_out.fetch_sync() - torch_out.detach().numpy())) < 1e-4
        
if __name__ == "__main__":
    unittest.main()
