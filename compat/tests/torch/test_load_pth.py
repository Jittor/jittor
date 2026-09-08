# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
from _helpers.logs import find_log_with_re
from _helpers.torch_runtime import import_torch_modules, modules_available
from _helpers.tuner_parser import simple_parser

model_test = os.environ.get("model_test", "") == "1"
_missing = []
if not model_test:
    _missing.append("model_test=1 (this comparison downloads and runs a full ResNet)")
if not modules_available("torch", "torchvision"):
    _missing.append("an independent torch + torchvision")
skip_model_test = bool(_missing)
# The reason has to name what is missing. "Skip model test" is exactly the
# unfalsifiable form 0.18 rejects: a file whose every case skips for a reason
# that names nothing cannot be told apart from a file that quietly stopped
# testing, and this one was the single entry blocking
# JITTOR_TEST_REQUIRE_EXECUTION=1 on the Torch-mode half of the gate.
SKIP_REASON = "needs " + " and ".join(_missing) if _missing else ""
torch = None
tv = None


def setUpModule():
    global torch, tv
    if not skip_model_test:
        torch, tv = import_torch_modules("torch", "torchvision")

@unittest.skipIf(skip_model_test, SKIP_REASON)
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
        assert np.max(np.abs(jt_out.fetch_sync() - torch_out.detach().numpy())) < 1e-3

        pth_name = os.path.join(jt.flags.cache_path, "x.pth")
        torch.save(torch_model.state_dict(), pth_name)
        jt_model.load(pth_name)

        # output
        jt_out = jt_model(jt_img)
        # torch_out = torch_model(torch_img)
        print(np.max(np.abs(jt_out.fetch_sync() - torch_out.detach().numpy())))
        assert np.max(np.abs(jt_out.fetch_sync() - torch_out.detach().numpy())) < 1e-3
    
if __name__ == "__main__":
    unittest.main()
