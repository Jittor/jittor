# ***************************************************************
# Copyright (c) Jittor 2020, Author:
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
import numpy as np
from jittor import models

pass_this_test = False
try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    import torchvision
except Exception as e:
    pass_this_test = True

def get_error(a, b):
    return np.abs(a-b) / max(np.abs(a), np.abs(b), 1e-5) , np.abs(a-b)

def check(jt_mod, torch_mod, rtol=1e-2, atol=1e-5, mean_atol=1e-5):
    pa = [ p for p in jt_mod.parameters() if not p.is_stop_grad() ]
    pb = list(torch_mod.parameters())
    assert len(pa) == len(pb)
    error_count = 0
    for a,b in zip(pa, pb):
        assert a.shape == list(b.shape), (a.shape, b.shape, a.name())
        stda, meana = np.std(a.numpy()), np.mean(a.numpy())
        stdb, meanb = np.std(b.detach().numpy()), np.mean(b.detach().numpy())

        r_err, a_err = get_error(stda, stdb)
        if r_err > rtol and a_err > atol:
            error_count += 1
            print("compare std error", stda, stdb, r_err, a_err, a.name(), a.shape)

        r_err, a_err = get_error(meana, meanb)
        if r_err > rtol and a_err > mean_atol:
            error_count += 1
            print("compare mean error", meana, meanb, r_err, a_err, a.name(), a.shape)
    assert error_count == 0

@unittest.skipIf(pass_this_test, f"pass init check, no torch found")
class TestInit(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        jt.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def test_conv(self):
        check(jt.nn.Conv(64, 256, 3), torch.nn.Conv2d(64, 256, 3), rtol=1e-1, mean_atol=1e-2)

    def test_resnet(self):
        check(models.resnet152(), torchvision.models.resnet152(), rtol=5e-2, mean_atol=1e-2)

if __name__ == "__main__":
    unittest.main()