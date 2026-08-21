# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn
import numpy as np
import unittest

from _helpers.torch_runtime import import_torch_modules, modules_available


torch = None
has_torch = modules_available("torch")


def setUpModule():
    global torch
    if has_torch:
        (torch,) = import_torch_modules("torch")


@unittest.skipIf(not has_torch, "No independent Torch found.")
class TestInterpolation(unittest.TestCase):
    def test_interpolation_area(self):
        img = np.random.uniform(0, 1, (1, 3, 24, 10))
        output_shape = (12, 5)
        jimg = jt.array(img)
        timg = torch.from_numpy(img)
        joutput = nn.interpolate(jimg, output_shape, mode="area")
        toutput = torch.nn.functional.interpolate(timg, output_shape, mode="area")
        np.testing.assert_allclose(joutput.numpy(), toutput.numpy(), rtol=1e-7, atol=1e-7)

if __name__ == "__main__":
    unittest.main()
