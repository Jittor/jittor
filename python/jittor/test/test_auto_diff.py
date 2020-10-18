# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import numpy as np
import os
import sys
import jittor as jt

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
class TestAutoDiff(unittest.TestCase):
    def test_pt_hook(self):
        code = '''
import numpy as np
from jittor_utils import auto_diff
import torch
import torchvision.models as tcmodels
net = tcmodels.resnet50()
net.train()
hook = auto_diff.Hook("resnet50")
hook.hook_module(net)

np.random.seed(0)
data = np.random.random((2,3,224,224)).astype('float32')
data = torch.Tensor(data)
net(data)
# assert auto_diff.has_error == 0, auto_diff.has_error
'''
        with open("/tmp/test_pt_hook.py", 'w') as f:
            f.write(code)
        assert os.system(sys.executable+" /tmp/test_pt_hook.py") == 0
        assert os.system(sys.executable+" /tmp/test_pt_hook.py") == 0
        code = '''
import numpy as np
import jittor as jt
from jittor_utils import auto_diff
from jittor.models import resnet50
net = resnet50()
net.train()
hook = auto_diff.Hook("resnet50")
hook.hook_module(net)

np.random.seed(0)
data = np.random.random((2,3,224,224)).astype('float32')
data = jt.array(data)
net(data)
# assert auto_diff.has_error == 0, auto_diff.has_error
'''
        with open("/tmp/test_jt_hook.py", 'w') as f:
            f.write(code)
        assert os.system(sys.executable+" /tmp/test_jt_hook.py") == 0
        

if __name__ == "__main__":
    unittest.main()
