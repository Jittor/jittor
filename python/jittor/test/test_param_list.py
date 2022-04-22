# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np



class TestParamList(unittest.TestCase):
    def test_param_list(self):
        ps = jt.nn.ParameterList([jt.array([1,2,3]), jt.rand(10)])
        assert len(ps.parameters()) == 2
        assert list(ps.state_dict().keys()) == ['0', '1'], ps.state_dict().keys()

    def test_with_module(self):
        class Net(jt.nn.Module):
            def __init__(self):
                self.ps1 = jt.nn.ParameterList([jt.array([1,2,3]), jt.rand(10)])
                self.ps2 = jt.nn.ParameterDict({
                    "aaa":jt.array([1,2,3]),
                    "bbb": jt.rand(10)
                })
        net = Net()
        assert list(net.state_dict().keys()) == ['ps1.0', 'ps1.1', 'ps2.aaa', 'ps2.bbb']

if __name__ == "__main__":
    unittest.main()