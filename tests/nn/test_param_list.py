# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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

    def test_parameter_containers_load_state_dict(self):
        class Net(jt.Module):
            def __init__(self, values):
                self.ps = jt.nn.ParameterList([
                    jt.array(values[0]), jt.array(values[1])])
                self.named = jt.nn.ParameterDict({
                    "scale": jt.array(values[2])})

        source = Net(([1, 2], [3, 4, 5], [6]))
        target = Net(([-1, -2], [-3, -4, -5], [-6]))
        target.load_state_dict(source.state_dict())

        loaded = target.state_dict(to="numpy")
        np.testing.assert_array_equal(loaded["ps.0"], [1, 2])
        np.testing.assert_array_equal(loaded["ps.1"], [3, 4, 5])
        np.testing.assert_array_equal(loaded["named.scale"], [6])

if __name__ == "__main__":
    unittest.main()
