# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_grad import ngrad
from .test_cuda import test_cuda

def get_node_info(s):
    mem_ptr = s.split(')')[0].split(',')[-1]
    name = s.split(')')[0].split(',')[-2]
    return name, mem_ptr

def get_info(graph):
    bop = [ node for node in graph.nodes_info if node.startswith("Var")]
    node_dict = {}
    for bop_ in bop:
        name, mem_ptr = get_node_info(bop_)
        node_dict[name] = mem_ptr
    return node_dict

def check_equal(a, b):
    eps = 1e-1 # icc error almost reaches 1e-1
    return abs(a - b) < eps

class TestReshapeOp(unittest.TestCase):
    def test_reshape(self):
        a = jt.random([123, 456, 789]).name("a")
        b = jt.reshape(a, [123 * 2, int(789 * 456 / 2)]).name("b")
        c = jt.reshape(b, [123 * 456 * 789]).name("c")
        d = jt.reshape(c, [2, int(123 / 3), 789, int(456 / 2), 3]).name("d")
        e = jt.reshape(d, [2, int(123 / 3), 789, -1, 3]).name("e")
        assert b.shape == [123 * 2, int(789 * 456 / 2)]
        assert c.shape == [123 * 456 * 789]
        assert d.shape == [2, int(123 / 3), 789, int(456 / 2), 3]
        assert e.shape == [2, int(123 / 3), 789, int(456 / 2), 3]
        a_mean = a.mean().data
        b_mean = b.mean().data
        c_mean = c.mean().data
        d_mean = d.mean().data
        e_mean = e.mean().data
        a = (a + 1).name("new_a")
        new_a_mean = a.mean().data
        new_b_mean = b.mean().data
        node_dict = get_info(jt.dump_all_graphs())
        assert check_equal(a_mean, b_mean), f"{a_mean} != {b_mean}"
        assert check_equal(a_mean, c_mean), f"{a_mean} != {c_mean}"
        assert check_equal(a_mean, d_mean), f"{a_mean} != {d_mean}"
        assert check_equal(a_mean, e_mean), f"{a_mean} != {e_mean}"
        assert check_equal(b_mean, new_b_mean), f"{b_mean} != {new_b_mean}"
        assert not check_equal(a_mean, new_a_mean), f"{a_mean} == {new_a_mean}"
        assert node_dict['a'] == node_dict['b']
        assert node_dict['a'] == node_dict['c']
        assert node_dict['a'] == node_dict['d']
        assert node_dict['a'] == node_dict['e']

    def test_view(self):
        a = jt.ones([2,3,4])
        assert a.view(2,-1).shape == [2,12]

    def test_flatten(self):
        a = jt.ones([2,3,4])
        assert a.flatten().shape == [24]
        assert a.flatten(1).shape == [2,12]
        assert a.flatten(0,-2).shape == [6,4]

    def test_reshape_var(self):
        a = jt.zeros(10)
        b = a.reshape(a.shape)


if __name__ == "__main__":
    unittest.main()