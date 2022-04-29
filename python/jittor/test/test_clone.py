# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestClone(unittest.TestCase):
    def test_mid_stop_grad(self):
        jt.clean()
        b = a = jt.array(1.0)
        for i in range(10):
            b = b.clone()
            if i==5: c=b
        b.sync()
        assert jt.number_of_lived_vars()==11
        c.name("c")
        c.stop_grad()
        for n in jt.dump_all_graphs().nodes_info:
            print(n)
        assert jt.number_of_lived_vars()==3, jt.number_of_lived_vars()

    def test2(self):
        a = jt.array([1,2])
        print(a.detach())

    @jt.flag_scope(lazy_execution=0)
    def test3(self):
        a = jt.array([1,2])
        print(a.detach())

if __name__ == "__main__":
    unittest.main()