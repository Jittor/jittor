# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
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
        # Counted from what is already alive: in a whole-tree run earlier
        # files keep Vars of their own (module globals, caches), and an
        # absolute count failed here for reasons that had nothing to do with
        # clone.
        base = jt.introspection.counters.live_vars
        b = a = jt.array(1.0)
        for i in range(10):
            b = b.clone()
            if i==5: c=b
        b.sync()
        assert jt.introspection.counters.live_vars - base == 11
        c.name("c")
        c.stop_grad()
        for n in jt.dump_all_graphs().nodes_info:
            print(n)
        live = jt.introspection.counters.live_vars - base
        assert live == 3, live

    def test2(self):
        a = jt.array([1,2])
        print(a.detach())

    @jt.flag_scope(lazy_execution=0)
    def test3(self):
        a = jt.array([1,2])
        print(a.detach())

if __name__ == "__main__":
    unittest.main()