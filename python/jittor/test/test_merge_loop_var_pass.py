# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as numpy

class TestMergeLoopVarPass(unittest.TestCase):
    def test(self):
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.sum([2,3])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range01" in src
            assert "range23" in src

    def test2(self):
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a + 1
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range0123" in src

    def test3(self):
        a = jt.ones([10,10,10,10])
        x = jt.ones([1,10,1,1])
        a.sync(), x.sync()
        with jt.profile_scope() as rep:
            b = a + x
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range23" in src

    def test4(self):
        # don't optimize reindex like op yet
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.reindex_reduce("add", [10,10], ["i0","i1"])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range23" not in src

    def test5(self):
        a = jt.ones([10,10,10,10])
        a.sync()
        with jt.profile_scope() as rep:
            b = a.sum([1])
            b.sync()
        with open(rep[1][1]) as f:
            src = f.read()
            assert "range01" not in src
            assert "range23" in src

@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestMergeLoopVarPassCuda(TestMergeLoopVarPass):
    def setUp(self):
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

if __name__ == "__main__":
    unittest.main()
