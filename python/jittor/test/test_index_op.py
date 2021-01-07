# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestIndexOp(unittest.TestCase):
    def test(self):
        assert (jt.index([2,2], 0).data==[[0,0],[1,1]]).all()
        assert (jt.index([2,2], 1).data==[[0,1],[0,1]]).all()
        a = jt.index([2,2], 0)
        b = jt.index([2,2], 1)
        c = a+b
        assert (c.data==[[0,1],[1,2]]).all(), c.data

    def test_multioutput(self):
        a,b = jt.index([2,2])
        jt.sync([a,b])
        assert (a.data==[[0,0],[1,1]]).all()
        assert (b.data==[[0,1],[0,1]]).all(), b.data

    def test_multioutput2(self):
        a,b = jt.index([3,3])
        assert (a.data==[[0,0,0],[1,1,1],[2,2,2]]).all()
        assert (b.data==[[0,1,2],[0,1,2],[0,1,2]]).all(), b.data
        a,b = jt.index([3,3])
        c = a+b
        assert (c.data==[[0,1,2],[1,2,3],[2,3,4]]).all(), c.data

    def test_multioutput3(self):
        a,b = jt.index([3,3])
        del a
        assert (b.data==[[0,1,2],[0,1,2],[0,1,2]]).all(), b.data

    def test_vary_shape_dep(self):
        a, = jt.where([1,0,1])
        b, = a.index_var()
        assert a.uncertain_shape==[-3] and b.uncertain_shape==[-3]
        assert (b.data==[0,1]).all()
        
    def test_vary_shape_dep2(self):
        a = jt.array([[1,2,3],[4,5,6],[7,8,9]])
        index0, = jt.where(a.sum(1)>7) # [1,2]
        index0 = index0.broadcast([1,3], dims=[1]) # [[1,1,1],[2,2,2]]
        index1 = index0.index_var(1) # [[0,1,2],[0,1,2]]
        b = a.reindex_var([index0, index1])
        assert b.uncertain_shape==[-3,3]
        assert (b.data==[[4,5,6],[7,8,9]]).all()
        assert (index0.data==[[1,1,1],[2,2,2]]).all()
        assert (index1.data==[[0,1,2],[0,1,2]]).all()

    def test_doc(self):
        assert "Index Operator" in jt.index.__doc__

if __name__ == "__main__":
    unittest.main()