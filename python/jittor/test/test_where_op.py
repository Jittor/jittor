# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np

class TestWhereOp(unittest.TestCase):
    def test(self):
        assert (jt.where([0,1,0,1])[0].data == [1,3]).all()
        a, = jt.where([0,1,0,1])
        assert a.uncertain_shape==[-4]
        a.data
        assert a.uncertain_shape==[2]
        a,b = jt.where([[0,0,1],[1,0,0]])
        assert (a.data==[0,1]).all() and (b.data==[2,0]).all()

    def test_reindex_dep(self):
        a = jt.random([10])
        b, = (a>1).where()
        assert len(b.data)==0
        b, = (a>0.5).where()
        assert (b.data==np.where(a.data>0.5)).all()
        b = a.reindex_var((a>0.5).where())
        assert (b.data==a.data[a.data>0.5]).all()

    def test_binary_dep(self):
        a = jt.random([10])
        b, = (a>0.5).where()
        b = b+1
        assert (b.data==np.where(a.data>0.5)[0]+1).all()
        b, = (a>1).where()
        b = b+1
        assert (b.data==np.where(a.data>1)[0]+1).all()

    def test_self_dep(self):
        a = jt.random([100])
        x = a.reindex_var((a>0.1).where())
        x = x.reindex_var((x<0.9).where())
        na = a.data
        assert (na[np.logical_and(na>0.1, na<0.9)]==x.data).all()

    def test_reduce_dep(self):
        a = jt.random([100,100])
        index = (a>0.5).where()
        x = a.reindex_var(index)
        xsum =x.sum()
        na = a.data
        assert np.allclose(np.sum(na[na>0.5]),xsum.data), (x.data, xsum.data, np.sum(na[na>0.5]))
        
    def test_doc(self):
        assert "Where Operator" in jt.where.__doc__

if __name__ == "__main__":
    unittest.main()