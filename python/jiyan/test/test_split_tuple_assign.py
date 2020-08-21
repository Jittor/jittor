# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
from jiyan.passes.split_tuple_assign import split_tuple_assign
import jiyan.ast as ja

class TestSplitTupleAssign(unittest.TestCase):
    def test(self):
        def test(shape):
            n,c,h,w = shape
            a,b = 1,2
            b,a = a,b
        jir = ja.AST.from_func(test)
        split_tuple_assign(jir)
        code='''def test(shape):
    n = shape[0]
    c = shape[1]
    h = shape[2]
    w = shape[3]
    a = 1
    b = 2
    _b = b
    b = a
    a = _b'''
        assert code in str(jir), jir

if __name__ == "__main__":
    unittest.main()
