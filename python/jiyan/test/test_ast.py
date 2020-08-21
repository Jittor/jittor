# ***************************************************************
# Copyright (c) 2020 Jittor. All Rights Reserved.
# Authors:
#   Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jiyan.ast as ja
from astunparse import unparse

class TestAST(unittest.TestCase):
    def test_parse_py(self):
        def test2(a, b):
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    a[i,j] = 0
                    a[i,j] += b[j,i]
                    a[i,j] += b[j,i]
                    aa = a[i,j]
                    if aa:
                        aa = 1
                        bb = 2.0
                    a[i,j] += aa + bb
        pir = ja.get_py_ast_from_func(test2)
        jir = ja.AST.from_func(test2)
        pir2 = jir.to_py_ast()
        assert unparse(pir) == unparse(pir2), (unparse(pir) + unparse(pir2))

if __name__ == "__main__":
    unittest.main()
