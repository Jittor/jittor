# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import time
from .test_core import expect_error

class TestNanoVector(unittest.TestCase):
    def test(self):
        nvector = jt.NanoVector
        nv = nvector()
        nv.append(1)
        nv.append(2)
        nv.append(3)
        nv.append(1<<40)
        assert nv[3] == (1<<40)
        assert str(nv) == "[1,2,3,1099511627776,]"
        assert nv == [1,2,3,1099511627776,]
        expect_error(lambda : nv.append(1<<40))
        assert len(nv)==4, nv
        s = 0
        for a in nv:
            s += a
        assert (s==1+2+3+(1<<40))
        s = max(nv)
        assert s == (1<<40)
        a, b, c, d = nv
        assert [a,b,c,d] == nv
        assert nv[-1] == (1<<40)
        assert nv[:2] == [1,2]
        assert nv[:-2] == [1,2]
        assert nv[::-1] == list(nv)[::-1], (list(nv)[::-1], nv[::-1])
        assert (nvector([1,2]) + nvector([3,4])) == [1,2,3,4]
        a = nvector([1,2])
        a += [3,4]
        assert a == [1,2,3,4], a

    def test_slice_bug(self):
        a = jt.NanoVector([2,3,4,5])
        assert a[:] == [2,3,4,5]
        assert a[1:] == [3,4,5]


if __name__ == "__main__":
    unittest.main()