# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from .test_core import expect_error

class TestStopFuse(unittest.TestCase):
    def test_stop_fuse(self):
        with jt.profile_scope() as report:
            a = jt.float32(0).stop_fuse()
            c = jt.float32(0)
            bs = [c]
            for i in range(2000):
                b = jt.float32(i)*2*c
                bs.append(b)
                a += b

            a = a*2

            dbs = jt.grad(a, bs)
            jt.sync(dbs+[a])
        
        for a in report[1:]:
            # origin is 50
            # after update queue, increase to 102
            assert len(a[0].split("opkey")) < 110, len(a[0].split("opkey"))

    def test_stop_fuse2(self):
        with jt.profile_scope() as report:
            a = jt.float32(0).stop_fuse()
            c = jt.float32(0).stop_fuse()
            bs = [c]
            for i in range(2000):
                b = jt.float32(i)*2*c
                bs.append(b)
                a += b

            a = a*2

            dbs = jt.grad(a, bs)
            jt.sync(dbs+[a])
        
        for a in report[1:]:
            # origin is 8
            # after update queue, increase to 12
            assert len(a[0].split("opkey")) < 16, len(a[0].split("opkey"))

if __name__ == "__main__":
    unittest.main()