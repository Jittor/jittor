# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import gc

class TestAllocator(unittest.TestCase):
    def test_stat(self):
        jt.clean()
        with jt.flag_scope(use_stat_allocator=1, use_sfrl_allocator = 0):
            a = jt.random([10,10])
            b = a+a
            c = a*b
            c.data
            del a,b,c
            gc.collect()
        assert jt.flags.stat_allocator_total_alloc_call == 2
        assert jt.flags.stat_allocator_total_alloc_byte == 800
        assert jt.flags.stat_allocator_total_free_call == 2
        assert jt.flags.stat_allocator_total_free_byte == 800

if __name__ == "__main__":
    unittest.main()
