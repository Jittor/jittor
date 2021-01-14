# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers: 
#     Guoye Yang <498731903@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import gc

def test(h, w, total_alloc_call, total_alloc_byte, total_free_call = 0, total_free_byte = 0):
        jt.clean()
        jt.gc()
        with jt.flag_scope(use_stat_allocator=1):
            a = jt.random([h,w])
            b = a+a
            c = a*b
            c.data
            del a,b,c
            gc.collect()
            x = (
                jt.flags.stat_allocator_total_alloc_call,
                jt.flags.stat_allocator_total_alloc_byte,
                jt.flags.stat_allocator_total_free_call,
                jt.flags.stat_allocator_total_free_byte
            )
            y = (total_alloc_call, total_alloc_byte, total_free_call, total_free_byte)
            assert x==y, (x, y)


class TestAllocator2(unittest.TestCase):
    def test_stat(self):
        #small_block
        test(10, 10, 1, 1048576) #800
        #small_block
        test(100, 100, 1, 1048576) #80000
        #large_block
        test(1000, 1000, 1, 20971520) #8000000
        #large_block2
        test(8000, 1000, 2, 67108864) #64000000

if __name__ == "__main__":
    unittest.main()
