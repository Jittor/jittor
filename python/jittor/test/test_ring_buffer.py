# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
from jittor_utils.ring_buffer import *
import unittest

def test_ring_buffer():
    buffer = mp.Array('c', 8000, lock=False)
    buffer = RingBuffer(buffer)
    def test_send_recv(data):
        print("test send recv", type(data))
        buffer.send(data)
        recv = buffer.recv()
        if isinstance(recv, np.ndarray):
            assert (recv == data).all()
        else:
            assert data == recv
    test_send_recv("float32")
    test_send_recv("")
    test_send_recv("xxxxxxxxxx")

    test_send_recv(1)
    test_send_recv(100000000000)

    test_send_recv(1e-5)
    test_send_recv(100000000000.0)

    test_send_recv([1,0.2])
    test_send_recv({'asd':1})

    test_send_recv(np.random.rand(10,10))
        
def test_ring_buffer_allocator(p=0.7):
    print("test_ring_buffer_allocator", p)
    n = 1000
    buffer = RingBufferAllocator(n)
    m = 10000
    sizes = [0]*m
    a = [-1]*n
    l = 0
    r = 0
    for i in range(m):
        if l==r or random.random()<0.7:
            size = random.randint(10, 20)
            location = buffer.alloc(size)
            if location is not None:
                sizes[r] = size
                for j in range(location, location+size):
                    a[j] = r
                r += 1
                continue
        assert l<r
        size = sizes[l]
        location = buffer.free(size)
        assert location is not None, buffer
        for j in range(location, location+size):
            assert a[j] == l
        l += 1


class TestReindexOp(unittest.TestCase):
    def test_ring_buffer_allocator(self):
        test_ring_buffer_allocator(0.7)
        test_ring_buffer_allocator(0.3)

    def test_ring_buffer(self):
        test_ring_buffer()
        

if __name__ == "__main__":
    unittest.main()