# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import unittest
import numpy as np
import random
from _helpers.assertions import expect_error
from jittor.dataset.mnist import MNIST
import jittor.transform as trans
from tqdm import tqdm
import pytest

class BBox:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return bool((self.x == other.x).all())

def test_ring_buffer():
    buffer = jt.RingBuffer(2000)
    def test_send_recv(data):
        print("test send recv", type(data))
        buffer.push(data)
        recv = buffer.pop()
        if isinstance(data, (np.ndarray, jt.Var)):
            actual = recv.data if isinstance(recv, jt.Var) else recv
            expected = data.data if isinstance(data, jt.Var) else data
            np.testing.assert_array_equal(actual, expected)
        else:
            assert data == recv

    n_byte = 0
    test_send_recv(1)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv(100000000000)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()

    test_send_recv(1e-5)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv(100000000000.0)
    n_byte += 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()

    test_send_recv("float32")
    n_byte += 1 + 8 + 7
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv("")
    n_byte += 1 + 8 + 0
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv("xxxxxxxxxx")
    n_byte += 1 + 8 + 10
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()

    test_send_recv([1,0.2])
    n_byte += 1 + 8 + 1 + 8 + 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()
    test_send_recv({'asd':1})
    n_byte += 1 + 8 + 1 + 8 + 3 + 1 + 8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push()

    test_send_recv(np.random.rand(10,10))
    n_byte += 1 + 16 + 4 + 10*10*8
    assert n_byte == buffer.total_pop() and n_byte == buffer.total_push(), \
        (n_byte, buffer.total_pop(), n_byte, buffer.total_push())
    view = np.arange(100, dtype=np.float64).reshape(10, 10)[:, ::2]
    view_buffer = jt.RingBuffer(2000)
    view_buffer.push(view)
    np.testing.assert_array_equal(view_buffer.pop().data, view)
    test_send_recv(test_ring_buffer)

    test_send_recv(jt.array(np.random.rand(10,10)))

    bbox = BBox(jt.array(np.random.rand(10,10)))
    test_send_recv(bbox)

    expect_error(lambda: test_send_recv(np.random.rand(10,1000)))


class TestRingBuffer(unittest.TestCase):

    def test_ring_buffer(self):
        test_ring_buffer()

    # Downloads an external dataset archive; see the note in
    # tests/models/test_resnet.py.
    @pytest.mark.network
    @unittest.expectedFailure
    def test_dataset(self):
        # KI-TEST-001: the repeated dataset/RingBuffer path remains unstable.
        self.train_loader = MNIST(train=True, transform=trans.Resize(224)) \
            .set_attrs(batch_size=300, shuffle=True)
        self.train_loader.num_workers = 1
        import time
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            # time.sleep(5)
            # print("break")
            # break
            # self.train_loader.display_worker_status()
            if batch_idx > 30:
                break
            pass
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            # time.sleep(5)
            # print("break")
            # break
            # self.train_loader.display_worker_status()
            if batch_idx > 300:
                break
            pass
        

if __name__ == "__main__":
    unittest.main()
