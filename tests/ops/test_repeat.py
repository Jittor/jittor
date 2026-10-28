# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: 
#     Zheng-Ning Liu <lzhengning@gmail.com>
#     Dun Liang <randonlang@gmail.com>. 
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import unittest
import jittor as jt
import numpy as np


class TestRepeatOp(unittest.TestCase):
    def test_repeat(self):
        np_a = np.arange(5)
        jt_a = jt.array(np_a)

        np_b = np.tile(np_a, (2, 3))
        jt_b = jt.repeat(jt_a, (2, 3))

        assert np.allclose(np_b, jt_b.data)

        np_b = np.tile(np_a, (2, 3, 1))
        jt_b = jt.repeat(jt_a, (2, 3, 1))

        assert np.allclose(np_b, jt_b.data)

        np_a = np.arange(24).reshape(2, 3, 4)
        jt_a = jt.array(np_a)

        np_b = np.tile(np_a, (2, 3))
        jt_b = jt.repeat(jt_a, (2, 3))

        assert np.allclose(np_b, jt_b.data)


    def test_repeat_results_are_their_own_dense_storage(self):
        # Both ops are copies in torch. Built as expand + reshape they were
        # stride-0 views: an in-place write through the result was refused and
        # the result kept the source's storage (PEFT's OFT expands an identity
        # with repeat and then adds into it; HRA trains a repeat_interleave
        # result in place).
        source = jt.array(np.array([[1., 2.], [3., 4.]], dtype="float32"))
        repeated = jt.repeat(source, 2, 2)
        self.assertTrue(repeated._storage_is_contiguous())
        self.assertEqual(list(repeated._storage_strides()), [4, 1])
        interleaved = jt.repeat_interleave(source, 2, dim=0)
        self.assertTrue(interleaved._storage_is_contiguous())
        self.assertEqual(list(interleaved._storage_strides()), [2, 1])

    def test_highdim(self):
        np_a = np.arange(64).reshape(2, 2, 2, 2, 2, 2)
        jt_a = jt.array(np_a)

        np_b = np.tile(np_a, (2, 3))
        jt_b = jt.repeat(jt_a, (2, 3))

        assert np.allclose(np_b, jt_b.data)

        np_b = np.tile(np_a, (2, 1, 1, 3))
        jt_b = jt.repeat(jt_a, (2, 1, 1, 3))

        assert np.allclose(np_b, jt_b.data)

        np_b = np.tile(np_a, (2, 1, 1, 1, 3, 1))
        jt_b = jt.repeat(jt_a, (2, 1, 1, 1, 3, 1))

        assert np.allclose(np_b, jt_b.data)

if __name__ == "__main__":
    unittest.main()