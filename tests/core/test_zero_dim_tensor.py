# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Core scalar tensors have rank zero while retaining one stored element."""

import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase, get_all_device_types, use_cuda_for


class TestZeroDimTensor(JittorTestCase):
    def _devices(self, body):
        for device in get_all_device_types():
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda_for(device)):
                    body(device)

    def test_scalar_construction_and_one_element_vector_are_distinct(self):
        def body(device):
            scalar = jt.array(np.array(2.0, dtype="float64"), dtype="float64")
            vector = jt.array(np.array([2.0], dtype="float64"), dtype="float64")
            self.assertEqual(tuple(scalar.shape), (), device)
            self.assertEqual(tuple(scalar.numpy().shape), (), device)
            self.assertEqual(tuple(vector.shape), (1,), device)

            tensor = jt.array(np.array([3.0, 4.0], dtype="float32"))
            self.assertEqual(str((vector * tensor).dtype), "float64", device)

        self._devices(body)

    def test_index_and_full_reduce_produce_zero_dim(self):
        def body(device):
            value = jt.array(np.arange(6, dtype="float32").reshape(2, 3))
            indexed = value[0, 1]
            reduced = value.sum()
            self.assertEqual(tuple(indexed.shape), (), device)
            self.assertEqual(tuple(reduced.shape), (), device)
            self.assertEqual(indexed.item(), 1.0)
            self.assertEqual(reduced.item(), 15.0)

        self._devices(body)

    def test_zero_dim_factories_reshape_arithmetic_and_grad(self):
        def body(device):
            zero = jt.zeros(())
            one = jt.ones(()).reshape(())
            self.assertEqual(tuple(zero.shape), (), device)
            self.assertEqual(tuple(one.shape), (), device)
            self.assertEqual(tuple((zero + one).shape), (), device)
            np.testing.assert_array_equal((zero + one).numpy(), np.array(1.0))

            value = jt.array(np.arange(4, dtype="float32"))
            grad = jt.grad((value * value).sum(), value)
            np.testing.assert_allclose(grad.numpy(), 2 * np.arange(4, dtype="float32"))

        self._devices(body)

    def test_cuda_full_reduce_fast_path_keeps_zero_rank(self):
        def body(device):
            if device != "cuda":
                return
            value = jt.ones((1 << 14,), dtype="float32")
            reduced = value.sum()
            self.assertEqual(tuple(reduced.shape), (), device)
            self.assertEqual(reduced.item(), float(1 << 14))

        self._devices(body)


if __name__ == "__main__":
    unittest.main()
