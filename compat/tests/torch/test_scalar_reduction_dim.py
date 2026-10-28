# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Torch accepts ``scalar.sum(dim=0)``; jittor's reduce does not.

torchmetrics reduces 0-D classification statistics with ``sum(dim=0)``. Torch
treats dim 0 (or -1) as the scalar's only axis and returns the scalar, while
the native reduce rejects every dim on a 0-D Var. Keep the Torch meaning in
the frontend and leave genuinely out-of-range dims to fail.
"""

import unittest

import torch


class TestScalarReductionDim(unittest.TestCase):
    def test_scalar_sum_accepts_dim_zero_and_minus_one(self):
        value = torch.tensor(3.5)
        self.assertEqual(tuple(value.sum(dim=0).shape), ())
        self.assertEqual(float(value.sum(dim=0)), 3.5)
        self.assertEqual(float(value.sum(dim=-1)), 3.5)
        self.assertEqual(tuple(value.sum(dim=(0,), keepdim=True).shape), (1,))
        self.assertEqual(float(value.sum(dim=(0,), keepdim=True)[0]), 3.5)

    def test_scalar_sum_still_rejects_an_out_of_range_dim(self):
        with self.assertRaises(IndexError):
            torch.tensor(3.5).sum(dim=1)


if __name__ == "__main__":
    unittest.main()
