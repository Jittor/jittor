"""torch.randint must accept the documented torch signature spellings."""
import unittest

import numpy as np

import jittor as jt
import torch


class TestRandintSignatures(unittest.TestCase):
    def test_high_size_keywords(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.randint(high=2, size=(4, 5))
        self.assertEqual(tuple(value.shape), (4, 5))
        self.assertTrue(set(np.unique(value.numpy()).tolist()) <= {0, 1})

    def test_low_high_size_keywords(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.randint(low=3, high=7, size=(2, 3))
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertTrue(((value.numpy() >= 3) & (value.numpy() < 7)).all())

    def test_positional_high_size(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.randint(2, (2, 3))
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertTrue(set(np.unique(value.numpy()).tolist()) <= {0, 1})

    def test_positional_low_high_size(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.randint(0, 4, (3, 2))
        self.assertEqual(tuple(value.shape), (3, 2))
        self.assertTrue(((value.numpy() >= 0) & (value.numpy() < 4)).all())

    def test_mixed_high_positional_size_keyword(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.randint(2, size=(2, 2))
        self.assertEqual(tuple(value.shape), (2, 2))

    def test_dtype_keyword_is_honoured(self):
        with jt.flag_scope(use_cuda=0):
            value = torch.randint(0, 5, (2, 2), dtype=torch.int64)
        self.assertEqual(str(value.dtype), "torch.int64")

    def test_torchmetrics_classification_fixture_shape(self):
        # Exact call from torchmetrics tests/unittests/classification/_inputs.py
        with jt.flag_scope(use_cuda=0):
            target = torch.randint(high=2, size=(3, 4))
        self.assertEqual(tuple(target.shape), (3, 4))
        self.assertTrue(set(np.unique(target.numpy()).tolist()) <= {0, 1})
