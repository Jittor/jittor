"""Non-persistent buffers built under the meta context survive materialization."""
import unittest

import numpy as np

import jittor as jt
import torch
import torch.nn as nn


class TestMetaBufferMaterialization(unittest.TestCase):
    def test_registered_meta_buffer_is_tagged(self):
        with jt.flag_scope(use_cuda=0):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "position_ids", torch.arange(6).expand((1, -1)), persistent=False)
            self.assertTrue(getattr(module.position_ids, "_jittor_torch_meta_data", False))

    def test_empty_like_of_meta_buffer_preserves_values(self):
        with jt.flag_scope(use_cuda=0):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "position_ids", torch.arange(6).expand((1, -1)), persistent=False)
            materialized = torch.empty_like(module.position_ids, device="cpu")
            np.testing.assert_array_equal(np.asarray(materialized).ravel(), np.arange(6))

    def test_materialization_round_trip_through_module_attribute(self):
        with jt.flag_scope(use_cuda=0):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "ids", torch.arange(5).expand((1, -1)), persistent=False)
            # transformers' _move_missing_keys_from_meta_to_device: empty_like
            # the non-persistent buffer and assign it back.
            module.ids = torch.empty_like(module.ids, device="cpu")
            np.testing.assert_array_equal(np.asarray(module.ids).ravel(), np.arange(5))

    def test_plain_empty_like_stays_uninitialized(self):
        with jt.flag_scope(use_cuda=0):
            source = torch.arange(6)
            out = torch.empty_like(source)
            self.assertEqual(tuple(out.shape), (6,))
            self.assertFalse(getattr(source, "_jittor_torch_meta_data", False))
