"""Non-persistent buffers built under the meta context survive materialization."""
import unittest

import numpy as np

import jittor as jt
import torch
import torch.nn as nn

from _helpers import capability as _test_capability


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


class TestMetaBufferPlacement(unittest.TestCase):
    """Materialized meta buffers must follow the requested device."""

    def test_materialization_forwards_the_requested_device(self):
        # White-box on CPU: the CUDA move itself needs a GPU, so record that
        # the requested device reaches the residency helper with force=True
        # instead of being dropped by the constructor adapter.
        from jittor.compat.torch.installers import factories

        recorded = []
        original = factories._make_cuda_resident

        def recording(value, **kwargs):
            recorded.append(kwargs)
            return value

        factories._make_cuda_resident = recording
        try:
            with jt.flag_scope(use_cuda=0):
                with torch.device("meta"):
                    module = nn.Module()
                    module.register_buffer(
                        "inv_freq", torch.arange(4), persistent=False)
                materialized = factories._materialize_meta_like(
                    module.inv_freq, torch.device("cuda", 0))
            self.assertEqual(len(recorded), 1)
            self.assertIs(recorded[0]["device"].type, "cuda")
            self.assertTrue(recorded[0]["force"])
            np.testing.assert_array_equal(
                np.asarray(materialized).ravel(), np.arange(4))
        finally:
            factories._make_cuda_resident = original

    def test_module_to_casts_a_non_persistent_buffer(self):
        # torch's Module.to touches _buffers, which includes non-persistent
        # entries; the dtype cast is observable without a second device.
        with jt.flag_scope(use_cuda=0):
            module = nn.Module()
            module.register_buffer(
                "inv_freq", torch.arange(4, dtype=torch.float32),
                persistent=False)
            module.to(torch.float64)
            self.assertEqual(str(module.inv_freq.dtype), "torch.float64")

    @unittest.skipUnless(
        _test_capability.check_accelerator("cuda", backend=jt).enabled,
        "requires CUDA",
    )
    def test_meta_buffer_materializes_on_the_requested_cuda_device(self):
        with jt.flag_scope(use_cuda=1):
            with torch.device("meta"):
                module = nn.Module()
                module.register_buffer(
                    "inv_freq", torch.arange(4, dtype=torch.float32),
                    persistent=False)
            materialized = torch.empty_like(
                module.inv_freq, device=torch.device("cuda", 0))
            self.assertTrue(materialized.is_cuda)
            np.testing.assert_array_equal(
                materialized.numpy(), np.arange(4, dtype="float32"))

    @unittest.skipUnless(
        _test_capability.check_accelerator("cuda", backend=jt).enabled,
        "requires CUDA",
    )
    def test_module_to_moves_a_non_persistent_buffer(self):
        module = nn.Module()
        module.register_buffer(
            "inv_freq", torch.arange(4, dtype=torch.float32), persistent=False)
        module.to("cuda")
        self.assertTrue(module.inv_freq.is_cuda)
