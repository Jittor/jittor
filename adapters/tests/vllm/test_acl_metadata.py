"""CPU reference checks for the ACL vLLM metadata fallback."""

import unittest

import numpy as np

from jittor_adapters.vllm.buffers import (
    _acl_compute_slot_mappings,
    _acl_compute_v1_slot_mapping,
    _acl_gather_block_tables,
)


class ACLMetadataReferenceTests(unittest.TestCase):
    def test_shim_view_copy_updates_parent(self):
        import torch
        parent = torch.zeros((2, 4), dtype=torch.int32, device='cpu')
        parent[1, 1:3].copy_(torch.tensor([5, 6], dtype=torch.int32))
        np.testing.assert_array_equal(
            parent.numpy(), [[0, 0, 0, 0], [0, 5, 6, 0]])

    def test_gather_reorders_requests_and_zeroes_padding(self):
        mirrors = [np.array([[7, 8, 9], [2, 3, 4]], dtype=np.int32)]
        counts = np.array([[2, 1]], dtype=np.int32)
        result = _acl_gather_block_tables(mirrors, counts, [1, 0], 4)[0]
        np.testing.assert_array_equal(
            result, [[2, 0, 0], [7, 8, 0], [0, 0, 0], [0, 0, 0]])

    def test_slot_mapping_uses_physical_blocks_and_padding(self):
        mirrors = [np.array([[7, 8, 9], [2, 3, 4]], dtype=np.int32)]
        counts = np.array([[2, 1]], dtype=np.int32)
        result = _acl_compute_slot_mappings(
            mirrors, counts, [1, 0], [0, 2, 4], [0, 1, 2, 5],
            [4], 8, -1)
        np.testing.assert_array_equal(
            result[0], [8, 9, 30, 33, -1, -1, -1, -1])

    def test_slot_mapping_multiple_kv_groups(self):
        mirrors = [np.array([[1, 3]], dtype=np.int32),
                   np.array([[2, 4, 6, 8]], dtype=np.int32)]
        counts = np.array([[2], [4]], dtype=np.int32)
        result = _acl_compute_slot_mappings(
            mirrors, counts, [0], [0, 3], [0, 3, 7], [4, 2], 4, -1)
        np.testing.assert_array_equal(result,
                                      [[4, 7, 15, -1], [4, 9, 17, -1]])

    def test_invalid_block_reference_fails_closed(self):
        with self.assertRaises(IndexError):
            _acl_compute_slot_mappings(
                [np.array([[1]], dtype=np.int32)],
                np.array([[1]], dtype=np.int32),
                [0], [0, 1], [5], [4], 1, -1)

    def test_v1_slot_mapping_reads_host_mirror_and_pads(self):
        table = np.array([[7, 8, 0], [2, 3, 0]], dtype=np.int32)
        slots = _acl_compute_v1_slot_mapping(
            table, [2, 2], [0, 2, 4], [0, 5, 3, 4], 4, 8, -1)
        np.testing.assert_array_equal(
            slots, [28, 33, 11, 12, -1, -1, -1, -1])

    def test_v1_slot_mapping_rejects_unallocated_block(self):
        with self.assertRaises(IndexError):
            _acl_compute_v1_slot_mapping(
                np.array([[4, 5]], dtype=np.int32), [1],
                [0, 1], [4], 4, 8, -1)


if __name__ == '__main__':
    unittest.main()
