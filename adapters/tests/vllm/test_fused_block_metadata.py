"""The fused vLLM block metadata path is limited to tested decode buckets."""

import sys
import types
import unittest
import uuid
from unittest.mock import patch

from jittor.compat import transaction as tx
from jittor_adapters.vllm.buffers import (
    _native_cuda_active, _native_metadata_batch_eligible,
    patch_block_table_metadata)


class NativeMetadataPolicyTests(unittest.TestCase):
    def test_batch_four_is_opt_in(self):
        with patch.dict("os.environ", {}, clear=False):
            from os import environ
            environ.pop("JITTOR_VLLM_NATIVE_METADATA_BATCH4", None)
            self.assertTrue(_native_metadata_batch_eligible(1))
            self.assertFalse(_native_metadata_batch_eligible(4))
            environ["JITTOR_VLLM_NATIVE_METADATA_BATCH4"] = "1"
            self.assertTrue(_native_metadata_batch_eligible(1))
            self.assertTrue(_native_metadata_batch_eligible(4))
            self.assertFalse(_native_metadata_batch_eligible(2))
            self.assertFalse(_native_metadata_batch_eligible(8))


def test_native_cuda_guard_rejects_acl_alias():
    fake = types.SimpleNamespace(
        flags=types.SimpleNamespace(use_cuda=1, use_acl=1),
        compiler=types.SimpleNamespace(has_cuda=0, has_acl=1),
    )
    assert not _native_cuda_active(fake)
    fake.compiler.has_cuda = 1
    assert not _native_cuda_active(fake)
    # Rebuild the namespace rather than writing `fake.flags.use_acl`: the
    # repository contract cannot tell a fake flags object from jt.flags and
    # would read the write as an unrestored global flag change.
    fake.flags = types.SimpleNamespace(use_cuda=1, use_acl=0)
    assert _native_cuda_active(fake)
    fake.compiler = None
    assert not _native_cuda_active(fake)


def test_batch_four_fuses_but_padded_and_other_shapes_fall_back(monkeypatch):
    monkeypatch.setenv("JITTOR_VLLM_FUSED_BLOCK_METADATA", "1")
    calls = types.SimpleNamespace(code=0, sync=0, gather=0, slots=0)

    class Var:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, _key):
            return self

    jt = types.ModuleType("jittor")
    jt.Var = Var
    jt.flags = types.SimpleNamespace(use_cuda=1)
    jt.compiler = types.SimpleNamespace(has_cuda=1, has_acl=0)

    def code(*, inputs, outputs, cuda_src):
        assert len(inputs) >= 5
        assert len(outputs) == 2
        assert "fused_block_metadata_native" in cuda_src
        calls.code += 1

    def sync(outputs, blocking):
        assert len(outputs) == 2
        assert blocking is False
        calls.sync += 1

    jt.code = code
    jt.sync = sync
    monkeypatch.setitem(sys.modules, "jittor", jt)

    class BlockTables:
        num_kv_cache_groups = 1
        cp_size = 1
        max_num_batched_tokens = 16
        kernel_block_sizes = (16,)

        def __init__(self):
            self.block_tables = [types.SimpleNamespace(gpu=Var((4, 32)))]
            self.num_blocks = types.SimpleNamespace(gpu=Var((4,)))
            self.input_block_tables = [Var((4, 32))]
            self.slot_mappings = Var((1, 16))

        def gather_block_tables(self, idx_mapping, num_reqs_padded):
            calls.gather += 1
            return tuple(self.input_block_tables)

        def compute_slot_mappings(self, idx_mapping, query_start_loc,
                                  positions, num_tokens_padded):
            calls.slots += 1
            return self.slot_mappings

    module = types.SimpleNamespace(BlockTables=BlockTables, PAD_SLOT_ID=-1)
    with tx.runtime_hook("fused-block-metadata-" + uuid.uuid4().hex) as hook:
        assert patch_block_table_metadata(module)
        table = BlockTables()
        idx4 = Var((4,))
        query = Var((5,))
        positions = Var((4,))

        table.gather_block_tables(idx4, 4)
        table.compute_slot_mappings(idx4, query, positions, 4)
        assert (calls.code, calls.sync, calls.gather, calls.slots) == (1, 1, 0, 0)

        table.gather_block_tables(idx4, 8)
        table.compute_slot_mappings(idx4, query, positions, 4)
        assert (calls.code, calls.gather, calls.slots) == (1, 1, 1)

        idx2 = Var((2,))
        table.gather_block_tables(idx2, 2)
        table.compute_slot_mappings(idx2, query, positions, 2)
        assert (calls.code, calls.gather, calls.slots) == (1, 2, 2)
    hook.rollback()
