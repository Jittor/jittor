"""The fused vLLM block metadata path is limited to tested decode buckets."""

import sys
import types
import uuid

from jittor.compat import transaction as tx
from jittor_adapters.vllm.buffers import patch_block_table_metadata


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
