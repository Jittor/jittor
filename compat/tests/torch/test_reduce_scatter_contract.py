"""DeepSpeed's list-form reduce-scatter packs rank chunks and writes the shard."""
import numpy as np
import pytest

import jittor as jt
import torch
from jittor.compat.torch.installers import distributed


def test_reduce_scatter_packs_rank_chunks_in_order(monkeypatch):
    observed = []

    def reduce_packed(packed):
        observed.append(packed.numpy().tolist())
        return jt.array(np.asarray([70, 80], dtype=np.int32))

    monkeypatch.setattr(distributed, "_require_supported_group", lambda group: 2)
    monkeypatch.setattr(distributed._collectives, "_reduce_scatter_padded", reduce_packed)
    output = jt.zeros((2,), dtype="int32")
    chunks = [
        jt.array(np.asarray([10, 20], dtype=np.int32)),
        jt.array(np.asarray([30, 40], dtype=np.int32)),
    ]
    assert torch.distributed.reduce_scatter is distributed._reduce_scatter
    assert torch.distributed.reduce_scatter(output, chunks) is None
    assert observed == [[10, 20, 30, 40]]
    np.testing.assert_array_equal(output.numpy(), [70, 80])


def test_reduce_scatter_rejects_chunk_shape_mismatch_before_issuing(monkeypatch):
    monkeypatch.setattr(distributed, "_require_supported_group", lambda group: 2)
    monkeypatch.setattr(
        distributed._collectives,
        "_reduce_scatter_padded",
        lambda packed: pytest.fail("collective should not be issued"),
    )
    output = jt.zeros((2,), dtype="int32")
    chunks = [jt.zeros((2,), dtype="int32"), jt.zeros((3,), dtype="int32")]
    with pytest.raises(ValueError, match="match output shape"):
        torch.distributed.reduce_scatter(output, chunks)