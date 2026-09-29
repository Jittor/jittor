"""Synchronous distributed barriers must finish device collectives."""

import os
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("JT_NCCL_WORLD_SIZE") != "2",
                    reason="requires real two-rank NCCL")
def test_world_barrier_blocks_host_until_other_rank_writes_file():
    import torch
    import jittor as jt

    shared_root = os.environ.get("JITTOR_BARRIER_TEST_ROOT")
    if not shared_root:
        pytest.skip("requires an isolated shared JITTOR_BARRIER_TEST_ROOT")
    root = Path(shared_root)
    root.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank()
    ready = root / "rank1_ready"
    marker = root / "rank0_complete"
    assert torch.distributed.get_world_size() == 2
    assert torch.distributed.get_backend() == "nccl"
    assert torch.cuda.is_available() and jt.flags.use_cuda == 1
    torch.distributed.barrier()
    torch.cuda.synchronize()
    assert not marker.exists()
    if rank == 1:
        ready.write_text("ready", encoding="ascii")
    else:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline, "rank1 did not reach the barrier"
            time.sleep(0.01)
        time.sleep(1)
        marker.write_text("complete", encoding="ascii")
    torch.distributed.barrier()
    if rank == 1:
        assert marker.is_file(), "barrier returned before rank0 finished its host write"
