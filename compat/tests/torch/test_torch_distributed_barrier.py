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


def test_world_barrier_delegates_to_process_group(monkeypatch):
    from jittor.compat.torch.installers import distributed as installer

    calls = []

    class ProcessGroup:
        def barrier(self, device_id=None):
            calls.append(device_id)

    process_group = ProcessGroup()
    monkeypatch.setattr(installer, "_require_supported_group", lambda group: 2)
    monkeypatch.setattr(
        installer._process_group, "get_default_group", lambda: process_group
    )

    assert installer._barrier(device_ids=[3]) is None
    assert calls == [3]


def test_native_process_group_barrier_uses_backend_device_and_sync(monkeypatch):
    import jittor.distributed.process_group as process_group_module
    from jittor.distributed.process_group import ProcessGroup

    events = []

    class Marker:
        def cuda(self, device_id=None):
            events.append(("cuda", device_id))
            return self

        def sync(self, device_sync=False):
            events.append(("sync", device_sync))
            return self

    marker = Marker()
    monkeypatch.setattr(process_group_module.jt, "array", lambda *a, **k: marker)
    process_group = ProcessGroup()
    process_group._backend_kind = "nccl"
    process_group._backend_handle = 0
    monkeypatch.setattr(process_group, "rank", lambda: 0)
    monkeypatch.setattr(process_group, "size", lambda: 2)

    def all_reduce(tensor, op):
        events.append(("all_reduce", op))
        return tensor

    monkeypatch.setattr(process_group, "all_reduce", all_reduce)
    assert process_group.barrier(device_id=4) is None
    assert events == [("cuda", 4), ("all_reduce", "sum"), ("sync", True)]
