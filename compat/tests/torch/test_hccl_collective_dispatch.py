"""Single-node HCCL backend routing for sharded Torch collectives."""

from types import SimpleNamespace

from jittor.compat import collectives


def test_hccl_gather_is_used_when_two_ascend_ranks_are_ready(monkeypatch):
    local = object()
    gathered = object()
    calls = []
    monkeypatch.setenv("JT_HCCL_WORLD_SIZE", "2")
    monkeypatch.setattr(collectives, "_world_size", lambda: 2)
    monkeypatch.setattr(
        collectives,
        "_hccl_ops",
        lambda: SimpleNamespace(hccl_all_gather=lambda shard: calls.append(shard) or gathered),
    )
    monkeypatch.setattr(collectives, "_nccl_ops", lambda: None)

    assert collectives._in_true_distributed()
    assert collectives._all_gather_shards(local) is gathered
    assert calls == [local]