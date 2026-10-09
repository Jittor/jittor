"""Synchronous HCCL reduction must finish the preceding lazy graph."""

from jittor.compat.torch.installers import distributed


def test_hccl_all_reduce_flushes_pending_graph_before_group_call(monkeypatch):
    events = []

    def flush_all(wait):
        assert wait is True
        events.append("flush pending graph")

    class DelayedTensor:
        def sync(self):
            events.append("materialize result")

        def update(self, value):
            assert value == "reduced"
            events.append("copy")

    class Group:
        def size(self):
            return 2

        def rank(self):
            return 0

        def _all_reduce(self, tensor, operation):
            assert events == ["flush pending graph"]
            assert operation == "sum"
            events.append("collective")
            return "reduced"

    monkeypatch.setenv("JT_HCCL_WORLD_SIZE", "2")
    monkeypatch.setattr(distributed.jt, "sync_all", flush_all)
    assert distributed._all_reduce(DelayedTensor(), group=Group()) is None
    assert events == ["flush pending graph", "collective", "copy", "materialize result"]