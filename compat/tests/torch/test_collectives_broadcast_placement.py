"""Collective outputs must use the placement of the tensor being broadcast."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jittor.compat import collectives


class _ProbeVar:
    def __init__(self, location, device_id=-1, placement_backend=-1, events=None):
        self._location = location
        self.device_id = device_id
        self.placement_backend = placement_backend
        self.events = events if events is not None else []

    def location(self):
        return self._location

    def mpi_broadcast(self):
        self.events.append(("broadcast",))
        return "rank0-value"

    def assign(self, value):
        self.events.append(("assign", value))


@pytest.mark.parametrize(
    ("location", "device_id", "placement_backend", "expected"),
    [
        ("device", 0, -1, (1, 0)),
        ("cpu", -1, -1, (0, 0)),
        ("none", 3, 1, (1, 3)),
    ],
)
def test_broadcast_uses_tensor_placement(location, device_id,
                                         placement_backend, expected):
    events = []
    token = object()

    def set_placement(backend, index):
        events.append(("set", backend, index))
        return token

    def reset_placement(value):
        events.append(("reset", value))

    fake_core = SimpleNamespace(
        _set_tensor_placement=set_placement,
        _reset_tensor_placement=reset_placement,
    )
    var = _ProbeVar(location, device_id, placement_backend, events)
    with patch.object(collectives, "_world_size", return_value=2):
        with patch.object(collectives.jt, "core", fake_core):
            assert collectives._broadcast_from_rank0(var) is var

    assert events == [
        ("set", expected[0], expected[1]),
        ("broadcast",),
        ("assign", "rank0-value"),
        ("reset", token),
    ]


def test_broadcast_restores_placement_when_collective_raises():
    events = []
    token = object()

    def set_placement(backend, index):
        events.append(("set", backend, index))
        return token

    def reset_placement(value):
        events.append(("reset", value))

    fake_core = SimpleNamespace(
        _set_tensor_placement=set_placement,
        _reset_tensor_placement=reset_placement,
    )
    var = _ProbeVar("device", 0, -1, events)

    def fail_broadcast():
        events.append(("broadcast",))
        raise RuntimeError("collective failed")

    var.mpi_broadcast = fail_broadcast
    with patch.object(collectives, "_world_size", return_value=2):
        with patch.object(collectives.jt, "core", fake_core):
            with pytest.raises(RuntimeError, match="collective failed"):
                collectives._broadcast_from_rank0(var)

    assert events == [("set", 1, 0), ("broadcast",), ("reset", token)]


def test_single_rank_broadcast_remains_an_identity():
    events = []
    var = _ProbeVar("device", 0, -1, events)
    with patch.object(collectives, "_world_size", return_value=1):
        assert collectives._broadcast_from_rank0(var) is var
    assert events == []
