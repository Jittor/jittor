import numpy as np
import pytest
import torch

import jittor as jt


class State:
    def __init__(self, device):
        self.value = torch.zeros((2, 3), device=device)
        self.offsets = [0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("record", [False, True])
@torch.no_grad()
def test_step_preserves_python_argument_updates(device, record):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    state = State(device)

    def forward(value, *, state):
        result = value + state.value + state.offsets[0]
        state.value = state.value + value
        state.offsets[0] += 1
        return result

    step = jt.capture_step(forward, record=record)
    total = 0
    with jt.flag_scope(auto_graph_replay=0, backend_fallback="error"):
        for offset in range(6):
            value = torch.full((2, 3), float(offset + 1), device=device)
            result = step(value, state=state)
            assert result.device.type == device
            total += offset + 1
            np.testing.assert_array_equal(result.cpu().numpy(), np.full((2, 3), total + offset))
            assert state.offsets[0] == offset + 1
        assert "Python argument state" in step.refused
        assert step.stats["replayed"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("change", ["offset", "tensor"])
@torch.no_grad()
def test_step_recaptures_when_read_only_object_state_changes(device, change):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    state = State(device)

    def forward(value, *, state):
        return value + state.value + state.offsets[0]

    step = torch.compile(forward, mode="reduce-overhead")
    with jt.flag_scope(auto_graph_replay=0, backend_fallback="error"):
        for offset in range(6):
            value = torch.full((2, 3), float(offset), device=device)
            result = step(value, state=state)
            assert result.device.type == device
            np.testing.assert_array_equal(result.cpu().numpy(), np.full((2, 3), offset))
        replays = step.stats["replayed"]
        assert replays > 0
        if device == "cuda":
            assert step.stats["graph"] > 0
        if change == "offset":
            state.offsets[0] = 9
        else:
            # Rebind the Var inside the same tensor holder.
            state.value._update(torch.full((2, 3), 9.0, device=device))
        result = step(torch.ones((2, 3), device=device), state=state)
        np.testing.assert_array_equal(result.cpu().numpy(), np.full((2, 3), 10))
        assert step.stats["replayed"] == replays
        assert step.last_invalidated == "Python argument state changed"
