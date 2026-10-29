"""Observed eager writes refuse graph capture; this does not implement replayed writes."""

import numpy as np
import pytest
import torch
import jittor as jt

from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor._runtime.graph_replay import _ALIAS_WRITE_WARMUPS


@pytest.fixture(params=("cpu", "cuda:0"))
def device(request):
    if request.param == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture
def execution(device):
    # Disable the separate automatic policy so the explicit wrapper is the
    # only source of capture. A dedicated CUDA job must verify no CUDA skips.
    use_cuda = int(device == "cuda:0")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=use_cuda, backend_fallback="error",
                          auto_graph_replay=0), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


def _values(tensor):
    return tensor.detach().cpu().numpy().copy()


def _device(tensor, device):
    assert str(tensor.device) == device
    expected = ("cuda", 0) if device == "cuda:0" else ("cpu", -1)
    assert tuple(jt.core.dispatch_context([tensor])) == expected


class _InputWrite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value, offset):
        self.calls += 1
        # Keep an independent old-value expression alive across the write.
        old = value + 2
        value += offset
        return old, value * 3


def test_warmup_write_refuses_capture_and_preserves_eager_aliases(device, execution):
    model = _InputWrite().eval()
    replay = jt.graph_replay(model)
    for step, offset in enumerate((1., -2., 4.), 1):
        source = torch.tensor([step, step + 2.], dtype=torch.float32, device=device)
        alias = source.detach()
        view = source.view(2)
        pointer = source.data_ptr()
        assert alias.data_ptr() == view.data_ptr() == pointer
        before = np.array([step, step + 2.], dtype=np.float32)
        delta = torch.tensor([offset, offset], device=device)
        with torch.no_grad():
            old, output = replay(source, delta)
        _device(source, device)
        _device(output, device)
        assert model.calls == step
        assert source.data_ptr() == alias.data_ptr() == view.data_ptr() == pointer
        np.testing.assert_array_equal(_values(old), before + 2.)
        np.testing.assert_array_equal(_values(source), before + offset)
        np.testing.assert_array_equal(_values(alias), before + offset)
        np.testing.assert_array_equal(_values(view), before + offset)
        np.testing.assert_array_equal(_values(output), (before + offset) * 3.)
    assert "alias-preserving" in replay.refused
    assert replay.stats["captured"] == replay.stats["replayed"] == 0
    assert replay.stats["rebuilt"] == 3


class _MaskLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_alias = None

    def forward(self, value, offset):
        self.calls += 1
        positions = torch.arange(0, 4, device=value.device, dtype=torch.int64)
        self.last_alias = positions.detach()
        positions += offset
        return positions + value


def test_internal_mask_like_write_returns_each_warmup_result(device, execution):
    model = _MaskLike().eval()
    replay = jt.graph_replay(model)
    for step, offset in enumerate((0, 2, 5), 1):
        value = torch.tensor([step, step, step, step], dtype=torch.int64, device=device)
        delta = torch.tensor(offset, dtype=torch.int64, device=device)
        with torch.no_grad():
            output = replay(value, delta)
        _device(output, device)
        assert model.calls == step
        np.testing.assert_array_equal(_values(model.last_alias), np.arange(4) + offset)
        np.testing.assert_array_equal(_values(output), np.arange(4) + offset + step)
    assert "alias-preserving" in replay.refused
    assert replay.stats["captured"] == replay.stats["replayed"] == 0
    assert replay.stats["rebuilt"] == 3


def test_automatic_policy_refuses_mask_write_before_capture(device, execution):
    model = _MaskLike().eval()
    observed_replay = None
    with jt.flag_scope(auto_graph_replay=1), torch.no_grad():
        for step, offset in enumerate((0, 2, 5), 1):
            value = torch.tensor([step] * 4, dtype=torch.int64, device=device)
            delta = torch.tensor(offset, dtype=torch.int64, device=device)
            result = model(value, delta)
            _device(result, device)
            np.testing.assert_array_equal(_values(result), np.arange(4) + offset + step)
            np.testing.assert_array_equal(_values(model.last_alias), np.arange(4) + offset)
            assert model.calls == step
            state = model.__dict__["_auto_graph_replay"]
            if step == 2:
                observed_replay = state.replay
                assert observed_replay is not None
                assert "alias-preserving" in observed_replay.refused
                assert observed_replay.stats["captured"] == 0
                assert observed_replay.stats["replayed"] == 0
            if step == 3:
                assert state.give_up and state.replay is None
    assert observed_replay is not None


class _Pure(torch.nn.Module):
    def forward(self, value):
        return value * 2 + 1


def test_no_write_still_captures_and_replays_distinct_inputs(device, execution):
    replay = jt.graph_replay(_Pure().eval())
    outputs = []
    for value in (1., 3., 7.):
        with torch.no_grad():
            result = replay(torch.tensor([value, value + 1], device=device))
        _device(result, device)
        outputs.append(_values(result))
    for result, value in zip(outputs, (1., 3., 7.)):
        np.testing.assert_array_equal(result, [2 * value + 1, 2 * value + 3])
    assert replay.refused is None
    assert replay.stats["captured"] >= 1
    assert replay.stats["replayed"] >= 3


class _WarmupError(torch.nn.Module):
    def forward(self, value):
        value += 1
        raise ValueError("warmup sentinel")


def test_warmup_error_propagates_and_restores_observer(device, execution):
    before_context = _ALIAS_WRITE_WARMUPS.get()
    before_keep = jt.flags.keep_graph
    source = torch.tensor([1., 2.], device=device)
    replay = jt.graph_replay(_WarmupError().eval())
    with torch.no_grad(), pytest.raises(ValueError, match="warmup sentinel"):
        replay(source)
    assert _ALIAS_WRITE_WARMUPS.get() == before_context
    assert jt.flags.keep_graph == before_keep
    assert replay.refused is None
    assert replay.stats["captured"] == replay.stats["replayed"] == 0
    np.testing.assert_array_equal(_values(source), [2., 3.])
    # A failed observation must not cause an unrelated pure model to refuse.
    pure = jt.graph_replay(_Pure().eval())
    with torch.no_grad():
        result = pure(torch.tensor([3., 4.], device=device))
    np.testing.assert_array_equal(_values(result), [7., 9.])
    assert pure.refused is None
    assert pure.stats["captured"] >= 1 and pure.stats["replayed"] >= 1


class _DifferentCaptureBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        # Eager warmup sees no write; capture sees one. The existing native
        # guard must reject it instead of silently claiming replay support.
        if self.calls > 1:
            value += 1
        return value * 2


def test_write_only_on_capture_remains_guarded(device, execution):
    model = _DifferentCaptureBranch().eval()
    replay = jt.graph_replay(model)
    with torch.no_grad(), pytest.raises(RuntimeError, match="captured/kept graphs"):
        replay(torch.tensor([1., 2.], device=device))
    assert model.calls == 2
    assert _ALIAS_WRITE_WARMUPS.get() == ()
