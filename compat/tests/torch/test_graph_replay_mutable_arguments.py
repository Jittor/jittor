"""Replay warmup must not execute mutable Python cache updates twice."""
import numpy as np
import pytest
import torch
import jittor as jt

from jittor._runtime.graph_replay import graph_replay


@pytest.fixture(autouse=True)
def cpu_policy():
    with jt.flag_scope(use_cuda=0, use_acl=0, auto_graph_replay=1), torch.no_grad():
        yield


class CursorModule(torch.nn.Module):
    def forward(self, x, *, cache, cursor):
        index = cursor[0]
        result = x + index + 1
        cache[index] = result
        cursor[0] += 1
        return result


@pytest.mark.parametrize("automatic", [False, True])
def test_mutating_cursor_and_cache_are_updated_once_per_public_call(automatic):
    module = CursorModule().eval()
    call = module if automatic else graph_replay(module)
    for index in range(4):
        x = torch.full((2, 3), float(index), device="cpu")
        cache, cursor = [None], [0]
        output = call(x, cache=cache, cursor=cursor)
        np.testing.assert_array_equal(output.numpy(), np.full((2, 3), index + 1, dtype=np.float32))
        assert cursor == [1]
        assert cache[0] is output
    if not automatic:
        assert "modified Python argument state" in call.refused
        assert call.stats["captured"] == 0


def test_rebinding_a_same_shape_tensor_in_a_nested_cache_refuses_replay():
    class UpdateCache(torch.nn.Module):
        def forward(self, x, cache):
            result = x + cache["values"][0]
            cache["values"][0] = result
            return result
    call = graph_replay(UpdateCache().eval())
    cache = {"values": [torch.zeros((2, 3), device="cpu")]}
    expected = np.zeros((2, 3), dtype=np.float32)
    for index in range(3):
        x = torch.full((2, 3), float(index + 1), device="cpu")
        expected += index + 1
        output = call(x, cache)
        np.testing.assert_array_equal(output.numpy(), expected)
        assert cache["values"][0] is output
    assert "modified Python argument state" in call.refused


def test_read_only_nested_containers_still_capture_and_replay():
    class ReadOnly(torch.nn.Module):
        def forward(self, x, extras):
            return x + extras["bias"][0] + extras["offsets"][0]
    call = graph_replay(ReadOnly().eval())
    extras = {"bias": [torch.ones((2, 3), device="cpu")], "offsets": [2.0]}
    bias = extras["bias"][0]
    for index in range(3):
        output = call(torch.full((2, 3), float(index), device="cpu"), extras)
        np.testing.assert_array_equal(output.numpy(), np.full((2, 3), index + 3, dtype=np.float32))
        assert extras["bias"][0] is bias and extras["offsets"] == [2.0]
    assert call.refused is None
    assert call.stats["captured"] >= 1 and call.stats["replayed"] >= 1
