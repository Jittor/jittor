"""Repeated placement scopes retain reset tokens and context isolation."""
from contextvars import Context, copy_context

import jittor as jt
import numpy as np
import pytest
import torch

from _rng_init_support import require_rng_device


def test_same_placement_token_restores_changes_made_inside_the_scope():
    before = jt.core._current_tensor_placement()
    outer = jt.core._set_tensor_placement(0, 0)
    try:
        # CPU indices normalize to zero, including an equal-value nested entry.
        same = jt.core._set_tensor_placement(0, 9)
        try:
            assert jt.core._current_tensor_placement() == (0, 0)
            # Metadata only: no CUDA operation or hardware claim in this case.
            jt.core._set_tensor_placement(1, 0)
            assert jt.core._current_tensor_placement() == (1, 0)
        finally:
            jt.core._reset_tensor_placement(same)
        assert jt.core._current_tensor_placement() == (0, 0)

        inherited = copy_context()
        independent = Context()

        def changed_context(expected):
            assert jt.core._current_tensor_placement() == expected
            token = jt.core._set_tensor_placement(0, 0)
            try:
                assert jt.core._current_tensor_placement() == (0, 0)
                jt.core._set_tensor_placement(1, 0)
            finally:
                jt.core._reset_tensor_placement(token)
            assert jt.core._current_tensor_placement() == expected

        inherited.run(changed_context, (0, 0))
        independent.run(changed_context, None)
        assert jt.core._current_tensor_placement() == (0, 0)
    finally:
        jt.core._reset_tensor_placement(outer)
    assert jt.core._current_tensor_placement() == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fail", [False, True])
def test_nested_module_restores_placement_after_body_changes_it(device, fail):
    require_rng_device(device)
    expected = (0 if device == "cpu" else 1, 0)
    before = jt.core._current_tensor_placement()

    class Inner(torch.nn.Module):
        def forward(self, value):
            assert jt.core._current_tensor_placement() == expected
            result = value * 2
            # A native caller can change its construction scope. The module's
            # own token must restore it on both return and exception paths.
            jt.core._set_tensor_placement(1 - expected[0], 0)
            if fail:
                raise ValueError("changed placement")
            return result

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, value):
            if fail:
                with pytest.raises(ValueError, match="changed placement"):
                    self.inner(value)
                result = value * 3
            else:
                result = self.inner(value)
            assert jt.core._current_tensor_placement() == expected
            return result + 1

    with jt.flag_scope(auto_graph_replay=0):
        value = torch.tensor([2., 3.], device=device, requires_grad=True)
        result = Outer()(value)
        gradient, = torch.autograd.grad(result.sum(), value)
        for tensor in (result, gradient):
            tensor.sync()
            assert tensor.device.type == device
            assert tensor.location() == ("cpu" if device == "cpu" else "device")
        factor = 3 if fail else 2
        np.testing.assert_array_equal(result.detach().cpu().numpy(), [2 * factor + 1, 3 * factor + 1])
        np.testing.assert_array_equal(gradient.cpu().numpy(), [factor, factor])
    assert jt.core._current_tensor_placement() == before
