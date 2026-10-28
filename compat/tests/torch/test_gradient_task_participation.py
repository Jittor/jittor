"""Public runtime query consumes native results, never stale gradient storage."""
import pytest
import torch

from jittor.compat.torch.gradient_task import participating_tensors
from _helpers.device_types import instantiate_device_type_tests


class TestGradientTaskParticipation:
    def test_current_targets_aliases_and_disconnected_stale_gradient(self, device):
        first = torch.tensor([2.], device=device, requires_grad=True)
        second = torch.tensor([3.], device=device, requires_grad=True)
        second.square().sum().backward()
        assert second.grad is not None
        observations = []
        first.register_post_accumulate_grad_hook(
            lambda parameter: observations.append(participating_tensors((second, first, first, object()))))
        first.square().sum().backward()
        assert len(observations) == 1 and len(observations[0]) == 1
        assert observations[0][0] is first
        with pytest.raises(RuntimeError, match="post-accumulate"):
            participating_tensors((first, second))

    def test_hook_failure_clears_participation(self, device):
        value = torch.tensor([2.], device=device, requires_grad=True)
        def failure(parameter):
            assert participating_tensors((value,)) == (value,)
            raise RuntimeError("post hook failed")
        handle = value.register_post_accumulate_grad_hook(failure)
        with pytest.raises(RuntimeError, match="post hook failed"):
            value.square().sum().backward()
        with pytest.raises(RuntimeError, match="post-accumulate"):
            participating_tensors((value,))
        handle.remove()
        value.square().sum().backward()


instantiate_device_type_tests(TestGradientTaskParticipation, globals(), only_for=("cpu", "cuda"))
