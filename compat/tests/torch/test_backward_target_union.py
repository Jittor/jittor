"""Shared optimizer targets still accumulate with independent and retained leaves."""
import os

import numpy as np
import pytest
import torch


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def _array(value):
    return value.detach().cpu().numpy().copy()


def test_shared_optimizer_target_preserves_input_and_retained_gradients(device):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
    independent = torch.tensor([3., 4.], device=device, requires_grad=True)
    first = torch.optim.SGD([parameter], lr=.125)
    second = torch.optim.AdamW([parameter], lr=1e-4)
    previous = None
    for count in (1, 2):
        intermediate = parameter * independent
        intermediate.retain_grad()
        intermediate.square().sum().backward()
        np.testing.assert_array_equal(_array(parameter.grad), [18. * count, 64. * count])
        np.testing.assert_array_equal(_array(independent.grad), [6. * count, 32. * count])
        np.testing.assert_array_equal(_array(intermediate.grad), [6., 16.])
        assert parameter.grad.device.type == device
        if previous is not None:
            assert parameter.grad is previous
        previous = parameter.grad
    # Clearing an optimizer-owned parameter must not clear the unrelated
    # input or the retained non-leaf, and both optimizers see the reset.
    parameter.grad = None
    first.step()
    second.step()
    np.testing.assert_array_equal(_array(parameter), [1., 2.])
    np.testing.assert_array_equal(_array(independent.grad), [12., 64.])
    np.testing.assert_array_equal(_array(intermediate.grad), [6., 16.])
