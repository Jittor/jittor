"""Adam shape checks retain scalar/empty updates and live Python overrides."""
import os
from unittest import mock

import numpy as np
import pytest
import torch


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    return request.param


@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3), (2, 1, 3)])
def test_adam_updates_scalar_empty_and_multidimensional_parameters(device, shape):
    param = torch.nn.Parameter(torch.full(shape, .125, device=device))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    expected = np.full(shape, .125, dtype=np.float64)
    moment, variance = np.zeros(shape), np.zeros(shape)
    for step, scale in enumerate((1., -.5, 2., -.25), 1):
        gradient = np.full(shape, .01 * scale, dtype=np.float32)
        param.grad = torch.tensor(gradient, device=device)
        optimizer.step()
        expected *= 1 - 1e-4 * .01
        moment = .9 * moment + .1 * gradient.astype(np.float64)
        variance = .999 * variance + .001 * gradient.astype(np.float64) ** 2
        expected -= 1e-4 * (moment / (1 - .9 ** step)) / (
            np.sqrt(variance / (1 - .999 ** step)) + 1e-8)
        np.testing.assert_allclose(param.detach().cpu().numpy(), expected,
                                   atol=1e-6, rtol=1e-4)
        assert tuple(param.shape) == shape
        assert param.device.type == device
        state = optimizer.state[param]
        assert int(state["step"].item()) == step
        np.testing.assert_allclose(state["exp_avg"].cpu().numpy(), moment,
                                   atol=1e-7, rtol=1e-4)
        np.testing.assert_allclose(state["exp_avg_sq"].cpu().numpy(), variance,
                                   atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("override", ["subclass-shape", "tensor-shape",
                                      "parameter-getattribute"])
def test_adam_observes_shape_overrides_after_validation_warms(device, override):
    class CustomParameter(torch.nn.Parameter):
        pass

    class ShapeError(RuntimeError):
        pass

    cls = CustomParameter if override == "subclass-shape" else torch.nn.Parameter
    param = cls(torch.full((2, 3), .125, device=device))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    param.grad = torch.ones_like(param)
    # Warm normal state validation before changing the descriptor owner.
    from jittor.compat.torch.optimizer_api import _sync_adam_group_params
    _sync_adam_group_params(optimizer.param_groups[0], validate_shapes=True)

    def shape(self):
        raise ShapeError("live shape override")

    def getattribute(self, name):
        if name == "shape":
            raise ShapeError("live shape override")
        return object.__getattribute__(self, name)

    if override == "parameter-getattribute":
        patch = mock.patch.object(torch.nn.Parameter, "__getattribute__", getattribute)
    else:
        owner = torch.Tensor if override == "tensor-shape" else CustomParameter
        patch = mock.patch.object(owner, "shape", property(shape))
    with patch, pytest.raises(ShapeError, match="live shape override"):
        optimizer.step()
    np.testing.assert_array_equal(param.detach().cpu().numpy(), np.full((2, 3), .125))
