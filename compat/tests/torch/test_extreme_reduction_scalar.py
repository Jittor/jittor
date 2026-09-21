"""Full extrema reductions preserve scalar rank and distribute tied gradients."""

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("name", ["amax", "amin"])
@pytest.mark.parametrize("keepdim", [False, True])
def test_full_extreme_scalar_shape_and_gradient(device, name, keepdim):
    if device == "npu":
        require_accelerator("acl")
    assert torch is not jt and torch.Tensor is not jt.Var
    values = np.array([[2., 2.], [-1., -1.]], dtype=np.float32)
    expected_gradient = (values == getattr(np, "max" if name == "amax" else "min")(values)) / 2
    with jt.flag_scope(use_acl=int(device == "npu"), use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        source = torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)
        output = getattr(torch, name)(source, keepdim=keepdim)
        gradient, = torch.autograd.grad(output, source)
        assert tuple(output.shape) == ((1, 1) if keepdim else ())
        output.sync()
        gradient.sync()
        if device == "npu":
            assert output.location() == gradient.location() == "device"
        np.testing.assert_array_equal(output.detach().cpu().numpy(),
                                      getattr(np, "max" if name == "amax" else "min")(
                                          values, keepdims=keepdim))
        np.testing.assert_array_equal(gradient.detach().cpu().numpy(), expected_gradient)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_full_extreme_reduction_reports_the_infinity_it_was_given(dtype):
    """The identity contract for this frontend, which is a separate session.

    `tests/ops/test_minmax_reduction_identity.py` owns this contract, but that
    path runs native-only -- `process_modes.TORCH_MODE_PATHS` does not include
    it -- and the reduce kernel is reached through different names here
    (`Tensor.max` and `torch.amax` on the shim). The fold starts from the
    dtype's identity, so a frontend that re-narrowed it to the lowest finite
    value would answer -3.4e38 for a tensor whose maximum really is -inf and
    nothing in the native file would notice. Native and shim were measured to
    agree on 2026-09-21; this is that measurement as a gate.
    """
    with forbid_backend_fallbacks():
        negative = torch.tensor(np.array([-np.inf] * 3, dtype=dtype))
        positive = torch.tensor(np.array([np.inf] * 3, dtype=dtype))
        assert float(negative.max()) == -np.inf
        assert float(positive.min()) == np.inf
        assert float(torch.amax(negative)) == -np.inf
        assert float(torch.amin(positive)) == np.inf
