"""Torch any/all keepdim applies to a full reduction as well as a named axis."""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("operation", ["any", "all"])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", ["bool", "float32"])
@pytest.mark.parametrize("method", [False, True])
def test_full_truth_reduction_keeps_requested_shape(device, operation, keepdim, dtype, method):
    if device == "npu":
        require_accelerator("acl")
    values = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=dtype)
    if dtype == "float32":
        values[1, 1] = np.nan
    expected = getattr(np, operation)(values, keepdims=keepdim)
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        source = torch.tensor(values, device=device)
        output = (getattr(source, operation)(dim=None, keepdim=keepdim) if method else
                  getattr(torch, operation)(source, dim=None, keepdim=keepdim))
        jt.sync_all(True)
        assert tuple(output.shape) == expected.shape
        assert output.dtype == torch.bool
        assert output.device.type == device
        np.testing.assert_array_equal(output.cpu().numpy(), expected)
