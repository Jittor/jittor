"""Independent Torch adaptive pooling must execute CANN forward and backward."""

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


_CASES = [
    pytest.param((1, 2, 5, 7), (3, 4), id="overlapping_bins"),
    pytest.param((1, 2, 2, 3), (4, 5), id="upsampling"),
    pytest.param((1, 2, 5, 7), (None, 3), id="preserve_height"),
    pytest.param((2, 5, 7), (3, None), id="unbatched_preserve_width"),
]


def _reference(source, output_size, cotangent):
    height, width = source.shape[-2:]
    oh, ow = cotangent.shape[-2:]
    output = np.empty(cotangent.shape, dtype=np.float32)
    gradient = np.zeros(source.shape, dtype=np.float32)
    for y in range(oh):
        y0, y1 = y * height // oh, ((y + 1) * height + oh - 1) // oh
        for x in range(ow):
            x0, x1 = x * width // ow, ((x + 1) * width + ow - 1) // ow
            output[..., y, x] = source[..., y0:y1, x0:x1].mean(axis=(-2, -1))
            gradient[..., y0:y1, x0:x1] += (
                cotangent[..., y, x, None, None] / ((y1 - y0) * (x1 - x0)))
    return output, gradient


@pytest.mark.npu
@pytest.mark.parametrize("shape,output_size", _CASES)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_adaptive_avg_pool2d_torch_forward_backward(shape, output_size, dtype):
    require_accelerator("acl")
    assert torch is not jt and torch.Tensor is not jt.Var
    assert torch.Tensor._frontend_backend is jt
    source_np = ((np.arange(np.prod(shape)) % 17) - 8).astype(np.float32).reshape(shape) / 8
    spatial = tuple(size if size is not None else shape[-2 + i]
                    for i, size in enumerate(output_size))
    output_shape = shape[:-2] + spatial
    cotangent_np = ((np.arange(np.prod(output_shape)) % 7) - 3).astype(np.float32).reshape(output_shape) / 8
    expected, expected_gradient = _reference(source_np, output_size, cotangent_np)
    with jt.flag_scope(use_acl=1, use_cuda=1), forbid_backend_fallbacks():
        source = torch.tensor(source_np, device="npu", dtype=getattr(torch, dtype), requires_grad=True)
        output = torch.nn.functional.adaptive_avg_pool2d(source, output_size)
        cotangent = torch.tensor(cotangent_np, device="npu", dtype=getattr(torch, dtype))
        gradient, = torch.autograd.grad((output * cotangent).sum(), source)
        for value in (output, gradient):
            assert type(value) is torch.Tensor
            assert value.dtype == getattr(torch, dtype)
            value.sync()
            assert value.location() == "device"
            assert value.placement_backend == 2 and value.device_id >= 0
        actual, actual_gradient = (value.detach().float().cpu().numpy() for value in (output, gradient))
    # Inputs and cotangents are exactly representable; pooled results and
    # overlapping gradient sums still round to the output dtype.
    tolerance = {"float32": 2e-6, "float16": 2e-3, "bfloat16": 2e-2}[dtype]
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance)
