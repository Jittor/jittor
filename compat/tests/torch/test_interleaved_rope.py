import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.dispatch import override_kernel
from jittor._runtime.fallback import forbid_backend_fallbacks


def _rotate(x):
    real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((-imag, real), dim=-1)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_stock_interleaved_rope_expression_dispatches(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def apply(x, cos, sin):
        rotated = _rotate(x).flatten(-2)
        wide = x.float() * cos + rotated.float() * sin
        out = wide.to(x.dtype)
        if x.dtype == torch.float32:
            assert out is wide
        return out

    with torch.no_grad(), jt.flag_scope(cuda_kernel_math="strict"):
        x = torch.randn(2, 17, 3, 18, device="cuda", dtype=dtype).transpose(1, 2)
        cos = torch.randn(1, 1, 17, 18, device="cuda")
        sin = torch.randn(2, 1, 1, 18, device="cuda")
        actual = apply(x, cos, sin)
        assert actual._producer_name() == "code"
        with override_kernel("nn.interleaved_rope.inference", "cuda", None):
            expected = apply(x, cos, sin)
        np.testing.assert_array_equal(actual.float().numpy(), expected.float().numpy())


@pytest.mark.cuda
def test_interleaved_rope_keeps_writes_to_the_rotated_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with torch.no_grad():
        x = torch.randn(2, 7, 12, dtype=torch.bfloat16, device="cuda")
        rotated = _rotate(x).flatten(-2)
        rotated.fill_(3)
        cos = torch.zeros((7, 12), device="cuda")
        sin = torch.ones((7, 12), device="cuda")
        actual = (x.float() * cos + rotated.float() * sin).to(x.dtype)
        np.testing.assert_array_equal(actual.float().numpy(), 3)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_slice_pair_rotation_preserves_each_product_rounding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(28)

    def rotate(x, cos, sin):
        real, imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = cos[..., ::2], sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = real * cos - imag * sin
        out[..., 1::2] = real * sin + imag * cos
        return out

    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(cuda_kernel_math="strict"):
        def random_tensor(shape):
            return torch.tensor(rng.normal(size=shape), device=device, dtype=torch.bfloat16)

        cos, sin = (random_tensor((1, 11, 1, 24)) for _ in range(2))
        run = torch.compile(rotate, mode="reduce-overhead") if device == "cuda" else rotate
        for _ in range(5):
            x = random_tensor((2, 11, 3, 24))
            with override_kernel("nn.interleaved_rope.inference", "cuda", None):
                expected = rotate(x, cos, sin)
            actual = run(x, cos, sin)
            assert actual.device.type == device and actual.dtype == x.dtype
            np.testing.assert_array_equal(actual.float().cpu().numpy(), expected.float().cpu().numpy())
        if device == "cuda":
            with jt.profile(device=True, memory=False) as trace:
                run(x, cos, sin).sync()
            assert any("interleaved_rope" in row["name"] for row in trace.result.kernel_records)
