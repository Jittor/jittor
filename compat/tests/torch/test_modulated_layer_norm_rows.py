"""Modulated LayerNorm keeps the low-precision rounding and wide statistics."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device,amplitude", [("cpu", 4.0), ("cuda", 4.0), ("cuda", 1e20)])
def test_modulation_after_layer_norm_with_many_rows(device, amplitude):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    width = 1176
    signs = np.tile(np.array([-1, 1], dtype=np.float32), width // 2)
    values = np.broadcast_to(signs * amplitude, (2, 257, width)).copy()
    with torch.no_grad(), forbid_backend_fallbacks():
        x = torch.tensor(values, dtype=torch.bfloat16, device=device)
        scale = torch.full((2, 1, width), 0.5, dtype=x.dtype, device=device)
        shift = torch.full((2, 1, width), 0.25, dtype=x.dtype, device=device)
        y = F.layer_norm(x, (width,), eps=1e-5) * (1 + scale) + shift
        assert y.device.type == device
        assert y.dtype == x.dtype
        if device == "cuda":
            assert y._producer_name() == "code"
        expected = np.broadcast_to(signs * 1.5 + 0.25, values.shape)
        np.testing.assert_array_equal(y.float().cpu().numpy(), expected)


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 768), (torch.float16, 123)])
def test_modulated_layer_norm_fuses_widening_and_narrowing(dtype, width):
    import jittor as jt

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(34)
    shape = (2, 13, width)
    with torch.no_grad(), forbid_backend_fallbacks():
        x = torch.tensor(rng.normal(size=shape), dtype=dtype, device="cuda")
        packed = torch.tensor(rng.normal(size=(2, 13, 4, width)),
                              dtype=torch.float32, device="cuda")
        x.sync()
        packed.sync()
        scale, shift = packed[:, :, 1, :], packed[:, :, 3, :]

        def modulate(x, scale, shift):
            return (F.layer_norm(x.float(), (width,)) * (1 + scale) + shift).type_as(x)

        run = torch.compile(modulate, mode="reduce-overhead")
        for _ in range(5):
            out = run(x, scale, shift)
            out.sync()
        with jt.profile(device=True, memory=False) as profile:
            out = run(x, scale, shift)
            out.sync()
        assert run.stats["graph"] > 0
        assert out.dtype == dtype and out.location() == "device"
        kernels = [row["name"] for row in profile.result.kernel_records]
        assert sum("mln_" in name for name in kernels) == 1
        assert all("jittor_nvrtc_kernel" not in name for name in kernels)
        x_np = x.float().cpu().numpy()
        scale_np, shift_np = (v.cpu().numpy() for v in (scale, shift))
        ref = (x_np - x_np.mean(-1, keepdims=True)) / np.sqrt(x_np.var(-1, keepdims=True) + 1e-5)
        ref = torch.tensor(ref * (1 + scale_np) + shift_np, dtype=dtype).float().numpy()
        np.testing.assert_allclose(out.float().cpu().numpy(), ref, atol=1e-3, rtol=1e-3)


def test_modulated_layer_norm_cast_preserves_mutation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with torch.no_grad(), forbid_backend_fallbacks():
        x = torch.ones((2, 5, 128), dtype=torch.bfloat16, device="cuda")
        scale = torch.ones(128, dtype=torch.float32, device="cuda")
        shift = torch.ones_like(scale)
        wide = F.layer_norm(x.float(), (128,)) * (1 + scale) + shift
        wide.add_(2)
        np.testing.assert_array_equal(wide.type_as(x).float().cpu().numpy(),
                                      np.full((2, 5, 128), 3, dtype=np.float32))


@pytest.mark.parametrize("width", [123, 768])
def test_modulated_layer_norm_reads_broadcast_add_slices(width):
    import jittor as jt

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(72)
    with torch.no_grad(), forbid_backend_fallbacks():
        def tensor(shape):
            return torch.tensor(rng.normal(size=shape), dtype=torch.bfloat16, device="cuda")

        x = tensor((2, 19, width))
        offsets = tensor((2, 1, 3, width))
        modulation = tensor((2, 19, 3, width))
        jt.sync_all(True)

        def apply(terms):
            scale, shift, gate = terms.unbind(2)
            norm = (F.layer_norm(x.float(), (width,)) * (1 + scale) + shift).type_as(x)
            return (norm * gate).type_as(x)

        def forward():
            return apply(offsets + modulation.float())

        terms = offsets + modulation.float()
        terms.sync()
        reference = apply(terms)
        reference.sync()
        actual = forward()
        actual.sync()
        with jt.profile(device=True, memory=False) as trace:
            actual = forward()
            actual.sync()
        kernels = [row["name"] for row in trace.result.kernel_records]
        assert actual.location() == "device"
        assert sum("mln_" in name for name in kernels) == 1
        assert all(row["out_bytes"] < terms.nbytes for row in trace.result.op_records)
        np.testing.assert_array_equal(actual.float().cpu().numpy(), reference.float().cpu().numpy())
