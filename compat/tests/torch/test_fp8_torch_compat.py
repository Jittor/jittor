"""Torch-compatibility FP8 (float8_e4m3fn, float8_e5m2, float8_e8m0fnu) tests."""

import numpy as np
import pytest
import torch
import safetensors.torch as st


def test_torch_fp8_tensor_creation_view_and_cast_cpu():
    vals = [0.0, 1.0, -2.5, 448.0, 500.0]
    t = torch.tensor(vals, dtype=torch.float8_e4m3fn, device="cpu")
    assert t.dtype is torch.float8_e4m3fn
    assert t.element_size() == 1
    assert t.is_floating_point()

    raw = t.view(torch.uint8)
    assert raw.dtype is torch.uint8
    np.testing.assert_array_equal(
        raw.numpy(),
        np.array([0x00, 0x38, 0xC2, 0x7E, 0x7E], dtype=np.uint8),
    )

    back = raw.view(torch.float8_e4m3fn)
    assert back.dtype is torch.float8_e4m3fn
    np.testing.assert_allclose(back.to(torch.float32).numpy(), [0.0, 1.0, -2.5, 448.0, 448.0])

    t_e5 = torch.tensor([0.0, 1.0, -2.5, 512.0], dtype=torch.float8_e5m2, device="cpu")
    assert t_e5.dtype is torch.float8_e5m2
    np.testing.assert_allclose(t_e5.to(torch.float32).numpy(), [0.0, 1.0, -2.5, 512.0])

    # E8M0FNU power-of-two scale view
    s_u8 = torch.tensor([126, 127, 128], dtype=torch.uint8, device="cpu")
    s_e8 = s_u8.view(torch.float8_e8m0fnu)
    assert s_e8.dtype is torch.float8_e8m0fnu
    np.testing.assert_allclose(s_e8.to(torch.float32).numpy(), [0.5, 1.0, 2.0])


def test_torch_fp8_safetensors_roundtrip_and_scaled_mm_cpu():
    rng = np.random.default_rng(7)
    a_np = rng.standard_normal((8, 64)).astype(np.float32) * 0.5
    b_np = rng.standard_normal((32, 64)).astype(np.float32) * 0.5
    sa_np = (np.max(np.abs(a_np), axis=1, keepdims=True) / 448.0).astype(np.float32)
    sb_np = (np.max(np.abs(b_np), axis=1, keepdims=True) / 448.0).astype(np.float32)

    a = torch.tensor(a_np / sa_np, dtype=torch.float8_e4m3fn, device="cpu")
    b = torch.tensor(b_np / sb_np, dtype=torch.float8_e4m3fn, device="cpu")
    sa = torch.tensor(sa_np, dtype=torch.float32, device="cpu")
    sb = torch.tensor(sb_np.T, dtype=torch.float32, device="cpu")
    bias = torch.tensor(np.ones((32,), dtype=np.float32) * 0.25, dtype=torch.bfloat16, device="cpu")

    # Safetensors byte round-trip preserves exact bits and FP8 dtypes
    payload = st.save({"a": a, "b": b, "e8": torch.tensor([127, 128], dtype=torch.uint8).view(torch.float8_e8m0fnu)})
    loaded = st.load(payload)
    assert loaded["a"].dtype is torch.float8_e4m3fn
    assert loaded["b"].dtype is torch.float8_e4m3fn
    assert loaded["e8"].dtype is torch.float8_e8m0fnu
    np.testing.assert_array_equal(loaded["a"].view(torch.uint8).numpy(), a.view(torch.uint8).numpy())
    np.testing.assert_array_equal(loaded["e8"].view(torch.uint8).numpy(), np.array([127, 128], dtype=np.uint8))

    out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, bias=bias, out_dtype=torch.bfloat16)
    assert out.dtype is torch.bfloat16
    assert tuple(out.shape) == (8, 32)
    ref = a_np @ b_np.T + 0.25
    out_f = out.to(torch.float32).numpy()
    cos = float(np.dot(out_f.ravel(), ref.ravel()) / (np.linalg.norm(out_f) * np.linalg.norm(ref)))
    assert cos > 0.998


def test_torch_scaled_mm_device_if_acl_fp8():
    import jittor as jt
    from jittor.backends.acl.kernels._soc import supports_fp8
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if not (getattr(jt.compiler, "has_acl", False) and supports_fp8()):
        pytest.skip("requires Ascend950-class NPU with ACL backend")

    rng = np.random.default_rng(19)
    M, K, N = 16, 128, 64
    a_np = rng.standard_normal((M, K)).astype(np.float32) * 0.5
    b_np = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    sa_np = (np.max(np.abs(a_np), axis=1, keepdims=True) / 448.0).astype(np.float32)
    sb_np = (np.max(np.abs(b_np), axis=1, keepdims=True) / 448.0).astype(np.float32)

    with forbid_backend_fallbacks():
        before = jt.core.backend_fallback_count()
        a = torch.tensor(a_np / sa_np, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        b = torch.tensor(b_np / sb_np, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        sa = torch.tensor(sa_np, dtype=torch.float32, device="cuda")
        sb = torch.tensor(sb_np.T, dtype=torch.float32, device="cuda")
        out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        out.sync()
        assert jt.core.backend_fallback_count() == before
        assert out.dtype is torch.bfloat16
        ref = a_np @ b_np.T
        out_f = out.to(torch.float32).numpy()
        cos = float(np.dot(out_f.ravel(), ref.ravel()) / (np.linalg.norm(out_f) * np.linalg.norm(ref)))
        assert cos > 0.998
