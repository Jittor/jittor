"""Converting a Tensor to Python values must preserve the original device."""

from contextlib import ExitStack
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("kind", ["matrix", "scalar", "view"])
def test_tolist_preserves_original_storage_device_and_gradient(device, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with ExitStack() as stack:
        jt = None
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        raw = np.array(2., dtype=np.float32) if kind == "scalar" else np.arange(6, dtype=np.float32).reshape(2, 3)
        source = torch.tensor(raw, device=device, requires_grad=True)
        value = source[1] if kind == "view" else source
        expected = raw[1] if kind == "view" else raw
        if jt is not None:
            value.sync()
            source.sync()
            assert value.location() == source.location() == ("device" if device == "cuda" else "cpu")
        actual = value.tolist()
        assert actual == expected.tolist()
        assert source.device.type == value.device.type == device
        if jt is not None:
            assert value.location() == source.location() == ("device" if device == "cuda" else "cpu")
            assert jt.core.dispatch_context([value])[0] == device
        source.sum().backward()
        np.testing.assert_array_equal(source.grad.detach().cpu().numpy(), np.ones_like(raw))
        if jt is not None:
            assert jt.core.backend_fallback_count() == before
