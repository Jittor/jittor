"""Unsupported antialiasing must not silently execute ordinary interpolation."""

import jittor as jt
import pytest
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", "acl"])
@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_interpolate_antialias_is_explicitly_unsupported(device, mode):
    if device == "acl" and not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=int(device == "acl"), use_acl=int(device == "acl")), forbid_backend_fallbacks(), torch.no_grad():
        x = torch.ones((1, 2, 4, 5), device="cuda" if device == "acl" else "cpu")
        with pytest.raises(NotImplementedError, match="antialias=True.*not implemented"):
            torch.nn.functional.interpolate(x, size=(2, 3), mode=mode, align_corners=False, antialias=True)
        # Ordinary interpolation remains usable; identity keeps an independent
        # oracle on CPU as well as the actual accelerator path.
        for kwargs in ({}, {"antialias": False}):
            result = torch.nn.functional.interpolate(x, size=(4, 5), mode=mode, align_corners=False, **kwargs)
            jt.sync_all(True)
            assert result.device.type == ("npu" if device == "acl" else "cpu")
            assert bool((result == x).all().item())
    assert jt.core.backend_fallback_count() == before
