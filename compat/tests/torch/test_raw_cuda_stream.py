"""Raw CUDA stream handoff to native Torch ecosystem kernels."""

import pytest

from _helpers.checkpoint_cuda import accelerator

import jittor as jt
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


def test_raw_cuda_stream_is_the_native_compute_stream():
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        assert torch.cuda.is_available()
        assert jt.core.backend_device_count("cuda") > 0
        assert torch._C._cuda_getCurrentRawStream(0) == 0x2
        assert torch._C._cuda_getCurrentRawStream(-1) == 0x2
        with pytest.raises(ValueError, match="out of range"):
            torch._C._cuda_getCurrentRawStream(jt.core.backend_device_count("cuda"))
        with pytest.raises(TypeError, match="device index"):
            torch._C._cuda_getCurrentRawStream("cuda")
        assert torch.tensor([1., 2.], device="cuda").sum().item() == 3.
        assert jt.core.device_pool_stats(0)[2] > 0
    assert jt.core.backend_fallback_count() == before
