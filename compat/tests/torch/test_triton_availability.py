"""Torch's Triton capability probe follows the Jittor CUDA bridge."""

import sys

import jittor as jt
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


def test_torch_triton_capability_namespace():
    from torch.utils._triton import has_triton, has_triton_package
    from jittor.compat.triton.backend import is_available, real_triton

    assert torch.cuda.is_available()
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        assert torch.utils._triton is sys.modules["torch.utils._triton"]
        assert isinstance(has_triton_package(), bool)
        assert isinstance(has_triton(), bool)
        assert has_triton_package() == (real_triton() is not None)
        assert has_triton() == is_available()
        assert not has_triton() or (has_triton_package() and jt.has_cuda)
        assert torch.tensor([1.0, 2.0], device="cuda").sum().item() == 3.0
    assert jt.core.backend_fallback_count() == before
