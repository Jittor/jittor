"""Backend guard shared by portable optimizer contracts and real Torch oracles."""

import sys

import pytest
import torch


@pytest.fixture(autouse=True)
def require_optimizer_backend():
    if not hasattr(torch, '_torch_compat_install_context'):
        yield
        return
    from jittor._runtime.fallback import forbid_backend_fallbacks
    with forbid_backend_fallbacks():
        yield
        sys.modules['jittor'].sync_all(True)

