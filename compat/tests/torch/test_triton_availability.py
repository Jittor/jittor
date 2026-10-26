"""Torch's Triton capability probe follows the Jittor CUDA bridge."""

import sys

import unittest

from _helpers.device_types import instantiate_device_type_tests

import jittor as jt
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


class TestTritonAvailability(unittest.TestCase):
    def test_torch_triton_capability_namespace(self, device):
        from torch.utils._triton import has_triton, has_triton_package
        from jittor.compat.triton.backend import is_available, real_triton

        before = jt.core.backend_fallback_count()
        with jt.flag_scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
            assert torch.utils._triton is sys.modules["torch.utils._triton"]
            assert isinstance(has_triton_package(), bool)
            assert isinstance(has_triton(), bool)
            assert has_triton_package() == (real_triton() is not None)
            assert has_triton() == is_available()
            assert not has_triton() or (has_triton_package() and jt.has_cuda)
            if device == "cuda":
                assert torch.cuda.is_available()
                assert torch.tensor([1.0, 2.0], device=device).sum().item() == 3.0
        assert jt.core.backend_fallback_count() == before


instantiate_device_type_tests(TestTritonAvailability, globals(), only_for=("cpu", "cuda"))
