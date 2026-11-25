"""ROCm library detection agrees with real HIP execution."""

import re

import numpy as np
import pytest
import jittor as jt
import torch
from jittor._runtime import forbid_backend_fallbacks


pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason="ROCm build required")


def test_hip_identity_with_native_device_execution():
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_rocm=1, use_parallel_op_compiler=0):
        with forbid_backend_fallbacks():
            value = jt.array([1.0, 2.0]) + 1
            value.sync()
            assert value.location() == "device"
            np.testing.assert_array_equal(value.numpy(), [2.0, 3.0])
        assert torch.cuda.is_available()
        assert isinstance(torch.version.hip, str)
        assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", torch.version.hip)
        assert torch.version.cuda is None
    assert jt.core.backend_fallback_count() == before
