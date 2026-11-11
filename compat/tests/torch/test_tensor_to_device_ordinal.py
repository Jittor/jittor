"""Integer device targets use the same Tensor.to placement path in both spellings."""

import numpy as np
import pytest
import jittor as jt
import torch
from jittor._runtime import forbid_backend_fallbacks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Accelerator device required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("keyword", [False, True])
def test_to_device_ordinal_keeps_source_and_moves_storage(dtype, keyword):
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        source = torch.ones((2, 2), dtype=dtype, device="cpu")
        result = source.to(device=0) if keyword else source.to(0)
        result.sync()
        assert source.device.type == "cpu"
        assert result.device == torch.device("cuda", 0)
        assert result.location() == "device"
        assert result.dtype == dtype
        np.testing.assert_array_equal(result.cpu().numpy(), np.ones((2, 2)))
    assert jt.core.backend_fallback_count() == before


def test_negative_keyword_ordinal_is_rejected():
    source = torch.ones(2, device="cpu")
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        source.to(device=-1)
