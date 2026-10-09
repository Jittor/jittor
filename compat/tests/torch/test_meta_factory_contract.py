"""Explicit meta placeholders used while Transformers scans safetensors headers."""

import pytest
import torch


@pytest.mark.parametrize("factory", [torch.empty, torch.zeros])
@pytest.mark.parametrize("requested", ["meta", torch.device("meta")])
def test_explicit_meta_factory_keeps_metadata_after_construction(factory, requested):
    value = factory((2, 3), dtype=torch.float32, device=requested)
    assert tuple(value.shape) == (2, 3)
    assert value.dtype == torch.float32
    assert value.device == torch.device("meta")
    assert value.get_device() == -1
    assert torch.zeros((1,), device="cpu").device.type == "cpu"