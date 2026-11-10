"""PyTorch's default pickle protocol is part of the serialization API."""

import torch


def test_default_protocol_matches_torch():
    assert torch.serialization.DEFAULT_PROTOCOL == 2
