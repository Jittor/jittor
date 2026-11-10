"""Unavailable MPS detection must match the PyTorch exception contract."""
import pytest
import torch


def test_mps_memory_probe_without_backend():
    if torch.backends.mps.is_available():
        pytest.skip("requires a runtime without MPS")
    with pytest.raises(RuntimeError):
        torch.mps.current_allocated_memory()
