"""Bounded Tensor.random_ overloads used by training data generators."""
import numpy as np
import pytest
import torch


def test_random_inplace_accepts_to_and_from_to_bounds():
    below = torch.empty(256, dtype=torch.long).random_(7).numpy()
    ranged = torch.empty(256, dtype=torch.long).random_(3, 11).numpy()
    keyword = torch.empty(256, dtype=torch.long).random_(**{"from": 4, "to": 9}).numpy()

    assert np.all((0 <= below) & (below < 7))
    assert np.all((3 <= ranged) & (ranged < 11))
    assert np.all((4 <= keyword) & (keyword < 9))


def test_random_inplace_rejects_invalid_bounds():
    with pytest.raises(RuntimeError, match="less than"):
        torch.empty(1, dtype=torch.long).random_(5, 5)
