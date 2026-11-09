"""Invalid window lengths fail instead of being silently coerced."""

import pytest
import torch


def test_hann_window_rejects_negative_length():
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.hann_window(-1)


def test_hann_window_rejects_non_integer_length():
    with pytest.raises(TypeError, match="int"):
        torch.hann_window(2.5)


@pytest.mark.parametrize("length", [False, True])
def test_hann_window_rejects_boolean_length(length):
    with pytest.raises(TypeError, match="int"):
        torch.hann_window(length)
