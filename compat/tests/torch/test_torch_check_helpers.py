"""torch._check_with parity for caller-selected exception types."""

import pytest
import torch


def test_check_with_true_does_not_raise():
    torch._check_with(ValueError, True, lambda: "unused")


def test_check_with_raises_the_selected_type_with_lazy_message():
    with pytest.raises(ValueError, match="expected square"):
        torch._check_with(ValueError, False, lambda: "expected square")
    with pytest.raises(TypeError):
        torch._check_with(TypeError, False, "plain message")


def test_check_with_default_message():
    with pytest.raises(RuntimeError, match="Expected cond to be True"):
        torch._check_with(RuntimeError, False)
