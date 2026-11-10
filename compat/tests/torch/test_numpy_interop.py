"""The explicit NumPy host view retains allocation ownership without copying."""
import gc
from unittest.mock import patch

import numpy as np
import pytest
import torch

from jittor.compat.torch.numpy_interop import numpy_view


def test_cpu_view_exports_current_allocation_and_preserves_strides():
    tensor = torch.arange(12, dtype=torch.int64, device="cpu")
    view = numpy_view(tensor)
    sliced = view[2:10:2]
    assert np.shares_memory(view, sliced)
    assert sliced.strides == (16,)
    assert sliced.__array_interface__["data"][0] == tensor.data_ptr() + 16
    sliced[0] = 71
    assert tensor.tolist()[2] == 71
    tensor[4] = 82
    assert tensor.tolist()[4] == 82
    current = numpy_view(tensor)
    assert current[4] == 82
    assert current.__array_interface__["data"][0] == tensor.data_ptr()
    assert current[10:].tolist() == [10, 11]


def test_view_owns_allocation_after_tensor_wrapper_is_released():
    tensor = torch.tensor([11, 22, 33], dtype=torch.int64, device="cpu")
    view = numpy_view(tensor)
    pointer = tensor.data_ptr()
    del tensor
    gc.collect()
    assert view.__array_interface__["data"][0] == pointer
    assert view.tolist() == [11, 22, 33]
    view[1] = 44
    assert view.tolist() == [11, 44, 33]
    del view
    gc.collect()


def test_storage_replacement_keeps_old_view_and_requires_reacquisition():
    tensor = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    old_view = numpy_view(tensor)
    tensor.data = torch.tensor([4, 5, 6], dtype=torch.int64, device="cpu")
    new_view = numpy_view(tensor)
    assert not np.shares_memory(old_view, new_view)
    old_view[0] = 19
    assert tensor.tolist() == [4, 5, 6]
    tensor[1] = 29
    assert tensor.tolist() == [4, 29, 6]
    # A lazy Tensor write may replace storage again: reacquire the current
    # allocation as the public API requires, rather than assuming eager writes.
    current_view = numpy_view(tensor)
    assert current_view.tolist() == [4, 29, 6]
    assert current_view.__array_interface__["data"][0] == tensor.data_ptr()
    assert old_view.tolist() == [19, 2, 3]


def test_invalid_type_and_grad_are_rejected():
    with pytest.raises(TypeError, match="Jittor tensor"):
        numpy_view(np.array([1]))
    with pytest.raises(RuntimeError, match="gradients"):
        numpy_view(torch.tensor([1.0], device="cpu", requires_grad=True))


def test_non_cpu_guard_is_fail_closed_without_calling_storage_export():
    # This is a control-flow negative fixture, not a CUDA execution result.
    tensor = torch.tensor([1], dtype=torch.int64, device="cpu")
    with patch("jittor.compat.torch.types._var_is_cpu_resident", return_value=False):
        with pytest.raises(TypeError, match="CPU storage"):
            numpy_view(tensor)
