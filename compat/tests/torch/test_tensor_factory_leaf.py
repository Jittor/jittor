"""torch.tensor creates fresh leaves; as_tensor preserves conversion history."""
import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestTensorFactoryLeaf:
    def test_dtype_conversion_is_a_leaf_even_when_grad_is_enabled_later(self, device):
        dtypes = (torch.float32, torch.float64)
        if device == "cuda":
            dtypes += (torch.bfloat16,)
        for dtype in dtypes:
            for enabled in (False, True):
                value = torch.tensor([1., 2.], dtype=dtype, device=device, requires_grad=enabled)
                assert value.is_leaf and value.grad_fn is None
                assert value.requires_grad == enabled and value.dtype == dtype
                value.requires_grad_()
                value.square().sum().backward()
                assert value.grad is not None and value.grad.dtype == dtype
                np.testing.assert_array_equal(value.grad.float().cpu().numpy(), [2., 4.])

    def test_tensor_input_copies_and_cuts_history_while_as_tensor_preserves_it(self, device):
        source = torch.tensor([1., 2.], device=device, requires_grad=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float64
        copied = torch.tensor(source, dtype=dtype, requires_grad=True, device=device)
        converted = torch.as_tensor(source, dtype=dtype, device=device)
        assert copied.is_leaf and copied.grad_fn is None
        assert not converted.is_leaf and converted.grad_fn is not None
        assert copied.data_ptr() != source.data_ptr()
        copied.square().sum().backward()
        assert source.grad is None
        np.testing.assert_array_equal(copied.grad.float().cpu().numpy(), [2., 4.])
        converted.sum().backward()
        np.testing.assert_array_equal(source.grad.cpu().numpy(), [1., 1.])


instantiate_device_type_tests(TestTensorFactoryLeaf, globals())
