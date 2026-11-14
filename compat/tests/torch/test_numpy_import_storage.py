"""Writable contiguous NumPy imports retain their real CPU storage owner."""
import gc
import weakref

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int64,
                                  np.uint8, np.complex64])
@jt.flag_scope(backend_fallback='error')
def test_from_numpy_shares_both_mutation_directions_and_retains_owner(dtype):
    array = np.arange(6).astype(dtype).reshape(2, 3)
    owner = weakref.ref(array)
    tensor = torch.from_numpy(array)
    assert tensor.device.type == 'cpu' and not tensor.requires_grad and tensor.is_leaf
    assert tensor.data_ptr() == array.__array_interface__['data'][0]
    array[0, 0] = 13
    assert tensor[0, 0].item() == 13
    tensor[1, 2] = 19
    assert array[1, 2] == 19
    expected = array.copy()
    del array
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(tensor.numpy(), expected)
    view = tensor.numpy()
    del tensor
    gc.collect()
    np.testing.assert_array_equal(view, expected)


@pytest.mark.cuda
@jt.flag_scope(backend_fallback='error')
def test_numpy_cpu_alias_is_copied_to_real_cuda_storage():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    array = np.arange(12, dtype='float32').reshape(3, 4)
    original = array.copy()
    tensor = torch.from_numpy(array).to('cuda')
    torch.cuda.synchronize()
    assert tensor.device.type == 'cuda' and tensor.location() == 'device'
    array[:] = -7
    np.testing.assert_array_equal(tensor.cpu().numpy(), original)
    tensor.mul_(2)
    np.testing.assert_array_equal(array, np.full((3, 4), -7))
