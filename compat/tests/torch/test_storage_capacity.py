"""Storage capacity belongs to the backing allocation, independently of views."""
import gc

import numpy as np
import pytest
import jittor as jt
import torch

from jittor._runtime import forbid_backend_fallbacks


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Accelerator device required')
    return request.param


@pytest.mark.parametrize('kind', ['prefix', 'offset', 'strided', 'transpose', 'empty', 'dtype'])
def test_view_retains_full_backing_extent_after_parent_drop(device, kind):
    with forbid_backend_fallbacks():
        whole = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        pointer = whole.untyped_storage().data_ptr()
        if kind == 'prefix':
            view = whole[:1]
        elif kind == 'offset':
            view = whole[1:, 1:]
        elif kind == 'strided':
            view = whole[:, ::2]
        elif kind == 'transpose':
            view = whole.t()
        elif kind == 'empty':
            view = whole[2:2]
        else:
            view = whole.view(torch.uint8)[:1]
        view.sync()
        assert view.device.type == device
        if device == 'cuda' and view.numel():
            assert view.location() == 'device'
        del whole
        gc.collect()
        storage = view.untyped_storage()
        assert storage.nbytes() == 48
        assert storage.data_ptr() == pointer
        assert storage.size() == 48


def test_storage_object_pins_its_original_allocation_across_data_rebind(device):
    with forbid_backend_fallbacks():
        source = torch.arange(12, dtype=torch.float32, device=device)
        storage = source.untyped_storage()
        original_pointer = storage.data_ptr()
        source.data = torch.ones(2, dtype=torch.float32, device=device)
        source.sync()
        assert source.untyped_storage().nbytes() == 8
        assert storage.nbytes() == 48
        assert storage.data_ptr() == original_pointer
        assert source.data_ptr() != original_pointer
        del source
        gc.collect()
        restored = torch.empty(0, dtype=torch.float32, device=device).set_(storage, 0, (12,), (1,))
        np.testing.assert_array_equal(restored.cpu().numpy(), np.arange(12, dtype=np.float32))


def test_typed_and_untyped_storage_sizes_count_different_units(device):
    with forbid_backend_fallbacks():
        source = torch.zeros((3, 4), dtype=torch.float32, device=device)
        view = source[:1]
        assert view.storage().size() == 12
        assert view.storage().nbytes() == 48
        assert view.untyped_storage().size() == 48
        assert view.untyped_storage().nbytes() == 48


def test_independent_empty_tensor_has_zero_storage(device):
    with forbid_backend_fallbacks():
        empty = torch.empty(0, dtype=torch.float32, device=device)
        assert empty.untyped_storage().nbytes() == 0
        assert empty.storage().size() == 0


@pytest.mark.parametrize('kind', ['prefix', 'offset', 'strided', 'transpose'])
def test_storage_byte_image_reads_the_complete_retained_allocation(device, kind):
    with forbid_backend_fallbacks():
        whole = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        view = {'prefix': lambda: whole[:1], 'offset': lambda: whole[1:],
                'strided': lambda: whole[:, ::2], 'transpose': lambda: whole.t()}[kind]()
        storage = view.untyped_storage()
        del view, whole
        gc.collect()
        image = torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (storage.nbytes(),), (1,))
        assert image.device.type == device
        np.testing.assert_array_equal(image.view(torch.float32).cpu().numpy(),
                                      np.arange(12, dtype=np.float32))


@pytest.mark.parametrize('kind', ['float32', 'int8', 'bool', 'int32_view'])
def test_storage_set_defaults_use_the_destination_dtype(device, kind):
    with forbid_backend_fallbacks():
        if kind == 'int32_view':
            array = np.array([1.25, -2.5, 0., 7.75], dtype=np.float32)
            source = torch.tensor(array, device=device)
            dtype = torch.int32
            expected = array.view(np.int32)
        else:
            dtype = getattr(torch, kind)
            array = np.array([-3, 0, 7, -1], dtype=kind)
            source = torch.tensor(array, dtype=dtype, device=device)
            expected = array
        storage = source[:1].untyped_storage()
        result = torch.empty(0, dtype=dtype, device=device).set_(storage)
        assert result.dtype == dtype
        assert tuple(result.shape) == (4,)
        np.testing.assert_array_equal(result.cpu().numpy(), expected)


def test_storage_set_accepts_an_empty_destination_view(device):
    with forbid_backend_fallbacks():
        source = torch.arange(4, dtype=torch.float32, device=device)
        result = torch.empty(0, dtype=torch.float32, device=device).set_(
            source.untyped_storage(), 0, (0,), (1,))
        assert result.numel() == 0
        assert result.untyped_storage().nbytes() == 16


def test_storage_set_reads_a_scalar_from_an_element_offset(device):
    with forbid_backend_fallbacks():
        source = torch.arange(4, dtype=torch.float32, device=device)
        result = torch.empty(0, dtype=torch.float32, device=device).set_(
            source.untyped_storage(), 2, (), ())
        assert tuple(result.shape) == ()
        assert result.item() == 2.


@pytest.mark.cpu
def test_numpy_backed_view_reports_foreign_allocation_extent():
    array = np.arange(12, dtype=np.float32)
    source = torch.from_numpy(array)
    view = source[:2]
    assert view.untyped_storage().nbytes() == array.nbytes
    assert view.untyped_storage().data_ptr() == source.data_ptr()


@pytest.mark.cpu
def test_shared_wrapper_preserves_capacity_without_sfrl():
    with jt.flag_scope(use_cuda=0, use_sfrl_allocator=0), forbid_backend_fallbacks():
        source = torch.arange(13, dtype=torch.float32, device='cpu')
        assert source[:2].untyped_storage().nbytes() == 52
