"""Trailing full slices retain advanced-index semantics and rank errors."""
import numpy as np
import pytest
import torch


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires an accelerator')
    return request.param


@pytest.mark.parametrize('kind', ['mask1', 'mask2', 'integer'])
def test_advanced_index_trailing_slices_get_and_set(device, kind):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    if kind == 'mask2':
        index = np.array([[True, False, True], [False, True, False]])
        tail = (slice(None),)
    else:
        index = np.array([True, False]) if kind == 'mask1' else np.array([[1, 0], [0, 1]])
        tail = (slice(None), slice(None))
    tensor = torch.tensor(data, device=device)
    key = (torch.tensor(index, device=device),) + tail
    np.testing.assert_array_equal(tensor[key].cpu().numpy(), data[(index,) + tail])
    tensor[key] = -2
    data[(index,) + tail] = -2
    np.testing.assert_array_equal(tensor.cpu().numpy(), data)


@pytest.mark.parametrize('kind', ['scalar', 'integer', 'mask1', 'mask2'])
def test_excess_full_slices_raise_before_get_or_set(device, kind):
    tensor = torch.zeros((2, 3), device=device)
    index = {'scalar': 0, 'integer': [0], 'mask1': [True, False],
             'mask2': [[True, False, True], [False, True, False]]}[kind]
    if kind != 'scalar':
        index = torch.tensor(index, device=device)
    key = (index, slice(None), slice(None))
    with pytest.raises(IndexError):
        tensor[key]
    with pytest.raises(IndexError):
        tensor[key] = 1
    np.testing.assert_array_equal(tensor.cpu().numpy(), np.zeros((2, 3)))


def test_trailing_slice_mask_assignment_keeps_rhs_gradient(device):
    hidden = torch.zeros((2, 3, 4), device=device)
    mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
    rhs = torch.tensor([1., 2., 3., 4.], device=device, requires_grad=True)
    hidden[mask, :] = rhs
    hidden.sum().backward()
    np.testing.assert_array_equal(rhs.grad.cpu().numpy(), np.full(4, 3.))


@pytest.mark.parametrize('empty', [False, True])
def test_acl_native_prefix_mask_assignment_and_gradient(empty):
    import jittor as jt
    if not jt.compiler.has_acl:
        pytest.skip('requires ACL')
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros((2, 3), dtype=bool) if empty else np.array(
        [[True, False, True], [False, True, False]])
    source = torch.tensor(data, device='cuda', requires_grad=True)
    rhs = torch.tensor([10., 20., 30., 40.], device='cuda', requires_grad=True)
    # Exercise the native backend Function directly, without the compat
    # differentiable mask shortcut used by __setitem__.
    out = source.setitem(torch.tensor(mask, device='cuda'), rhs)
    out.sum().backward()
    expected = data.copy()
    expected[mask] = np.array([10., 20., 30., 40.])
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
    np.testing.assert_array_equal(rhs.grad.cpu().numpy(), np.full(4, mask.sum()))
    expected_grad = np.ones_like(data)
    expected_grad[mask] = 0
    np.testing.assert_array_equal(source.grad.cpu().numpy(), expected_grad)
