"""High-rank Cast must honor logical strides on the real ACL backend."""
import numpy as np
import pytest
import torch
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.skipif(not getattr(jt.compiler, 'has_acl', False), reason='requires ACL')


@pytest.mark.parametrize('kind', ['dense', 'broadcast', 'transpose', 'reverse_axes'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_acl_high_rank_cast_keeps_shape_values_and_device(kind, dtype):
    data = (np.arange(512, dtype=np.float32) % 64).reshape((2,) * 9)
    with jt.flag_scope(use_acl=1, use_cuda=1, propagate_storage_layout=0,
                       backend_fallback='error'), forbid_backend_fallbacks():
        tensor = torch.tensor(data, device='cuda')
        expected = data
        if kind == 'broadcast':
            tensor = torch.tensor(data[0:1], device='cuda').expand((2,) * 9)
            expected = np.broadcast_to(data[0:1], (2,) * 9)
        elif kind == 'transpose':
            tensor = tensor.transpose(0, 8)
            expected = data.swapaxes(0, 8)
        elif kind == 'reverse_axes':
            tensor = tensor.permute(*reversed(range(9)))
            expected = data.transpose(tuple(reversed(range(9))))
        if kind != 'dense':
            assert not tensor.is_contiguous()
        actual = tensor.to(dtype)
        actual.sync()
        assert actual.location() == 'device'
        assert tuple(actual.shape) == (2,) * 9
        np.testing.assert_array_equal(actual.float().cpu().numpy(), expected)


def test_acl_high_rank_cast_preserves_input_gradient():
    with jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback='error'), forbid_backend_fallbacks():
        source = torch.tensor(np.arange(512, dtype=np.float32).reshape((2,) * 9),
                              device='cuda', requires_grad=True)
        result = source.to(torch.float16)
        gradient, = torch.autograd.grad(result, source, torch.ones_like(result))
        gradient.sync()
        assert gradient.location() == 'device'
        np.testing.assert_array_equal(gradient.cpu().numpy(), np.ones((2,) * 9))


def test_acl_dense_high_rank_pointwise_with_scalar_broadcast():
    with jt.flag_scope(use_acl=1, use_cuda=1, backend_fallback='error'), forbid_backend_fallbacks():
        data = (np.arange(512, dtype=np.float32) % 64).reshape((2,) * 9)
        source = torch.tensor(data, device='cuda')
        actual = source * torch.full_like(source, 2.) + 3.
        actual.sync()
        assert actual.location() == 'device'
        np.testing.assert_array_equal(actual.cpu().numpy(), data * 2 + 3)
