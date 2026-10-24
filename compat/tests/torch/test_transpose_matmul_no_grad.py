"""Public matmul must preserve no-grad transpose alias boundaries."""
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from _rng_init_support import _array, preserve_random_streams, require_rng_device


@pytest.fixture(autouse=True)
def no_backend_fallback():
    with preserve_random_streams():
        yield


def resident(value, device):
    assert value.device.type == device
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        value.sync()
        assert jt.core.dispatch_context([value])[0] == device
        assert value.location() == ('device' if device == 'cuda' else 'cpu')
        assert jt.core.backend_fallback_count() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(3,4), (2,3,4), (1,2,1,3,4)])
@pytest.mark.parametrize('key_grad', [False, True])
@pytest.mark.parametrize('no_grad', [False, True])
@pytest.mark.parametrize('variant', ['plain', 'contiguous', 'same_dtype', 'roundtrip'])
def test_transpose_matmul_flags_values_and_full_vjp(device, shape, key_grad, no_grad, variant):
    require_rng_device(device)
    key_values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)/16
    left_shape=shape[:-2]+(2,shape[-1])
    left_values=(np.arange(np.prod(left_shape),dtype=np.float32).reshape(left_shape)+1)/8
    key=torch.tensor(key_values,device=device,requires_grad=key_grad)
    left=torch.tensor(left_values,device=device,requires_grad=True)
    with torch.no_grad() if no_grad else nullcontext():
        source=key.reshape(shape)
        view=source.transpose(-1,-2)
        if variant=='contiguous': view=view.contiguous()
        elif variant=='same_dtype': view=view.to(torch.float32)
        elif variant=='roundtrip': view=view.to(torch.float64).to(torch.float32)
        # No sync, snapshot, producer inspection or graph rewrite before matmul.
        result=left@view
    assert result.requires_grad is (not no_grad)
    assert key.requires_grad is key_grad and left.requires_grad is True
    resident(result,device)
    expected=left_values.astype(np.float64)@key_values.astype(np.float64).swapaxes(-1,-2)
    np.testing.assert_allclose(_array(result),expected,atol=1e-5,rtol=1e-5)
    if not no_grad:
        targets=(left,source,key) if key_grad else (left,)
        gradients=torch.autograd.grad(result.sum(),targets)
        ones=np.ones(expected.shape,dtype=np.float64)
        left_grad=ones@key_values.astype(np.float64)
        key_gradient=ones.swapaxes(-1,-2)@left_values.astype(np.float64)
        expected_gradients=(left_grad,key_gradient,key_gradient) if key_grad else (left_grad,)
        for actual,reference in zip(gradients,expected_gradients):
            resident(actual,device)
            np.testing.assert_allclose(_array(actual),reference,atol=1e-5,rtol=1e-5)
    np.testing.assert_array_equal(_array(key),key_values)
    np.testing.assert_array_equal(_array(left),left_values)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(3,4), (2,3,4), (1,2,1,3,4)])
def test_transpose_created_without_grad_does_not_reconnect_source(device,shape):
    require_rng_device(device)
    values=(np.arange(np.prod(shape),dtype=np.float32).reshape(shape)+1)/16
    key=torch.tensor(values,device=device,requires_grad=True)
    with torch.no_grad():
        view=key.transpose(-1,-2)
    assert view.requires_grad and view.is_leaf and view.grad_fn is None
    left_shape=shape[:-2]+(2,shape[-1])
    left=torch.ones(left_shape,device=device,requires_grad=True)
    result=left@view
    gradients=torch.autograd.grad(result.sum(),(key,view,left),allow_unused=True)
    assert gradients[0] is None
    # Native Torch's unrecorded no-grad view inherits requires_grad, but has
    # no recorded accumulator edge: neither it nor its source receives a VJP.
    assert gradients[1] is None and gradients[2] is not None
    expected=np.ones(left_shape,dtype=np.float64)@values.astype(np.float64).swapaxes(-1,-2)
    resident(result,device)
    np.testing.assert_allclose(_array(result),expected,atol=1e-5,rtol=1e-5)
    resident(gradients[2],device)
    np.testing.assert_allclose(_array(gradients[2]),
        np.ones(expected.shape,dtype=np.float64)@values.astype(np.float64),atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('api', ['operator', 'method', 'function'])
@pytest.mark.parametrize('no_grad', [False, True])
def test_cuda_owned_matmul_public_protocol_broadcast_full_vjp(api, no_grad):
    require_rng_device('cuda')

    def prove(value):
        assert type(value) is torch.Tensor
        resident(value, 'cuda')
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            assert jt.core._is_tensor_carrier(value)
            assert not isinstance(value, jt.core.Var)
            assert jt.core.dispatch_context([value]) == ('cuda', 0)
            assert (value.location(), value.device_id, value.placement_backend) == ('device', 0, 1)
            assert jt.core.backend_fallback_count() == 0
        else:
            assert value.is_cuda and value.device.index == 0

    def array(value):
        # Independent transport; never CPU-copy the original sharing group.
        before = (tuple(value.shape), tuple(value.stride()), value.dtype,
                  value.requires_grad, value.is_leaf, value.grad_fn is None, value.device)
        transport = value.detach().clone()
        assert transport is not value and transport.data_ptr() != value.data_ptr()
        prove(transport)
        result = transport.cpu().numpy().copy()
        assert (tuple(value.shape), tuple(value.stride()), value.dtype,
                value.requires_grad, value.is_leaf, value.grad_fn is None, value.device) == before
        return result

    def unbroadcast(value, shape):
        while value.ndim > len(shape):
            value = value.sum(axis=0)
        for axis, size in enumerate(shape):
            if size == 1 and value.shape[axis] != 1:
                value = value.sum(axis=axis, keepdims=True)
        return value.reshape(shape)

    # 2D, true broadcast-batch, and the HF rotary outer-product+transpose form.
    cases = [((2, 3), (3, 4), False),
             ((1, 2, 3, 4), (5, 1, 4, 2), False),
             ((3, 2, 4, 1), (3, 2, 1, 5), True)]
    for left_shape, right_shape, rotary_transpose in cases:
        a = (np.arange(np.prod(left_shape), dtype=np.float32).reshape(left_shape) % 13 - 6) / 19
        b = (np.arange(np.prod(right_shape), dtype=np.float32).reshape(right_shape) % 11 - 5) / 23
        left = torch.tensor(a, device='cuda', requires_grad=True)
        right = torch.tensor(b, device='cuda', requires_grad=True)
        with torch.no_grad() if no_grad else nullcontext():
            if api == 'operator':
                output = left.float() @ right.float()
            elif api == 'method':
                output = left.float().matmul(right.float())
            else:
                output = torch.matmul(left.float(), right.float())
            if rotary_transpose:
                output = output.transpose(2, 3)
        prove(left); prove(right); prove(output)
        assert left.requires_grad and right.requires_grad
        assert output.requires_grad is (not no_grad)
        expected_product = np.matmul(a.astype(np.float64), b.astype(np.float64))
        expected = expected_product.swapaxes(2, 3) if rotary_transpose else expected_product
        np.testing.assert_allclose(array(output), expected, atol=1e-5, rtol=1e-5)
        if no_grad:
            assert output.is_leaf and output.grad_fn is None
        else:
            dy = (np.arange(expected.size, dtype=np.float32).reshape(expected.shape) % 17 - 8) / 29
            cotangent = torch.tensor(dy, device='cuda')
            gradients = torch.autograd.grad(output, (left, right), grad_outputs=cotangent,
                                            create_graph=False, allow_unused=False)
            product_dy = dy.swapaxes(2, 3) if rotary_transpose else dy
            expected_a = unbroadcast(np.matmul(product_dy.astype(np.float64), b.astype(np.float64).swapaxes(-1, -2)), left_shape)
            expected_b = unbroadcast(np.matmul(a.astype(np.float64).swapaxes(-1, -2), product_dy.astype(np.float64)), right_shape)
            assert len(gradients) == 2
            for gradient, reference in zip(gradients, (expected_a, expected_b)):
                prove(gradient)
                assert not gradient.requires_grad and gradient.is_leaf and gradient.grad_fn is None
                np.testing.assert_allclose(array(gradient), reference, atol=1e-5, rtol=1e-5)
        assert left.grad is None and right.grad is None
        np.testing.assert_array_equal(array(left), a)
        np.testing.assert_array_equal(array(right), b)


def test_cuda_process_owned_cpu_and_cuda_rng_state_roundtrip():
    require_rng_device('cuda')

    def state_bytes(value):
        assert type(value) is torch.Tensor and value.dtype == torch.uint8
        assert value.ndim == 1 and value.device.type == 'cpu'
        resident(value, 'cpu')
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            assert jt.core._is_tensor_carrier(value)
            assert not isinstance(value, jt.core.Var)
            assert jt.core.dispatch_context([value]) == ('cpu', -1)
            assert (value.location(), value.device_id, value.placement_backend) == ('cpu', -1, 0)
        before = (tuple(value.shape), value.dtype, value.device,
                  value.requires_grad, value.is_leaf, value.grad_fn is None)
        raw = value.detach().clone().numpy().tobytes(order='C')
        assert len(raw) == value.numel()
        assert (tuple(value.shape), value.dtype, value.device,
                value.requires_grad, value.is_leaf, value.grad_fn is None) == before
        return raw

    def cuda_bytes(value):
        resident(value, 'cuda')
        assert type(value) is torch.Tensor and value.device.index == 0
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            assert jt.core._is_tensor_carrier(value)
            assert jt.core.dispatch_context([value]) == ('cuda', 0)
            assert (value.location(), value.device_id, value.placement_backend) == ('device', 0, 1)
            assert jt.core.backend_fallback_count() == 0
        return value.detach().clone().cpu().numpy().tobytes(order='C')

    # Real CUDA work, unrelated to any CPU tensor computation or random draw.
    anchor = torch.tensor([1., -2., 3.], device='cuda')
    anchor_result = anchor.square()
    resident(anchor_result, 'cuda')

    # CPU generator state is legitimate CPU ByteTensor metadata. Seeding its
    # public default owner changes state, but draws nothing on the CPU.
    torch.default_generator.manual_seed(73191)
    cpu_state = torch.get_rng_state().clone()
    cpu_saved = state_bytes(cpu_state)
    torch.cuda.manual_seed(73192)
    cuda_before_cpu_restore = state_bytes(torch.cuda.get_rng_state(0))
    torch.default_generator.manual_seed(73193)
    assert state_bytes(torch.get_rng_state()) != cpu_saved
    torch.set_rng_state(cpu_state)
    assert state_bytes(torch.get_rng_state()) == cpu_saved
    assert state_bytes(cpu_state) == cpu_saved
    assert state_bytes(torch.cuda.get_rng_state(0)) == cuda_before_cpu_restore

    # Only CUDA draws advance this stream; restore must replay the continuation
    # AND the exact saved/end states within each framework's own public format.
    warm = torch.rand(7, device='cuda')
    cuda_bytes(warm)
    cuda_state = torch.cuda.get_rng_state(0).clone()
    cuda_saved = state_bytes(cuda_state)
    expected = torch.rand(19, device='cuda')
    expected_bytes = cuda_bytes(expected)
    cuda_end = state_bytes(torch.cuda.get_rng_state(0))
    assert cuda_end != cuda_saved
    torch.cuda.set_rng_state(cuda_state, 0)
    assert state_bytes(torch.cuda.get_rng_state(0)) == cuda_saved
    actual = torch.rand(19, device='cuda')
    assert cuda_bytes(actual) == expected_bytes
    assert state_bytes(torch.cuda.get_rng_state(0)) == cuda_end
    assert state_bytes(cuda_state) == cuda_saved
    assert state_bytes(torch.get_rng_state()) == cpu_saved
    resident(anchor, 'cuda'); resident(anchor_result, 'cuda')
