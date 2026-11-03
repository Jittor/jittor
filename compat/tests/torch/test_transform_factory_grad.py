"""Tensor transforms keep input AD edges; sampling accepts Torch size keywords."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("name", ["tril", "triu"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_triangular_transform_preserves_input_gradient(device, name, requires_grad):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    value = torch.tensor([[1., 2.], [3., 4.]], device=device,
                         requires_grad=requires_grad)
    output = getattr(torch, name)(value, diagonal=1)
    mask = getattr(np, name)(np.ones((2, 2)), k=1)
    np.testing.assert_array_equal(output.detach().cpu().numpy(),
                                  np.array([[1., 2.], [3., 4.]]) * mask)
    assert output.requires_grad == requires_grad
    if requires_grad:
        output.sum().backward()
        np.testing.assert_array_equal(value.grad.cpu().numpy(), mask)
    else:
        assert output.is_leaf and output.grad_fn is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("explicit_generator", [False, True])
def test_normal_keyword_size_and_stopped_tensor_inputs(device, explicit_generator):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    kwargs = dict(device=device)
    if explicit_generator:
        kwargs["generator"] = torch.Generator(device=device).manual_seed(17)
    scalar = torch.normal(mean=0., std=.02, size=(2, 3, 1), **kwargs)
    assert tuple(scalar.shape) == (2, 3, 1)
    assert not scalar.requires_grad and scalar.is_leaf
    assert np.isfinite(scalar.cpu().numpy()).all()
    mean = torch.ones(7, device=device, dtype=torch.float64)
    options = {"generator": kwargs["generator"]} if explicit_generator else {}
    output = torch.normal(mean, .2, **options)
    assert tuple(output.shape) == (7,) and output.dtype == mean.dtype
    assert output.device == mean.device and not output.requires_grad
    assert output.is_leaf and np.isfinite(output.cpu().numpy()).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trainable_normal_rejects_without_sampling_in_the_frontend(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    mean = torch.ones(3, device=device, requires_grad=True)
    generator = torch.Generator(device=device).manual_seed(31)
    before = generator.get_state().cpu().numpy().copy()
    if hasattr(torch, "_torch_compat_install_context"):
        with pytest.raises(NotImplementedError, match="trainable mean/std"):
            torch.normal(mean, .2, generator=generator)
        np.testing.assert_array_equal(generator.get_state().cpu().numpy(), before)
    else:
        torch.normal(mean, .2, generator=generator).sum().backward()
        np.testing.assert_array_equal(mean.grad.cpu().numpy(), np.zeros(3))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("no_grad", [False, True])
def test_collate_retains_only_the_input_gradient_edges(device, requires_grad, no_grad):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    left = torch.tensor([1., 2.], device=device, requires_grad=requires_grad)
    right = torch.tensor([3., 4.], device=device, requires_grad=requires_grad)
    with torch.no_grad() if no_grad else nullcontext():
        output = torch.utils.data.default_collate([left, right])
    assert output.requires_grad == (requires_grad and not no_grad)
    np.testing.assert_array_equal(output.detach().cpu().numpy(), [[1., 2.], [3., 4.]])
    if output.requires_grad:
        output.sum().backward()
        np.testing.assert_array_equal(left.grad.cpu().numpy(), [1., 1.])
        np.testing.assert_array_equal(right.grad.cpu().numpy(), [1., 1.])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_plain_empty_defaults_to_a_stopped_leaf(device, shape):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with torch.device(device):
        output = torch.empty(*shape)
    assert output.is_leaf and not output.requires_grad
    assert output.device.type == device
    output.zero_()
    np.testing.assert_array_equal(output.cpu().numpy(), np.zeros(shape))
