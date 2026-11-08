"""Public storage alias writes preserve eager values and saved-tensor safety."""
import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def values(tensor):
    return tensor.detach().double().cpu().numpy()


@pytest.mark.parametrize("kind", ["detach", "parameter"])
@pytest.mark.parametrize("materialized", [False, True])
def test_bidirectional_alias_write_before_and_after_materialization(device, kind, materialized):
    source = torch.tensor([4., 5.], dtype=torch.float64, device=device)
    target = (source.detach() if kind == "detach" else
              torch.nn.Parameter(source, requires_grad=True))
    if materialized:
        np.testing.assert_array_equal(values(target), [4., 5.])
        np.testing.assert_array_equal(values(source), [4., 5.])
    with torch.no_grad():
        source.add_(2.)
    np.testing.assert_array_equal(values(target), [6., 7.])
    with torch.no_grad():
        target.add_(3.)
    np.testing.assert_array_equal(values(source), [9., 10.])
    assert not source.requires_grad
    assert target.requires_grad == (kind == "parameter")


@pytest.mark.parametrize("kind", ["self", "detach", "parameter"])
@pytest.mark.parametrize("materialized", [False, True])
def test_native_saved_input_alias_write_rejects_backward(device, kind, materialized):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    loss = (source * source).sum()
    if materialized:
        assert loss.item() == 13.
    target = (source if kind == "self" else source.detach() if kind == "detach" else
              torch.nn.Parameter(source, requires_grad=True))
    with torch.no_grad():
        target.add_(1.)
    np.testing.assert_array_equal(values(source), [3., 4.])
    with pytest.raises(RuntimeError, match="inplace operation"):
        torch.autograd.grad(loss, source)


@pytest.mark.parametrize("materialized", [False, True])
def test_pending_forward_survives_write_and_constant_derivative_still_works(device, materialized):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    old_forward = source + 1.
    if materialized:
        np.testing.assert_array_equal(values(old_forward), [3., 4.])
    alias = source.detach()
    with torch.no_grad():
        alias.add_(10.)
    np.testing.assert_array_equal(values(old_forward), [3., 4.])
    grad, = torch.autograd.grad(old_forward.sum(), source)
    np.testing.assert_array_equal(values(grad), [1., 1.])


def test_custom_function_saved_caller_alias_is_version_checked(device):
    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, source):
            ctx.save_for_backward(source)
            return source * source

        @staticmethod
        def backward(ctx, grad):
            source, = ctx.saved_tensors
            return 2. * source * grad

    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    loss = Square.apply(source).sum()
    with torch.no_grad():
        torch.nn.Parameter(source).add_(1.)
    with pytest.raises(RuntimeError, match="inplace operation"):
        torch.autograd.grad(loss, source)


def test_mutating_exp_input_does_not_reject_saved_output_backward(device):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    output = source.exp()
    with torch.no_grad():
        source.detach().add_(1.)
    grad, = torch.autograd.grad(output.sum(), source)
    np.testing.assert_allclose(values(output), np.exp([2., 3.]), rtol=1e-12)
    np.testing.assert_allclose(values(grad), np.exp([2., 3.]), rtol=1e-12)


@pytest.mark.parametrize("keep_exp_holder", [False, True])
def test_materialized_loss_preserves_saved_exp_before_input_write(device, keep_exp_holder):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    if keep_exp_holder:
        output = source.exp()
        loss = output.sum()
    else:
        loss = source.exp().sum()
    old_loss = loss.item()
    with torch.no_grad():
        source.add_(1.)
    assert loss.item() == old_loss
    grad, = torch.autograd.grad(loss, source)
    # Read the gradient first: source/exp readback must not materialize an
    # otherwise unheld saved intermediate before backward execution.
    np.testing.assert_allclose(values(grad), np.exp([2., 3.]), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(values(source), [3., 4.])
    np.testing.assert_allclose(old_loss, np.exp([2., 3.]).sum(), rtol=1e-12, atol=1e-12)


def test_mutating_sum_input_leaves_shape_only_backward_valid(device):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    loss = source.sum()
    with torch.no_grad():
        source.detach().add_(1.)
    assert loss.item() == 5.
    grad, = torch.autograd.grad(loss, source)
    np.testing.assert_array_equal(values(grad), [1., 1.])


def test_mutating_exp_output_rejects_saved_output_backward(device):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    output = source.exp()
    with torch.no_grad():
        output.detach().add_(1.)
    with pytest.raises(RuntimeError, match="inplace operation"):
        torch.autograd.grad(output.sum(), source)


def test_grad_enabled_nonleaf_inplace_retains_original_gradient(device):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    output = source * 2.
    output.add_(1.)
    grad, = torch.autograd.grad(output.sum(), source)
    np.testing.assert_array_equal(values(output), [5., 7.])
    np.testing.assert_array_equal(values(grad), [2., 2.])


@pytest.mark.parametrize("materialized", [False, True])
def test_contiguous_view_alias_preserves_holder_and_old_forward(device, materialized):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    view = source.view(1, 2)
    old_forward = view + 1.
    if materialized:
        np.testing.assert_array_equal(values(old_forward), [[3., 4.]])
    with torch.no_grad():
        view.add_(10.)
    np.testing.assert_array_equal(values(source), [12., 13.])
    np.testing.assert_array_equal(values(old_forward), [[3., 4.]])
    grad, = torch.autograd.grad(old_forward.sum(), source)
    np.testing.assert_array_equal(values(grad), [1., 1.])


def test_independent_clone_write_does_not_invalidate_saved_source(device):
    source = torch.tensor([2., 3.], dtype=torch.float64, device=device, requires_grad=True)
    loss = (source * source).sum()
    independent = source.detach().clone()
    with torch.no_grad():
        independent.add_(10.)
    np.testing.assert_array_equal(values(source), [2., 3.])
    grad, = torch.autograd.grad(loss, source)
    np.testing.assert_array_equal(values(grad), [4., 6.])


@pytest.mark.parametrize("materialized", [False, True])
def test_data_alias_keeps_an_independent_version_counter(device, materialized):
    source = torch.tensor([2., 3.], device=device, requires_grad=True)
    loss = (source * source).sum()
    if materialized:
        assert loss.item() == 13.
    source.data.add_(1.)
    gradient, = torch.autograd.grad(loss, source)
    np.testing.assert_array_equal(values(source), [3., 4.])
    np.testing.assert_array_equal(values(gradient), [6., 8.])


def test_data_replacement_preserves_saved_version(device):
    source = torch.tensor([2., 3.], device=device, requires_grad=True)
    loss = (source * source).sum()
    source.data = torch.tensor([4., 5.], device=device)
    gradient, = torch.autograd.grad(loss, source)
    np.testing.assert_array_equal(values(gradient), [8., 10.])
