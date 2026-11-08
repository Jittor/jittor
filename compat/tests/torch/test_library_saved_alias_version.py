"""A library backward must reject changes through aliases of saved inputs."""
import pytest
import torch


def test_library_saved_input_mutated_through_detached_alias_rejects_backward():
    library = torch.library.Library("saved_alias_version_regression", "DEF")
    library.define("square(Tensor x) -> Tensor")
    library.impl("square", lambda x: (x * x).detach(), "CPU")

    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 2 * x

    torch.library.register_autograd(
        "saved_alias_version_regression::square", backward, setup_context=setup
    )
    x = torch.tensor([2.0, 3.0], device="cpu", requires_grad=True)
    result = torch.ops.saved_alias_version_regression.square(x)
    # Evaluate the forward before the in-place write so the test exercises a
    # saved value rather than letting lazy execution hide the mutation.
    result.detach().cpu().numpy().copy()
    alias = x.detach()
    with torch.no_grad():
        alias.fill_(7.0)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(result.sum(), x)
