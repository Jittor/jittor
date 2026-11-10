"""A library backward must reject changes through aliases of saved inputs."""
import numpy as np
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

def test_pending_setitem_attention_keeps_mutation_guard_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused float32 attention")

    def make_loss():
        arrays = [np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4) / scale
                  for scale in (100, 110, 120, 130)]
        parts = [torch.tensor(array, device="cuda:0", requires_grad=True)
                 for array in arrays]
        query = torch.cat(parts[:2], dim=-1)
        key = torch.cat(parts[2:], dim=-1)
        value = torch.tensor(
            np.arange(128, dtype=np.float32).reshape(1, 4, 4, 8) / 140,
            device="cuda:0", requires_grad=True)
        loss = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True).sum()
        float(loss)  # Materialize the pending setitem before backward.
        return query, loss, parts[0]

    query, loss, first = make_loss()
    loss.backward()
    assert float(first.grad.sum()) == pytest.approx(0.1866113, abs=1e-5)

    query, loss, _ = make_loss()
    with torch.no_grad():
        query.detach().fill_(7.0)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        loss.backward()
