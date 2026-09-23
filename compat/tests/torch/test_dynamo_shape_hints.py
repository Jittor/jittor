"""Eager shape hints preserve real tensor values and import ownership."""

import importlib

import numpy as np
import pytest


def test_dynamo_decorators_module_identity():
    import torch

    module = importlib.import_module("torch._dynamo.decorators")
    assert torch._dynamo.decorators is module
    assert callable(module.mark_unbacked)
    with pytest.raises(AttributeError):
        getattr(module, "nonexistent_shape_hint")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [0, 1, 3])
def test_mark_unbacked_keeps_eager_values_and_accumulates_dimensions(device, rows):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("real CUDA unavailable")
    expected = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    tensor = torch.tensor(expected, device=device)
    mark = torch._dynamo.decorators.mark_unbacked
    assert mark(tensor, 0) is None
    assert mark(tensor, [1, 0]) is None
    assert mark(tensor, (0,)) is None
    assert tensor._dynamo_unbacked_indices == {0, 1}
    assert tensor.device.type == device
    assert tuple(tensor.shape) == (rows, 4)
    np.testing.assert_array_equal(tensor.cpu().numpy(), expected)
    if rows:
        # Eager rank calculation, matching the consumer's use of the hint.
        rank = (tensor > tensor[:, :1]).sum(dim=-1)
        np.testing.assert_array_equal(rank.cpu().numpy(), np.full(rows, 3))
        assert rank.device.type == device


def test_shim_reports_eager_only_shape_hint_fidelity():
    import torch

    if not hasattr(torch, "_torch_compat_install_context"):
        pytest.skip("shim-specific fidelity boundary")
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of

    record = fidelity_of("torch._dynamo.decorators.mark_unbacked")
    assert record.implementation is torch._dynamo.decorators.mark_unbacked
    assert record.level is Fidelity.APPROXIMATE
    tensor = torch.zeros((1, 4), device="cpu")
    for options in ({"strict": True}, {"hint_override": 8},
                    {"specialize_on": []}, {"shape_id": "shared_batch"}):
        with pytest.raises(NotImplementedError, match="symbolic"):
            torch._dynamo.decorators.mark_unbacked(tensor, 0, **options)
