"""Attention must retain explicit AD targets and read strided values correctly."""
import os

import numpy as np
import pytest
import torch


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    return request.param


def _reference(q, k, v, cotangent, mask):
    q, k, v, cotangent = (x.astype("float64") for x in (q, k, v, cotangent))
    scores = (q @ k.swapaxes(-1, -2)) * q.shape[-1] ** -.5
    if mask is not None:
        scores = np.where(mask, scores, -np.inf) if mask.dtype == bool else scores + mask
    maximum = scores.max(-1, keepdims=True)
    probability = np.exp(scores - np.where(np.isinf(maximum), 0, maximum))
    total = probability.sum(-1, keepdims=True)
    probability = np.divide(probability, total, out=np.zeros_like(probability), where=total > 0)
    dp = cotangent @ v.swapaxes(-1, -2)
    ds = probability * (dp - (dp * probability).sum(-1, keepdims=True))
    return (probability @ v, (ds @ k) * q.shape[-1] ** -.5,
            (ds.swapaxes(-1, -2) @ q) * q.shape[-1] ** -.5,
            probability.swapaxes(-1, -2) @ cotangent)


@pytest.mark.parametrize("layout", ["transposed", "strided", "mixed"])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_explicit_input_gradients_keep_the_view_edge_and_values(device, layout, mask_kind):
    rng = np.random.RandomState(393)
    arrays, tensors = [], []
    for index, length in enumerate((5, 7, 7)):
        if layout == "strided":
            storage = rng.randn(2, 3, length, 16).astype("float32")
            value = torch.tensor(storage, device=device, requires_grad=True)[..., ::2]
            array = storage[..., ::2].copy()
        else:
            array = rng.randn(2, 3, length, 8).astype("float32")
            if layout == "transposed" or (layout == "mixed" and index != 1):
                value = torch.tensor(array.transpose(0, 2, 1, 3).copy(), device=device,
                                     requires_grad=True).transpose(1, 2)
            else:
                value = torch.tensor(array, device=device, requires_grad=True)
        arrays.append(array)
        tensors.append(value)
    mask = None
    if mask_kind == "bool":
        mask = np.ones((5, 7), dtype=bool)
        mask[1, :] = False
        mask[2, 3:] = False
    elif mask_kind == "additive":
        mask = rng.uniform(-2, .1, (5, 7)).astype("float32")
    mask_tensor = None if mask is None else torch.tensor(mask, device=device)
    output = torch.nn.functional.scaled_dot_product_attention(*tensors, attn_mask=mask_tensor)
    assert isinstance(output, torch.Tensor) and output.requires_grad and output.grad_fn is not None
    for _ in range(2):
        cotangent = rng.randn(*arrays[0].shape).astype("float32")
        expected = _reference(*arrays, cotangent, mask)
        loss = (output * torch.tensor(cotangent, device=device)).sum()
        gradients = torch.autograd.grad(loss, tensors, retain_graph=True)
        for actual, reference in zip([output, *gradients], expected):
            assert isinstance(actual, torch.Tensor)
            if device == "cuda":
                actual.sync()
                assert actual.location() == "device" and actual.device_id >= 0
            np.testing.assert_allclose(actual.detach().cpu().numpy(), reference, rtol=1e-4,
                                       atol=1e-5 * np.abs(reference).max())
