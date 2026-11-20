"""QR keeps the mode, named factors and input device."""

import numpy as np
import pytest
import torch

from _helpers.capability import require_accelerator


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("shape", ((6, 3), (3, 6), (2, 6, 3)))
def test_linalg_qr_modes(device, shape):
    if device == "cuda":
        require_accelerator("cuda")
    values = np.random.default_rng(938).standard_normal(shape).astype("float32")
    source = torch.tensor(values, device=device)
    m, n = shape[-2:]
    batch = shape[:-2]
    k = min(m, n)
    for mode in ("reduced", "complete", "r"):
        factors = torch.linalg.qr(source, mode=mode)
        assert factors.Q is factors[0] and factors.R is factors[1]
        assert factors.Q.device == factors.R.device == source.device
        count = m if mode == "complete" else k
        assert tuple(factors.Q.shape) == ((0,) if mode == "r" else batch + (m, count))
        assert tuple(factors.R.shape) == batch + (count, n)
        r = factors.R.cpu().numpy()
        np.testing.assert_allclose(r, np.triu(r), atol=1e-7)
        if mode == "r":
            np.testing.assert_allclose(r.swapaxes(-1, -2) @ r,
                                       values.swapaxes(-1, -2) @ values, atol=1e-5, rtol=1e-5)
        else:
            q = factors.Q.cpu().numpy()
            np.testing.assert_allclose(q @ r, values, atol=1e-5, rtol=1e-5)
            identities = np.broadcast_to(np.eye(count), batch + (count, count))
            np.testing.assert_allclose(q.swapaxes(-1, -2) @ q, identities, atol=1e-5, rtol=1e-5)


def test_linalg_qr_rejects_derivatives_for_modes_without_defined_gradients():
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], requires_grad=True)
    with pytest.raises(RuntimeError, match="Complete QR is not differentiable"):
        torch.linalg.qr(source, mode="complete").R.sum().backward()
    with pytest.raises(RuntimeError, match="QR gradients require Q"):
        torch.linalg.qr(source, mode="r").R.sum().backward()
    torch.linalg.qr(source, mode="reduced").R.sum().backward()
    assert torch.isfinite(source.grad).all().item()
