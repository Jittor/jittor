"""In-place tensor arithmetic used by third-party optimizer updates."""
import numpy as np
import torch


def test_lerp_and_lerp_inplace_match_optimizer_formula():
    source = torch.tensor([2.0, 6.0])
    end = torch.tensor([10.0, 2.0])
    np.testing.assert_allclose(source.lerp(end, 0.25).numpy(), [4.0, 5.0])

    result = source.lerp_(end, 0.25)

    assert result is source
    np.testing.assert_allclose(source.numpy(), [4.0, 5.0])


def test_addcmul_and_addcdiv_inplace_preserve_identity_and_values():
    destination = torch.tensor([1.0, 2.0])
    first = torch.tensor([2.0, 6.0])
    second = torch.tensor([4.0, 3.0])

    result = destination.addcmul_(first, second, value=0.5)
    assert result is destination
    np.testing.assert_allclose(destination.numpy(), [5.0, 11.0])

    result = destination.addcdiv_(first, second, value=2.0)
    assert result is destination
    np.testing.assert_allclose(destination.numpy(), [6.0, 15.0])
