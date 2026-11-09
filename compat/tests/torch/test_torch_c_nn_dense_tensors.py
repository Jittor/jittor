"""Dense flatten helpers exposed through the Torch C namespace."""
import numpy as np
import torch


def test_flatten_and_unflatten_dense_tensors_preserve_values_shapes_and_views():
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second = torch.tensor([5.0, 6.0, 7.0])

    flat = torch._C._nn.flatten_dense_tensors([first, second])
    np.testing.assert_array_equal(flat.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    restored = torch._C._nn.unflatten_dense_tensors(flat, [first, second])
    assert [tuple(value.shape) for value in restored] == [(2, 2), (3,)]
    np.testing.assert_array_equal(restored[0].numpy(), first.numpy())
    np.testing.assert_array_equal(restored[1].numpy(), second.numpy())

    with torch.no_grad():
        restored[1][0] = 9.0
    assert float(flat[4].item()) == 9.0
