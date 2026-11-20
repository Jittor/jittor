"""The Torch Embedding constructor invokes its initialization hook."""
import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


class CustomEmbedding(torch.nn.Embedding):
    def reset_parameters(self):
        self.reset_calls = getattr(self, "reset_calls", 0) + 1
        self.reset_device = self.weight.device.type
        self.reset_dtype = self.weight.dtype
        with torch.no_grad():
            self.weight.fill_(3.0)


def test_embedding_calls_subclass_hook_after_placement(device):
    embedding = CustomEmbedding(4, 3, dtype=torch.float64, device=device)
    assert embedding.reset_calls == 1
    assert embedding.reset_device == embedding.weight.device.type == device
    assert embedding.reset_dtype == embedding.weight.dtype == torch.float64
    assert embedding.weight.requires_grad
    np.testing.assert_array_equal(
        embedding.weight.detach().cpu().numpy(), np.full((4, 3), 3.0),
    )
    indices = torch.tensor([1, 1, 3], device=device)
    embedding(indices).sum().backward()
    expected = np.repeat([[0.0], [2.0], [0.0], [1.0]], 3, axis=1)
    np.testing.assert_array_equal(embedding.weight.grad.cpu().numpy(), expected)


def test_embedding_skips_reset_when_given_weights(device):
    weight = torch.full((4, 3), 2.0, dtype=torch.float64, device=device)
    embedding = CustomEmbedding(4, 3, _weight=weight, dtype=torch.float64, device=device)
    assert not hasattr(embedding, "reset_calls")
    np.testing.assert_array_equal(
        embedding.weight.detach().cpu().numpy(), np.full((4, 3), 2.0),
    )


def test_embedding_default_reset_zeroes_padding_and_retains_freeze(device):
    embedding = torch.nn.Embedding(
        4, 3, padding_idx=-1, _freeze=True, dtype=torch.float64, device=device,
    )
    assert embedding.weight.device.type == device
    assert embedding.weight.dtype == torch.float64
    assert not embedding.weight.requires_grad
    data = embedding.weight.detach().cpu().numpy()
    np.testing.assert_array_equal(data[-1], np.zeros(3))
    assert np.isfinite(data).all() and np.count_nonzero(data[:-1])
