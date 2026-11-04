"""Public norm spellings share one optional native vector-norm query boundary."""
import numpy as np
import pytest
import torch
import jittor as jt


def test_norm_entry_points_share_canonical_query(monkeypatch):
    calls = []
    def record(x, order, dim, keepdim):
        calls.append((order, dim, keepdim))
        return None
    monkeypatch.setattr(jt.linalg, "_try_vector_norm", record)
    source = np.asarray([[3., 4.], [5., 12.]], dtype="float32")
    with jt.runtime.scope(use_cuda=0), torch.no_grad():
        value = torch.tensor(source, device="cpu")
        norm = torch.norm(value, p=2, dim=-1, keepdim=True)
        normalized = torch.nn.functional.normalize(value, p=2, dim=-1, eps=1e-4)
        np.testing.assert_allclose(norm.numpy(), [[5.], [13.]])
        np.testing.assert_allclose(normalized.numpy(), source / [[5.], [13.]], rtol=1e-6)
    assert calls == [(2., -1, True), (2, -1, True)]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_cpu_norm_and_normalize_keep_existing_dtype_shape_and_gradients(dtype):
    # Ordinary finite values verify unchanged fallback; the new ACL SDK route
    # does not claim to repair legacy CPU half overflow / underflow semantics.
    source = np.asarray([[3., 4.], [5., 12.]], dtype=dtype)
    with jt.runtime.scope(use_cuda=0), torch.enable_grad():
        x = torch.tensor(source, device="cpu", requires_grad=True)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        assert norm.dtype == x.dtype and tuple(norm.shape) == (2, 1)
        np.testing.assert_allclose(norm.detach().numpy(), [[5.], [13.]], rtol=1e-3)
        norm.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), source / [[5.], [13.]], rtol=2e-3, atol=1e-5)
