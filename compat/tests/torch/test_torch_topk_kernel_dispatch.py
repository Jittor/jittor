"""Torch Topk uses the canonical selection seam and preserves result fields."""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability


_DEVICES = ["cpu"] + (["npu" if jt.compiler.has_acl else "cuda"]
    if capability.any_accelerator_enabled(backend=jt) else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_torch_topk_public_values_int64_indices_and_result_fields(device, dtype):
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=int(device != "cpu")), torch.no_grad():
        source = np.array([[1., 4., -2., 3., 0.], [7., -1., 5., 9., 6.]], dtype=np.float32)
        for count in (1, 3, 5):
            value = torch.tensor(source, dtype=dtype, device=device)
            result = torch.topk(value, count, dim=-1, largest=True, sorted=True)
            assert result.values.dtype == dtype and result.indices.dtype == torch.int64
            assert result.values.device.type == value.device.type == result.indices.device.type
            ids = np.argsort(-source, axis=-1)[:, :count]
            np.testing.assert_array_equal(result.indices.cpu().numpy(), ids)
            np.testing.assert_array_equal(result.values.float().cpu().numpy(), np.take_along_axis(source, ids, -1))
    assert jt.core.backend_fallback_count() == before


def test_torch_topk_consults_the_canonical_selector_without_new_sort(monkeypatch):
    from jittor.ops import sorting
    calls = []
    with jt.runtime.scope(use_cuda=0):
        x = torch.ones((2, 17))
        value, index = torch.ones((2, 3)), torch.zeros((2, 3), dtype=torch.int64)
        def kernel(*args):
            calls.append(args)
            return value, index
        monkeypatch.setattr(sorting, "select_kernel", lambda name, *args: kernel if name == "tensor.topk" else None)
        result = torch.topk(x, 3, dim=-1)
        assert result.values is value and result.indices is index
    assert calls == [(x, 3, -1, True, True)]
