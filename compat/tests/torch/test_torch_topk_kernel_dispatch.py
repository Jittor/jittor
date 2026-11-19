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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [1, 5, 32])
def test_cuda_wide_topk_preserves_dtype_values_and_indices(dtype, k):
    if not jt.has_cuda or jt.compiler.has_acl:
        pytest.skip("requires native CUDA")
    from jittor.backends.cuda.kernels.misc.topk_cuda import wide_topk_cuda
    from jittor.ops.sorting import _select_topk_kernel

    # Force winners across tiles, including the non-aligned final tile.
    source = np.full((2, 24515), -123, dtype=np.float32)
    columns = (3 + np.arange(32) * 793) % source.shape[1]
    source[0, columns] = np.arange(-16, 16, dtype=np.float32)
    source[1, columns] = np.arange(16, 48, dtype=np.float32)[::-1]
    source[1, columns[6]] = np.inf
    source[1, columns[30]] = np.nan
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1), torch.no_grad():
        value = torch.tensor(source, dtype=dtype, device="cuda")
        assert _select_topk_kernel(value, k, -1, True, True) is wide_topk_cuda
        result = torch.topk(value, k, dim=-1)
        assert result.values.dtype == dtype and result.indices.dtype == torch.int64
        assert result.values.device.type == result.indices.device.type == "cuda"
        order = np.where(np.isnan(source), np.inf, source)
        # NaN sorts ahead of positive infinity for the wide selection.
        expected = np.argsort(-order, axis=-1, kind="stable")[:, :k]
        expected[1, 0] = columns[30]
        if k > 1:
            expected[1, 1] = columns[6]
        np.testing.assert_array_equal(result.indices.cpu().numpy(), expected)
        np.testing.assert_array_equal(result.values.float().cpu().numpy(),
                                      np.take_along_axis(source, expected, -1))
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_wide_low_precision_topk_keeps_input_order_for_ties(dtype):
    if not jt.has_cuda or jt.compiler.has_acl:
        pytest.skip("requires native CUDA")
    source = np.zeros((1, 24515), dtype=np.float32)
    peak = np.arange(15, 24000, 1337)
    source[:, peak] = 1
    with jt.runtime.scope(use_cuda=1), torch.no_grad():
        value = torch.tensor(source, dtype=dtype, device="cuda")
        result = torch.topk(value, 5, dim=-1)
        np.testing.assert_array_equal(result.indices.cpu().numpy(), peak[None, :5])
        np.testing.assert_array_equal(result.values.float().cpu().numpy(), np.ones((1, 5)))


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
