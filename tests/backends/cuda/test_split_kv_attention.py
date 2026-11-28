import numpy as np
import pytest

import jittor as jt
from _helpers.capability import require_accelerator
from jittor.nn.functional.attention import scaled_dot_product_attention


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype, grouped, mask_kind, layout",
    [
        ("float16", False, None, "dense"),
        ("float16", True, "bool", "pending"),
        ("bfloat16", False, "float32", "sequence"),
    ],
)
def test_split_kv_inference(dtype, grouped, mask_kind, layout, monkeypatch):
    require_accelerator("cuda")
    if min(jt.flags.cuda_archs or [0]) < 80:
        pytest.skip("Split-KV attention uses Ampere tensor cores")

    from jittor.backends.cuda.kernels.nn import split_kv_attention_cuda as provider

    batches, heads, queries, keys, width = 2, 4, 13, 537, 48
    kv_heads = heads // 2 if grouped else heads
    rng = np.random.default_rng(704)
    masks = None
    if mask_kind == "bool":
        masks = rng.random((1, 1, queries, keys)) > 0.3
        masks[..., 0, :] = False
    elif mask_kind == "float32":
        masks = rng.normal(0, 0.25, (batches, 1, queries, keys)).astype(np.float32)
        masks[..., 0, :] = -np.inf
        masks[..., 1, :173] = -np.inf

    def tensor(h, length):
        array = rng.normal(0, 0.75, (batches, h, length, width)).astype(np.float32)
        base = jt.array(array if layout == "dense" else
                        np.ascontiguousarray(array.swapaxes(1, 2))).cast(dtype)
        exact = base.float32().numpy().astype(np.float64)
        if layout == "dense":
            return base, exact
        view = base._storage_permute((0, 2, 1, 3))
        return (view.contiguous() if layout == "pending" else view), exact.swapaxes(1, 2)

    calls = []
    original = provider.split_kv_attention

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(provider, "split_kv_attention", observe)
    with jt.flag_scope(use_cuda=1, backend_fallback="error", keep_graph=2), jt.no_grad():
        q, q_ref = tensor(heads, queries)
        k, k_ref = tensor(kv_heads, keys)
        v, v_ref = tensor(kv_heads, keys)
        if layout == "pending":
            assert all(t._is_pending_contiguous() for t in (q, k, v))
        mask = jt.array(masks) if masks is not None else None
        scale = 0.17
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=grouped)
        out.sync()
        assert out.location() == "device"
        assert calls == [True]
        assert out._storage_is_contiguous() == (layout != "sequence")

        groups = heads // kv_heads
        scores = q_ref @ np.repeat(k_ref, groups, axis=1).swapaxes(-1, -2) * scale
        if mask_kind == "bool":
            scores = np.where(masks, scores, -np.inf)
        elif mask_kind == "float32":
            scores += masks
        top = scores.max(-1, keepdims=True)
        weights = np.exp(scores - np.where(np.isneginf(top), 0.0, top))
        totals = weights.sum(-1, keepdims=True)
        expected = (weights / np.where(totals == 0.0, 1.0, totals)) @ np.repeat(v_ref, groups, axis=1)
        actual = out.float32().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-4 if dtype == "float16" else 6e-4,
                                   rtol=4e-3 if dtype == "float16" else 1e-2)
        if mask is not None:
            np.testing.assert_array_equal(actual[..., 0, :], 0)
