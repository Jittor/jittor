"""Native dense FlashAttention must preserve logical noncontiguous views."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('JITTOR_FLASH_ATTN_JITTOR_SRC'),
    reason='requires official FlashAttention CUDA backend',
)


@pytest.mark.parametrize('layout', ['contiguous', 'transposed'])
def test_native_dense_gqa_inference_respects_view_layout(layout):
    import jittor as jt
    import torch
    import flash_attn
    from jittor.compat.shim.backends import flash_attention

    rng = np.random.default_rng(112)
    raw = [rng.normal(size=shape).astype(np.float32) for shape in
           ((1, 32, 4, 32), (1, 32, 2, 32), (1, 32, 2, 32))]
    with torch.no_grad(), jt.flag_scope(use_cuda=1):
        q, k, v = [torch.tensor(a, device='cuda').to(torch.bfloat16) for a in raw]
        if layout == 'transposed':
            q = q.transpose(1, 2).contiguous().transpose(1, 2)
            k = k.transpose(1, 2).contiguous().transpose(1, 2)
            assert not q.is_contiguous() and not k.is_contiguous()
        source = [t.detach().clone().cpu().float().numpy() for t in (q, k, v)]
        output = flash_attn.flash_attn_func(q, k, v, causal=True)
        assert flash_attn.flashattn_jittor_backend().startswith('flashattn_jittor_official:')
        assert flash_attention.load_backend()._flashattn_jittor_packed_low_level is not None
        actual = output.detach().clone().cpu().float().numpy()

    queries, keys, values = source
    expected = np.empty_like(queries)
    causal_mask = np.triu(np.ones((32, 32), dtype=bool), 1)
    for h in range(4):
        score = queries[0, :, h] @ keys[0, :, h // 2].T / np.sqrt(32)
        score[causal_mask] = -np.inf
        score -= score.max(axis=-1, keepdims=True)
        probability = np.exp(score); probability /= probability.sum(axis=-1, keepdims=True)
        expected[0, :, h] = probability @ values[0, :, h // 2]
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize('causal', [False, True])
def test_dense_graph_replays_changing_inputs(causal):
    import jittor as jt
    import torch
    import flash_attn

    rng = np.random.default_rng(478)
    runner = jt.capture_step(lambda q, k, v: flash_attn.flash_attn_func(q, k, v, causal=causal))
    try:
        for _ in range(8):
            arrays = [rng.normal(0, .2, shape).astype('float32') for shape in
                      [(1, 37, 4, 64), (1, 53, 2, 64), (1, 53, 2, 64)]]
            inputs = [torch.tensor(value, dtype=torch.bfloat16, device='cuda') for value in arrays]
            values = [value.detach().float().cpu().numpy().astype('float64') for value in inputs]
            expected = np.empty_like(values[0])
            for head in range(4):
                score = values[0][0, :, head] @ values[1][0, :, head//2].T * .125
                if causal:
                    # FlashAttention aligns rectangular causal masks at bottom right.
                    score[np.triu(np.ones((37, 53), bool), 53-37+1)] = -np.inf
                score -= score.max(-1, keepdims=True)
                probability = np.exp(score)
                probability /= probability.sum(-1, keepdims=True)
                expected[0, :, head] = probability @ values[2][0, :, head//2]
            actual = runner(*inputs)
            actual.sync()
            assert actual.device.type == 'cuda' and not actual.requires_grad
            np.testing.assert_allclose(actual.float().cpu().numpy(), expected, atol=3e-4, rtol=.02)
        assert runner.refused is None
        assert runner.stats['replayed'] >= 4
        assert runner.stats['graph'] >= 1
    finally:
        runner.invalidate()


@pytest.mark.parametrize('query_length,key_length', [(37, 53), (53, 37)])
def test_torch_rectangular_causal_attention_uses_top_left_mask(query_length, key_length):
    import jittor as jt
    import torch
    from jittor.compat.diagnostics import sdpa_flash_stats

    rng = np.random.default_rng(661)
    shapes = [(1, 4, query_length, 64), (1, 4, key_length, 64), (1, 4, key_length, 64)]
    values = [torch.tensor(rng.normal(0, .2, shape).astype('float32'), device='cuda',
                            dtype=torch.bfloat16, requires_grad=True) for shape in shapes]
    q, k, v = [value.detach().float().cpu().numpy().astype('float64') for value in values]
    raw_seed = rng.normal(size=shapes[0]).astype('float32')
    seed_tensor = torch.tensor(raw_seed, device='cuda').to(torch.bfloat16)
    seed = seed_tensor.float().cpu().numpy().astype('float64')
    scores = q @ k.swapaxes(-1, -2) * .125
    scores[..., np.triu(np.ones((query_length, key_length), bool), 1)] = -np.inf
    scores -= scores.max(-1, keepdims=True)
    probability = np.exp(scores)
    probability /= probability.sum(-1, keepdims=True)
    dp = seed @ v.swapaxes(-1, -2)
    ds = probability*(dp-(probability*dp).sum(-1, keepdims=True))
    expected = (probability @ v, ds @ k*.125,
                ds.swapaxes(-1, -2) @ q*.125, probability.swapaxes(-1, -2) @ seed)
    before = sdpa_flash_stats(jt)['hits']
    output = torch.nn.functional.scaled_dot_product_attention(*values, is_causal=True)
    (output.float()*seed_tensor.float()).sum().backward()
    assert sdpa_flash_stats(jt)['hits'] > before
    for actual, wanted in zip([output]+[value.grad for value in values], expected):
        np.testing.assert_allclose(actual.detach().float().cpu().numpy(), wanted,
                                   rtol=.04, atol=max(1e-7, np.abs(wanted).max()*.004))


@pytest.mark.parametrize('causal', [False, True])
def test_packed_varlen_training_normalizes_saved_views(causal):
    import torch
    import flash_attn
    rng = np.random.default_rng(993)
    raw = rng.normal(0, .2, (5, 3, 4, 64)).astype('float32')
    qkv = torch.tensor(raw, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    value = qkv.detach().float().cpu().numpy().astype('float64')
    lengths = torch.tensor([0, 3, 5], dtype=torch.int32, device='cuda')
    seed_tensor = torch.tensor(rng.normal(size=(5, 4, 64)).astype('float32'),
                               dtype=torch.bfloat16, device='cuda')
    seed = seed_tensor.float().cpu().numpy().astype('float64')
    outputs, gradients = [], []
    for begin, end in ((0, 3), (3, 5)):
        q, k, v = [value[begin:end, i].transpose(1, 0, 2) for i in range(3)]
        dy = seed[begin:end].transpose(1, 0, 2)
        score = q @ k.swapaxes(-1, -2)*.125
        if causal:
            score[..., np.triu(np.ones((end-begin, end-begin), bool), 1)] = -np.inf
        probability = np.exp(score-score.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        dp = dy @ v.swapaxes(-1, -2)
        ds = probability*(dp-(probability*dp).sum(-1, keepdims=True))
        outputs.append((probability @ v).transpose(1, 0, 2))
        gradients.append(np.stack([(ds @ k*.125).transpose(1, 0, 2),
                                   (ds.swapaxes(-1, -2) @ q*.125).transpose(1, 0, 2),
                                   (probability.swapaxes(-1, -2) @ dy).transpose(1, 0, 2)], axis=1))
    output = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, lengths, 3, causal=causal)
    (output.float()*seed_tensor.float()).sum().backward()
    for actual, wanted in ((output, np.concatenate(outputs)), (qkv.grad, np.concatenate(gradients))):
        np.testing.assert_allclose(actual.detach().float().cpu().numpy(), wanted,
                                   rtol=.04, atol=max(1e-7, np.abs(wanted).max()*.004))
