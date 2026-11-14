"""The upstream CUTLASS attention kernels execute on CUDA, against FP64 math."""
import os

import numpy as np
import pytest

import jittor as jt
from _helpers import capability


@pytest.fixture
def kernel():
    if not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('CUDA is unavailable')
    if not os.environ.get('JITTOR_EFFICIENT_ATTN_SRC'):
        pytest.skip('upstream CUTLASS attention source is not configured')
    from jittor.backends.cuda.kernels.nn import efficient_attention_cuda
    with jt.flag_scope(use_cuda=1, backend_fallback='error'):
        yield efficient_attention_cuda


def reference(q, k, v, cotangent, causal, mask):
    q, k, v, cotangent = [value.astype('float64') for value in (q, k, v, cotangent)]
    scores = q @ k.swapaxes(-1, -2) * q.shape[-1]**-.5
    if causal:
        scores = np.where(np.triu(np.ones(scores.shape[-2:], bool), 1), -np.inf, scores)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf) if mask.dtype == bool else scores+mask
    top = scores.max(-1, keepdims=True)
    weights = np.exp(scores-np.where(np.isinf(top), 0., top))
    total = weights.sum(-1, keepdims=True)
    probability = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    dp = cotangent @ v.swapaxes(-1, -2)
    ds = probability*(dp-(probability*dp).sum(-1, keepdims=True))
    return (probability @ v, ds @ k * q.shape[-1]**-.5,
            ds.swapaxes(-1, -2) @ q * q.shape[-1]**-.5,
            probability.swapaxes(-1, -2) @ cotangent)


@pytest.mark.parametrize('head_dim', [64, 80, 128])
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('mask_kind', ['none', 'boolean', 'additive', 'fully-masked-row'])
def test_forward_backward_partial_tiles_and_broadcast_mask(kernel, head_dim, dtype, causal, mask_kind):
    rng = np.random.default_rng(417+head_dim)
    q, k, v = [rng.normal(0, .3, shape).astype('float32') for shape in
               [(2, 4, 65, head_dim), (2, 4, 97, head_dim), (2, 4, 97, 32)]]
    cotangent = rng.normal(size=(2, 4, 65, 32)).astype('float32')
    mask = None
    if mask_kind == 'boolean':
        mask = rng.random((2, 1, 1, 97)) > .2
        mask[..., 0] = True
    elif mask_kind in ('additive', 'fully-masked-row'):
        mask = rng.normal(0, .1, (2, 1, 65, 97)).astype('float32')
        if mask_kind == 'fully-masked-row':
            mask[0, :, 0, :] = -np.inf
    inputs = [jt.array(value).cast(dtype).start_grad() for value in (q, k, v)]
    values = [value.float32().numpy() for value in inputs]
    seed = jt.array(cotangent).cast(dtype).float32().numpy()
    typed_mask = None if mask is None else jt.array(mask).stop_grad()
    reference_mask = mask
    if mask is not None and mask.dtype != bool:
        typed_mask = typed_mask.cast(dtype).stop_grad()
        reference_mask = typed_mask.float32().numpy()
    expected = reference(*values, seed, causal, reference_mask)
    before = jt.core.backend_fallback_count()
    output = kernel._efficient_attention(*inputs, attn_mask=typed_mask,
                                         is_causal=causal, allow_reduced_precision=True)
    assert output is not None
    gradients = jt.grad((output*jt.array(cotangent)).sum(), inputs)
    for actual, wanted in zip([output]+list(gradients), expected):
        actual.sync()
        assert actual.location() == 'device' and actual.device_id >= 0
        tolerance = {'float32': 1e-4, 'float16': .004, 'bfloat16': .04}[dtype]
        np.testing.assert_allclose(actual.float32().numpy(), wanted, rtol=tolerance,
                                   atol=max(1e-9, np.abs(wanted).max()*tolerance*.1))
    assert jt.core.backend_fallback_count() == before


def test_registered_sdpa_uses_the_upstream_provider(kernel):
    from jittor._runtime.dispatch import select_kernel
    from jittor.nn.functional.attention import scaled_dot_product_attention
    values = [jt.randn((1, 2, 33, 64)).stop_grad() for _ in range(3)]
    assert select_kernel('nn.fused_attention', *values) is kernel._efficient_attention
    actual = scaled_dot_product_attention(*values)
    direct = kernel._efficient_attention(*values)
    np.testing.assert_allclose(actual.numpy(), direct.numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('head_dim,value_dim', [(8, 8), (32, 32), (64, 128), (128, 128)])
def test_head_and_value_dimension_boundaries(kernel, dtype, head_dim, value_dim):
    rng = np.random.default_rng(148+head_dim+value_dim)
    arrays = [rng.normal(0, .2, shape).astype('float32') for shape in
              [(1, 2, 9, head_dim), (1, 2, 17, head_dim), (1, 2, 17, value_dim),
               (1, 2, 9, value_dim)]]
    inputs = [jt.array(value).cast(dtype).start_grad() for value in arrays[:3]]
    dy = jt.array(arrays[3]).cast(dtype).stop_grad()
    values = [value.float32().numpy() for value in inputs]
    expected = reference(*values, dy.float32().numpy(), False, None)
    output = kernel._efficient_attention(*inputs, allow_reduced_precision=True)
    gradients = jt.grad((output.float32()*dy.float32()).sum(), inputs)
    tolerance = {'float32': 1e-4, 'float16': .004, 'bfloat16': .04}[dtype]
    for actual, wanted in zip([output]+list(gradients), expected):
        actual.sync()
        assert actual.location() == 'device'
        np.testing.assert_allclose(actual.float32().numpy(), wanted, rtol=tolerance,
                                   atol=max(1e-8, np.abs(wanted).max()*tolerance*.2))


def test_captured_training_reads_changing_inputs_and_cotangents(kernel):
    rng = np.random.default_rng(951)

    def step(q, k, v, cotangent):
        output = kernel._efficient_attention(q, k, v)
        return (output, *jt.grad((output*cotangent).sum(), [q, k, v]))

    runner = jt.capture_step(step)
    try:
        for _ in range(8):
            arrays = [rng.normal(0, .3, shape).astype('float32') for shape in
                      [(1, 2, 33, 64), (1, 2, 49, 64), (1, 2, 49, 32), (1, 2, 33, 32)]]
            expected = reference(*arrays, False, None)
            operands = [jt.array(value) for value in arrays]
            operands[-1].stop_grad()
            actual = runner(*operands)
            for value, wanted in zip(actual, expected):
                value.sync()
                assert value.location() == 'device'
                np.testing.assert_allclose(value.numpy(), wanted, rtol=1e-4, atol=2e-6)
        assert runner.refused is None
        assert runner.stats['replayed'] >= 4
        assert runner.stats['graph'] >= 1
    finally:
        runner.invalidate()


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_dropout_graph_reuses_forward_mask_in_backward_and_refreshes_it(kernel, dtype):
    # Identity V reveals each dropout decision, so FP64 softmax and that mask
    # independently determine all three nonzero input gradients.
    rng = np.random.default_rng(536)
    q = jt.array(rng.normal(0, .2, (1, 2, 17, 64)).astype('float32')).cast(dtype).start_grad()
    k = jt.array(rng.normal(0, .2, (1, 2, 64, 64)).astype('float32')).cast(dtype).start_grad()
    identity = np.broadcast_to(np.eye(64, dtype='float32'), (1, 2, 64, 64)).copy()
    v = jt.array(identity).cast(dtype).start_grad()
    cotangent = jt.array(rng.normal(size=(1, 2, 17, 64)).astype('float32')).cast(dtype).stop_grad()
    qn, kn, dyn = [x.float32().numpy().astype('float64') for x in (q, k, cotangent)]
    score = qn @ kn.swapaxes(-1, -2)*.125
    probability = np.exp(score-score.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)

    def step():
        output = kernel._efficient_attention(q, k, v, dropout_p=.25,
                                             allow_reduced_precision=True)
        return (output, *jt.grad((output.float32()*cotangent.float32()).sum(), [q, k, v]))

    runner = jt.capture_step(step)
    masks = []
    try:
        for _ in range(8):
            output, dq, dk, dv = runner()
            actual = output.float32().numpy()
            mask = actual > 0
            masks.append(mask.copy())
            tolerance = {'float32': 1e-5, 'float16': .003, 'bfloat16': .03}[dtype]
            dropped = probability*mask/.75
            np.testing.assert_allclose(actual, dropped, rtol=tolerance, atol=1e-6)
            dp = dyn*mask/.75
            ds = probability*(dp-(probability*dp).sum(-1, keepdims=True))
            expected = [ds @ kn*.125, ds.swapaxes(-1, -2) @ qn*.125,
                        dropped.swapaxes(-1, -2) @ dyn]
            for gradient, wanted in zip((dq, dk, dv), expected):
                np.testing.assert_allclose(gradient.float32().numpy(), wanted,
                                           rtol=tolerance, atol=max(1e-6, np.abs(wanted).max()*tolerance*.2))
        assert .72 < np.mean(masks) < .78
        assert len({mask.tobytes() for mask in masks}) == len(masks), 'dropout mask froze on replay'
        assert runner.refused is None and runner.stats['graph'] >= 1
    finally:
        runner.invalidate()
