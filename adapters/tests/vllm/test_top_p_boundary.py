"""An observed model-rounding boundary must not be 'fixed' in the sampler.

Fixtures retain the largest 40 raw logits at the first divergent Qwen step.
Both pipelines must produce the same result for each fixed input, even though
the two inputs intentionally fall on opposite sides of the nucleus boundary.
"""
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('source,expected_sample', [('oracle', 8251), ('jittor', 8171)])
def test_captured_nucleus_boundary(source, expected_sample):
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample

    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'top_p_boundary.json').read_text())[source]
    ids = np.asarray(fixture['ids'])
    values = np.asarray(fixture['raw'], dtype=np.float64) / fixture['temperature']
    order = np.argsort(values)[::-1]
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    # Independent float64 nucleus reference: include the token crossing p.
    count = np.searchsorted(np.cumsum(probabilities[order]), fixture['top_p']) + 1
    expected_kept = np.sort(ids[order[:count]])
    np.testing.assert_array_equal(expected_kept, fixture['expected_kept'])
    raw = np.full((1, 151936), -np.inf, dtype=np.float32)
    raw[0, ids] = fixture['raw']

    def tensor(value, dtype):
        return torch.tensor(value, dtype=dtype, device='cuda')

    logits = tensor(raw, torch.float32)
    mapping = tensor([0], torch.int32)
    temperatures = tensor([fixture['temperature']], torch.float32)
    apply_temperature(logits, mapping, temperatures)
    filtered = apply_top_k_top_p(logits, tensor([40], torch.int32),
                                tensor([fixture['top_p']], torch.float32))
    assert filtered.device.type == 'cuda'
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(filtered.cpu().numpy()[0])), expected_kept)
    sampled = gumbel_sample(filtered, mapping, temperatures,
                           tensor([fixture['seed']], torch.int64),
                           tensor([fixture['position']], torch.int64),
                           apply_temperature=False)
    assert sampled.device.type == 'cuda'
    assert sampled.cpu().tolist() == [expected_sample]
