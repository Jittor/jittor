"""Minimal vLLM Triton sampling path regression."""

import numpy as np


def test_gumbel_sample_smoke():
    import torch
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    logits = torch.tensor(np.array([[.1, 1.2, -.4, 2., .3]], dtype=np.float32))
    expanded = torch.tensor(np.array([0], dtype=np.int32))
    temperature = torch.tensor(np.ones(1, dtype=np.float32))
    seed = torch.tensor(np.array([13], dtype=np.int64))
    pos = torch.tensor(np.array([0], dtype=np.int64))
    sampled = gumbel_sample(logits, expanded, temperature, seed, pos, False)
    assert sampled.shape == (1,)
