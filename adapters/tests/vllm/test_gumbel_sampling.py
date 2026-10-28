"""Minimal vLLM Triton sampling path regression."""

import importlib.util

import numpy as np
import pytest

_VLLM = importlib.util.find_spec("vllm")
_HAS_VLLM = _VLLM is not None and _VLLM.origin is not None


@pytest.fixture(autouse=True)
def _vllm_engine_available():
    # Skip at call time, not module level: the session policy records per-file
    # skip reasons only for test reports, so a module-level skip would report
    # the file as collecting nothing.
    if not _HAS_VLLM:
        pytest.skip("vllm is not installed")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA accelerator")


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
