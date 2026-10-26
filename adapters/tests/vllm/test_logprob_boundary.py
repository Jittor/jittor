"""Exercise the installed vLLM logprob kernel at the vocabulary boundary."""

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


def test_last_valid_token_logprob():
    import torch
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    vocab = 151936
    values = np.zeros((1, vocab), dtype=np.float32)
    values[0, -1] = 2.0
    logits = torch.tensor(values, device="cuda", dtype=torch.float32)
    top = torch.topk(logits, 1, dim=-1)
    assert top.indices.cpu().numpy().tolist() == [[vocab - 1]]
    ids = torch.tensor([[0, vocab - 1]], device="cuda", dtype=torch.int64)
    actual = compute_token_logprobs(logits, ids).cpu().numpy()
    logsum = np.log(vocab - 1 + np.exp(2.0))
    np.testing.assert_allclose(actual, [[-logsum, 2.0 - logsum]], atol=2e-5)
