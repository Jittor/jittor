"""Exercise the installed vLLM logprob kernel at the vocabulary boundary."""

import numpy as np


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
