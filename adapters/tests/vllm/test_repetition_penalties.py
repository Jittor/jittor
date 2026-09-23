"""vLLM repetition penalties: real tensors, independent arithmetic oracle.

Default execution requires CUDA. Set VLLM_PENALTY_TEST_DEVICE=cpu for the
Jittor CPU primitive check; native vLLM's compiled op only supports CUDA.
The same file can be copied outside this checkout for native Torch verification.
"""

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def runtime():
    import torch

    device = os.environ.get("VLLM_PENALTY_TEST_DEVICE", "cuda")
    if hasattr(torch, "_torch_compat_install_context"):
        from jittor_adapters.vllm import custom_ops
        custom_ops.register(torch)
    else:
        from vllm.platforms import current_platform
        current_platform.import_kernels()  # Native v0.24 stable-libtorch extension.
        assert device == "cuda", "native compiled vLLM oracle requires CUDA"
    return torch, device


def _check(runtime, values, prompt, output, penalties, dtype_name):
    torch, device = runtime
    dtype = getattr(torch, dtype_name)
    logits = torch.tensor(values, dtype=dtype, device=device)
    alias = logits
    prompt_tensor = torch.tensor(prompt, dtype=torch.bool, device=device)
    output_tensor = torch.tensor(output, dtype=torch.bool, device=device)
    penalty_tensor = torch.tensor(penalties, dtype=dtype, device=device)
    # Compute from the actual rounded inputs, including BF16 penalty rounding.
    original = logits.float().cpu().numpy().copy()
    rounded_penalties = penalty_tensor.float().cpu().numpy().copy()
    expected = original.copy()
    for row in range(len(penalties)):
        for col in np.flatnonzero(prompt[row] | output[row]):
            value = original[row, col]
            penalty = rounded_penalties[row]
            expected[row, col] = value / penalty if value > 0 else value * penalty
    expected = torch.tensor(expected, dtype=dtype, device=device).float().cpu().numpy()
    result = torch.ops._C.apply_repetition_penalties_(
        logits, prompt_tensor, output_tensor, penalty_tensor)
    assert result is None
    assert logits is alias
    assert logits.device.type == device
    assert logits.dtype == dtype
    actual = alias.float().cpu().numpy()
    seen = prompt | output
    # CUDA division may differ from the scalar oracle by a final rounding.
    # Unseen tokens must remain exact; tolerance applies only to arithmetic.
    rtol = {"float32": 1e-6, "float16": 1e-3, "bfloat16": 8e-3}[dtype_name]
    np.testing.assert_allclose(actual[seen], expected[seen], rtol=rtol, atol=0)
    np.testing.assert_array_equal(actual[~seen], original[~seen])
    np.testing.assert_array_equal(prompt_tensor.cpu().numpy(), prompt)
    np.testing.assert_array_equal(output_tensor.cpu().numpy(), output)
    np.testing.assert_array_equal(penalty_tensor.float().cpu().numpy(), rounded_penalties)
    return logits


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_signed_logits_union_masks_and_per_request_penalties(runtime, dtype_name):
    values = np.tile(np.array([4, -4, 0, 8, -8, 3, -3, 0.375], np.float32), (5, 1))
    prompt = np.tile([True, False, True, True, False, False, False, True], (5, 1))
    output = np.tile([False, True, False, True, True, False, False, False], (5, 1))
    _check(runtime, values, prompt, output, [2, 1, 0.5, 1.1, 0.7], dtype_name)


@pytest.mark.parametrize("vocab", [17, 50272, 151936])
def test_last_vocab_token_and_non_power_of_two_boundary(runtime, vocab):
    values = np.linspace(-7, 7, vocab, dtype=np.float32)[None, :].copy()
    prompt = np.zeros((1, vocab), dtype=bool)
    output = np.zeros_like(prompt)
    prompt[0, 0] = True
    output[0, -1] = True
    _check(runtime, values, prompt, output, [1.1], "float32")


def test_all_unseen_logits_are_untouched(runtime):
    values = np.array([[1, -2, 0, -np.inf, np.inf]], dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    _check(runtime, values, mask, mask, [1.1], "float32")


def test_empty_batch(runtime):
    values = np.empty((0, 17), dtype=np.float32)
    mask = np.empty((0, 17), dtype=bool)
    _check(runtime, values, mask, mask, [], "float32")


def test_successive_calls_update_the_same_logits(runtime):
    torch, device = runtime
    logits = torch.tensor([[4, -4, 6]], dtype=torch.float32, device=device)
    alias = logits
    prompt = torch.tensor([[True, False, False]], dtype=torch.bool, device=device)
    output = torch.tensor([[False, True, False]], dtype=torch.bool, device=device)
    penalties = torch.tensor([2], dtype=torch.float32, device=device)
    for expected in ([[2, -8, 6]], [[1, -16, 6]]):
        torch.ops._C.apply_repetition_penalties_(logits, prompt, output, penalties)
        np.testing.assert_array_equal(alias.cpu().numpy(), expected)
    assert alias.device.type == device
