"""Type probes for vLLM's packed prompt penalty mask."""

import numpy as np


def _run_packed_kernel(vocab_size, block_size):
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(prompt, output, prompt_stride, vocab, BLOCK: tl.constexpr):
        block_idx = tl.program_id(1)
        packed_block = block_idx * BLOCK // 32 + tl.arange(0, BLOCK // 32)
        packed_mask = tl.load(
            prompt + packed_block,
            mask=packed_block < tl.cdiv(vocab, 32),
            other=0,
        )
        bits = (packed_mask[:, None] >> tl.arange(0, 32)[None, :]) & 1
        bits = bits.to(tl.int1).reshape(BLOCK)
        offsets = tl.arange(0, BLOCK)
        tl.store(output + offsets, bits.reshape(BLOCK).to(tl.int32))

    packed = (vocab_size + 31) // 32
    values = np.zeros((packed,), dtype=np.int32)
    values[:2] = [1, 2]
    prompt = torch.tensor(values, dtype=torch.int32, device="cuda")
    output = torch.zeros((block_size,), dtype=torch.int32, device="cuda")
    kernel[(1, 1)](prompt, output, packed, vocab_size, BLOCK=block_size)
    expected = np.zeros(block_size, dtype=np.int32)
    expected[0] = 1
    if block_size > 33:
        expected[33] = 1
    np.testing.assert_array_equal(output.cpu().numpy(), expected)
    return prompt, output


def test_prompt_mask_tensor_shape_stride_device_dtype():
    prompt, output = _run_packed_kernel(151936, 32)
    assert tuple(prompt.shape) == (4748,)
    assert prompt.is_contiguous()
    assert str(prompt.dtype) == "torch.int32"
    assert str(prompt.device) == "cuda:0"
    assert str(output.dtype) == "torch.int32"


def test_penalties_exact_large_block_masked_load():
    prompt, output = _run_packed_kernel(151936, 8192)
    assert tuple(prompt.shape) == (4748,)
    assert str(prompt.dtype) == "torch.int32"
    assert output.shape == (8192,)


def test_penalties_small_vocab_masked_tail():
    prompt, output = _run_packed_kernel(34, 8192)
    assert tuple(prompt.shape) == (2,)
    assert output.shape == (8192,)
