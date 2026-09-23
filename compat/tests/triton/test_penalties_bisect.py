"""A-E bisect of vLLM penalty-kernel integer expressions."""

import numpy as np
import jittor as jt


def _inputs(vocab=151936):
    packed = (vocab + 31) // 32
    prompt = jt.array(np.zeros(packed, dtype=np.int32))
    output = jt.zeros((8192,), dtype="int32")
    logits = jt.zeros((8192,), dtype="float32")
    return prompt, output, logits


def test_a_packed_mask_shift():
    import triton
    import triton.language as tl
    @triton.jit
    def k(prompt, out, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK // 32)
        p = tl.load(prompt + o, mask=o < 4748, other=0)
        b = (p[:, None] >> tl.arange(0, 32)[None, :]) & 1
        tl.store(out + tl.arange(0, BLOCK), b.reshape(BLOCK).to(tl.int32))
    p, o, _ = _inputs(); k[(1,)](p, o, BLOCK=8192)


def test_b_output_count_load_compare():
    import triton
    import triton.language as tl
    @triton.jit
    def k(counts, out, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK); x = tl.load(counts + o, mask=o < 8192, other=0)
        tl.store(out + o, (x > 0).to(tl.int32), mask=o < BLOCK)
    _, o, _ = _inputs(); k[(1,)](o, o, BLOCK=8192)


def test_c_token_match_cast_and_loop():
    import triton
    import triton.language as tl
    @triton.jit
    def k(counts, out, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK); x = tl.load(counts + o, mask=o < BLOCK, other=0)
        acc = x
        for i in tl.range(2):
            acc = acc + (o == i).to(tl.int32)
        tl.store(out + o, (acc > 0).to(tl.int32), mask=o < BLOCK)
    _, o, _ = _inputs(); k[(1,)](o, o, BLOCK=8192)


def test_d_dynamic_repetition_branch():
    import triton
    import triton.language as tl
    @triton.jit
    def k(prompt, logits, out, USE_REP: tl.constexpr, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK); x = tl.load(logits + o, mask=o < BLOCK, other=0.)
        if USE_REP:
            packed = tl.arange(0, BLOCK // 32)
            p = tl.load(prompt + packed, mask=packed < 4748, other=0)
            bits = ((p[:, None] >> tl.arange(0, 32)[None, :]) & 1).reshape(BLOCK)
            x = tl.where(bits, x * 2., x)
        tl.store(out + o, x, mask=o < BLOCK)
    p, o, logits = _inputs(); k[(1,)](p, logits, o, True, BLOCK=8192)


def test_e_full_parameter_layout():
    import triton
    import triton.language as tl
    @triton.jit
    def k(prompt, counts, logits, out, prompt_stride, count_stride,
          vocab, USE_REP: tl.constexpr, BLOCK: tl.constexpr):
        token = tl.program_id(0); o = tl.arange(0, BLOCK)
        x = tl.load(logits + token * BLOCK + o, mask=o < vocab, other=0.)
        base = tl.load(counts + token * count_stride + o, mask=o < vocab, other=0)
        if USE_REP:
            packed = tl.arange(0, BLOCK // 32)
            pm = tl.load(prompt + token * prompt_stride + packed,
                         mask=packed < tl.cdiv(vocab, 32), other=0)
            bits = ((pm[:, None] >> tl.arange(0, 32)[None, :]) & 1).reshape(BLOCK)
            x = tl.where(bits | (base > 0), x * 2., x)
        tl.store(out + token * BLOCK + o, x, mask=o < vocab)
    p, o, logits = _inputs(); k[(1,)](p, o, logits, logits, 4748, 8192, 151936, True, BLOCK=8192)


def test_f_full_penalties_kernel_expression():
    import triton
    import triton.language as tl
    @triton.jit
    def k(logits_ptr, logits_stride, output_ptr, output_stride, prompt_ptr,
          prompt_stride, token_ids_ptr, expanded_pos_ptr, rep_penalty,
          freq_penalty, pres_penalty, vocab_size, USE_REP: tl.constexpr,
          USE_FREQ: tl.constexpr, USE_PRES: tl.constexpr, BLOCK: tl.constexpr):
        token_idx = tl.program_id(0); block_idx = tl.program_id(1)
        block = block_idx * BLOCK + tl.arange(0, BLOCK)
        mask = block < vocab_size
        logits = tl.load(logits_ptr + token_idx * logits_stride + block,
                         mask=mask, other=0.).to(tl.float32)
        base = tl.load(output_ptr + token_idx * output_stride + block,
                       mask=mask, other=0)
        pos = tl.load(expanded_pos_ptr + token_idx)
        start_idx = token_idx - pos
        counts = base
        for prev_pos in tl.range(pos):
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            counts = counts + (block == prev_token).to(tl.int32)
        output_bin_mask = counts > 0
        if USE_REP:
            packed = block_idx * BLOCK // 32 + tl.arange(0, BLOCK // 32)
            packed_mask = tl.load(prompt_ptr + packed,
                                  mask=packed < tl.cdiv(vocab_size, 32), other=0)
            prompt_bin_mask = (packed_mask[:, None] >> tl.arange(0, 32)[None, :]) & 1
            prompt_bin_mask = prompt_bin_mask.to(tl.int1).reshape(BLOCK)
            scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.)
            logits *= tl.where(logits > 0, 1. / scale, scale)
        if USE_FREQ:
            logits -= freq_penalty * counts
        if USE_PRES:
            logits -= tl.where(output_bin_mask, pres_penalty, 0.)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)

    p, counts, logits = _inputs()
    pos = jt.array(np.array([0], dtype=np.int32))
    tokens = jt.zeros((8192,), dtype="int32")
    k[(1, 1)](logits, 8192, counts, 8192, p, 4748, tokens, pos,
              1.2, .1, .1, 151936, True, True, True, BLOCK=8192)
