"""Replay captured Qwen prefill attention on real CUDA with an independent float64 reference.

No acceptance tolerance is imposed: this reports errors, rounding and exact
reproduction of the captured operator. Host NumPy is used only as a reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def error(actual, reference):
    delta = actual.astype(np.float64) - reference
    return dict(max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta ** 2))),
                different=int(np.count_nonzero(delta)), elements=int(delta.size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['jittor', 'oracle'], required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.backend == 'jittor':
        import jittor as jt
        jt.flags.use_cuda = 1
        jt.flags.use_parallel_op_compiler = 0
    import torch
    assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
    saved = np.load(args.input)
    name = 'model.layers.0.self_attn.attn'
    q0 = saved[name + '.input.0'].reshape(-1, 16, 128)
    k0 = saved[name + '.input.1'].reshape(-1, 8, 128)
    v0 = saved[name + '.input.2'].reshape(-1, 8, 128)
    size = q0.shape[0]
    q, k, v = [torch.tensor(x, dtype=torch.float16, device='cuda') for x in (q0, k0, v0)]
    cu = torch.tensor([0, size], dtype=torch.int32, device='cuda')
    with torch.inference_mode():
        if args.backend == 'jittor':
            cache = torch.zeros((1, 2, 16, 8, 128), dtype=torch.float16, device='cuda')
            slots = torch.arange(size, dtype=torch.int64, device='cuda')
            seq = torch.tensor([size], dtype=torch.int32, device='cuda')
            blocks = torch.tensor([[0]], dtype=torch.int32, device='cuda')
            jt.nn.reshape_and_cache(k, v, cache, slots)
            output = jt.nn.paged_attention(q, cache, cu, seq, blocks, scale=128 ** -.5,
                                           causal=True)
        else:
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            output = flash_attn_varlen_func(q, k, v, size, cu, size, cu,
                                            softmax_scale=128 ** -.5, causal=True, fa_version=2)
    assert output.device.type == 'cuda'
    actual = output.detach().cpu().numpy().reshape(size, 2048)
    q64 = q0.astype(np.float64).transpose(1, 0, 2)
    k64 = np.repeat(k0.astype(np.float64), 2, axis=1).transpose(1, 0, 2)
    v64 = np.repeat(v0.astype(np.float64), 2, axis=1).transpose(1, 0, 2)
    scores = q64 @ k64.transpose(0, 2, 1) * (128 ** -.5)
    scores = np.where(np.triu(np.ones((size, size), dtype=bool), 1), -np.inf, scores)
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    denominator = exponentials.sum(axis=-1, keepdims=True)
    reference = ((exponentials / denominator) @ v64).transpose(1, 0, 2).reshape(size, 2048)
    # This alternate numerical evaluation is diagnostic, never a proposed fix.
    exp_half = ((exponentials.astype(np.float16).astype(np.float64) @ v64) /
                denominator).transpose(1, 0, 2).reshape(size, 2048)
    report = dict(backend=args.backend, device=str(output.device), input=args.input,
                  vs_float64=error(actual, reference),
                  vs_rounded_float64=error(actual, reference.astype(np.float16)),
                  vs_half_exponential_reference=error(actual, exp_half.astype(np.float16)),
                  vs_capture=error(actual, saved[name + '.output']))
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(prefix) + '.npz', actual=actual, float64_reference=reference,
                        half_exponential_reference=exp_half)
    Path(str(prefix) + '.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
