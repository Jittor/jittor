"""Microbenchmark actual vLLM paged decode attention and greedy sampling.

Run this file once in the native PyTorch vLLM environment and once in the
Jittor torch-shim environment, on the same physical GPU.  Every sample is
bounded by a CUDA synchronization.  This is a diagnostic kernel benchmark,
not an end-to-end throughput result.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def timed(call, synchronize, warmups, repetitions):
    for _ in range(warmups):
        call()
        synchronize()
    samples = []
    for _ in range(repetitions):
        synchronize()
        start = time.perf_counter_ns()
        call()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return distribution(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("oracle", "jittor"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--vocab-size", type=int, default=151936)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        parser.error("output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "jittor":
        import jittor as jt

        jt.flags.use_parallel_op_compiler = 0
        jt.flags.use_cuda = 1
    import torch

    is_shim = hasattr(torch, "_torch_compat_install_context")
    assert is_shim == (args.backend == "jittor")
    assert torch.cuda.is_available()
    synchronize = torch.cuda.synchronize

    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    report = {
        "status": "running",
        "backend": args.backend,
        "arguments": vars(args),
        "scope": "synchronized isolated CUDA calls; not end-to-end latency",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "measurements": [],
    }

    rng = np.random.default_rng(20261001)
    q_heads, kv_heads, head_dim, block_size = 16, 8, 128, 64
    scale = head_dim**-0.5

    for batch in args.batches:
        logits_np = rng.standard_normal((batch, args.vocab_size), dtype=np.float32)
        logits = torch.tensor(logits_np, device="cuda", dtype=torch.float32)
        mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
        temperature = torch.zeros(batch, device="cuda", dtype=torch.float32)
        seeds = torch.arange(batch, device="cuda", dtype=torch.int64) + 17
        positions = torch.arange(batch, device="cuda", dtype=torch.int64) + 127

        sampled_holder = [None]

        def run_gumbel():
            sampled_holder[0] = gumbel_sample(
                logits,
                mapping,
                temperature,
                seeds,
                positions,
                apply_temperature=False,
            )

        def run_argmax():
            sampled_holder[0] = logits.argmax(dim=-1)

        gumbel_timing = timed(run_gumbel, synchronize, args.warmups, args.repetitions)
        gumbel_tokens = [int(v) for v in sampled_holder[0].cpu().tolist()]
        argmax_timing = timed(run_argmax, synchronize, args.warmups, args.repetitions)
        argmax_tokens = [int(v) for v in sampled_holder[0].cpu().tolist()]
        expected = logits_np.argmax(axis=-1).astype(np.int64).tolist()
        assert gumbel_tokens == expected
        assert argmax_tokens == expected
        report["measurements"].append(
            {
                "case": "greedy_sampling",
                "batch": batch,
                "gumbel_sample": gumbel_timing,
                "direct_argmax": argmax_timing,
                "token_ids": expected,
            }
        )

        for sequence_length in args.sequence_lengths:
            used_blocks = math.ceil(sequence_length / block_size)
            total_blocks = batch * used_blocks
            query_np = (
                rng.standard_normal((batch, q_heads, head_dim), dtype=np.float32)
                * 0.2
            ).astype(np.float16)
            cache_np = (
                rng.standard_normal(
                    (total_blocks, 2, block_size, kv_heads, head_dim),
                    dtype=np.float32,
                )
                * 0.2
            ).astype(np.float16)
            table_np = np.arange(total_blocks, dtype=np.int32).reshape(
                batch, used_blocks
            )
            lengths_np = np.full(batch, sequence_length, dtype=np.int32)
            starts_np = np.arange(batch + 1, dtype=np.int32)
            query = torch.tensor(query_np, device="cuda", dtype=torch.float16)
            cache = torch.tensor(cache_np, device="cuda", dtype=torch.float16)
            block_table = torch.tensor(table_np, device="cuda", dtype=torch.int32)
            seq_lens = torch.tensor(lengths_np, device="cuda", dtype=torch.int32)
            cu_seqlens_q = torch.tensor(starts_np, device="cuda", dtype=torch.int32)
            attention_holder = [None]

            if args.backend == "jittor":
                import jittor as jt

                def run_attention():
                    attention_holder[0] = jt.nn.paged_attention(
                        query,
                        cache,
                        cu_seqlens_q,
                        seq_lens,
                        block_table,
                        scale=scale,
                        causal=True,
                        query_lengths=starts_np.tolist(),
                        key_lengths=lengths_np.tolist(),
                    )

            else:
                from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

                key_cache, value_cache = cache.unbind(1)
                attention_output = torch.empty_like(query)

                def run_attention():
                    attention_holder[0] = flash_attn_varlen_func(
                        q=query,
                        k=key_cache,
                        v=value_cache,
                        out=attention_output,
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=1,
                        seqused_k=seq_lens,
                        max_seqlen_k=sequence_length,
                        softmax_scale=scale,
                        causal=True,
                        block_table=block_table,
                        fa_version=2,
                    )

            attention_timing = timed(
                run_attention, synchronize, args.warmups, args.repetitions
            )
            attention_np = np.asarray(attention_holder[0].float().cpu().numpy())
            assert np.isfinite(attention_np).all()
            report["measurements"].append(
                {
                    "case": "paged_decode_attention",
                    "batch": batch,
                    "sequence_length": sequence_length,
                    "timing": attention_timing,
                    "output": {
                        "sum": float(attention_np.sum(dtype=np.float64)),
                        "l2": float(np.linalg.norm(attention_np.astype(np.float64))),
                        "first8": attention_np.reshape(-1)[:8].tolist(),
                    },
                }
            )

    report["status"] = "completed"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
