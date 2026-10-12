# vLLM TP1 decode CUDA Graph prototype

Date: 2026-10-05

## Scope

- Model: `/home/chk/models/Qwen3-0.6B`, FP16, vLLM 0.24.0.
- Physical GPU: RTX 4090 GPU 3.
- Protocol: input 128 tokens, output 64 tokens, greedy decode.
- The Jittor graph captures the stable `model.forward` decode path. Prefill and
  sampling remain outside the graph, matching the staged rollout plan.
- Capture supports Jittor native lazy operators and compatibility-layer Triton
  launches in one CUDA graph. Input, metadata, KV-cache and output allocations
  retain stable addresses between replays.

## Result

| Batch | Runtime | Throughput (tok/s) | Median ITL (ms) | P95 ITL (ms) | TTFT (ms) |
|---:|---|---:|---:|---:|---:|
| 1 | Jittor eager | 58.649 | 16.730 | 17.039 | 38.517 |
| 1 | Jittor CUDA Graph | 189.293 | 4.716 | 5.133 | 39.207 |
| 1 | PyTorch CUDA Graph | 407.082 | 2.301 | 2.372 | 10.840 |
| 4 | Jittor CUDA Graph | 706.734 | 5.012 | 5.285 | 44.349 |
| 4 | PyTorch CUDA Graph | 1511.621 | 2.506 | 2.539 | 10.737 |

For batch 1, CUDA Graph improves Jittor median decode ITL by 71.8% (3.55x)
and throughput by 222.8% (3.23x). This exceeds the 15% acceptance target.
The batch-4 path is stable and delivers 706.7 tok/s; the remaining Jittor to
PyTorch decode gap is about 2.4 ms per step and is no longer dominated by
Python/lazy-graph submission of every model layer.

## Implementation

- Implemented `torch.cuda.CUDAGraph`, `torch.cuda.graph` and graph pool handles
  on top of Jittor graph capture/replay/release.
- Submitted the lazy native tail into the capture without a blocking drain.
- Added capture-aware Triton launch handling so external kernels are recorded
  instead of treated as graph barriers. Dynamic scratch allocation is rejected
  during capture because graph replay requires stable addresses.
- Deferred frees of capture-time allocations until graph release.
- Added an optional CUDA graph DOT dump. The Qwen graph contains 283 nodes and
  spans embedding, 28 decoder layers and final normalization.
- Added a regression test whose graph contains a Jittor native producer and a
  Triton consumer and verifies changing input values across replays.

## Correctness and validation

- Jittor eager, Jittor CUDA Graph and PyTorch CUDA Graph produced exactly the
  same 64 greedy token IDs for batch 1.
- For batch 4, every output contains 64 tokens and all four sequences match the
  reference in both warmup and measured runs.
- The mixed native/Triton CUDA Graph regression test passed.
- Python compilation and `git diff --check` passed.
- A temporary experiment that cleared `seq_lens` inside capture was removed:
  it changed the second generated token and could later trigger an illegal
  address. Metadata contents must be updated by vLLM, not rewritten by the
  graph adapter.

The benchmark JSON files may say `status: failed` because the harness asserts
during shutdown even though generation completed, worker exit code is zero,
and no shared-memory objects remain. The timings and output checks above are
from the completed generation section.

## Evidence

- Jittor eager batch 1: `/home/chk/jittor-lab/_state/vllm-tp/20261005/tp1-baseline-64.json`
- Jittor CUDA Graph batch 1: `/home/chk/jittor-lab/_state/vllm-tp/20261005/tp1-cudagraph-final-64.json`
- PyTorch CUDA Graph batch 1: `/home/chk/jittor-lab/_state/vllm-tp/20261005/tp1-oracle-cudagraph-64.json`
- Jittor CUDA Graph batch 4: `/home/chk/jittor-lab/_state/vllm-tp/20261005/tp1-cudagraph-b4-64.json`
- PyTorch CUDA Graph batch 4: `/home/chk/jittor-lab/_state/vllm-tp/20261005/tp1-oracle-cudagraph-b4-64.json`
- Captured graph: `/tmp/jittor-vllm-graph.dot`

