# vLLM TP1 prefill attribution and guarded full-prefill A/B

Status: diagnostic candidate, opt-in; not a 90% performance claim.
Date: 2026-10-12
Baseline commit: `312d740581b773b14d81495ba564fda80bb09518` (`chk-refactor2-integration`)
Owner: vLLM/Jittor serving integration
Review when: the paged-attention implementation, CUDA Graph ownership, Qwen3 adapter, or vLLM 0.24 forward path changes.

## Protocol

- Server: `cscg-qh00`, one free RTX 4090 (physical GPU 5), TP=1, Qwen3-0.6B, FP16, vLLM 0.24.0, 128 prompt tokens, prefix cache disabled, greedy sampling.
- Independent `/home/chk/.venvs/vllm-jittor` shim and `/home/chk/.venvs/vllm-oracle` native PyTorch environments; same installed vLLM and model, no network model fetch.
- `multigpu_latency.py` uses warm offline `engine.step` token delivery. The A/B Graph run used `FULL_DECODE_ONLY`, one warmup, three measured 32-token requests. The profile run used eager mode, one warmup and one measured 2-token request, with prefill-only instrumentation. Profiler runs are diagnostic and not throughput results.
- Jittor's `jt.profile` records execution-host phases and CUPTI kernel time for the same `model.forward` prefill. Native PyTorch used `torch.profiler` on the same vLLM model-forward boundary. Both profilers perturb wall time; uninstrumented A/B decides speed.

## Attribution

The warmed instrumented model-forward prefill was about 29.25 ms on Jittor versus about 11.19 ms on native PyTorch. Jittor's device was busy for 3.80 ms, launching 955 kernels from 1,827 Jittor operators. Host decomposition was 16.43 ms Python/graph construction, 10.92 ms operator launch, 1.18 ms executor planning, and under 0.5 ms device wait. Native PyTorch's warmed profile reported about 2.27 ms summed CUDA kernel time (the profiler's `self_device_time_total` double-counts events; use its kernel table total instead); FlashAttention's 28 prefill launches took about 0.21 ms device time, versus Jittor's 56 batched matmuls taking about 0.77 ms alone. The model GEMMs were similar: 112 `mm` calls and ~1.64 ms GPU on both sides.

The vLLM input/attention metadata spans were small in this protocol: warmed Jittor `prepare_inputs` roughly 0.65-0.81 ms and `prepare_attn` 0.12 ms; native PyTorch roughly 0.28-0.38 ms and 0.08-0.09 ms. They cannot explain the ~18 ms prefill gap. `sample_tokens` is outside the model-forward boundary and is not the principal TTFT difference.

The primary difference from bare Transformers inference is path-specific: bare Qwen3 uses ordinary SDPA/dynamic cache, while the vLLM Jittor adapter routes prefill through `reshape_and_cache` followed by `jt.nn.paged_attention`, whose current CUDA prefill fallback gathers from the paged cache, casts to FP32, expands GQA heads and constructs separate matmul/softmax operations in every decoder layer. Native vLLM calls a fused FlashAttention forward. Consequently bare-model parity does not imply vLLM prefill parity.

## A/B intervention

`JITTOR_VLLM_DIRECT_FULL_PREFILL=1` enables a guarded, single-request full-prefill fast path. It still writes the KV cache for subsequent decode, but computes attention directly from current Q/K/V with `jt.nn.scaled_dot_product_attention(enable_gqa=True)`. It declines one-token decode, chunked prefill, multiple requests, prefix-attending-to-prior-cache and graph capture without host metadata. Default remains off.

| TP1, 128 prompt + 32 output, Graph decode | Original | Candidate |
| --- | ---: | ---: |
| Median TTFT | 29.971 ms | 28.146 ms |
| Output throughput | 248.03 tok/s | 250.81 tok/s |
| Median ITL | 3.165 ms | 3.194 ms |

All 32 greedy token IDs matched between A/B in the three measured requests. The eager 2-token A/B median TTFT was 29.376 -> 27.890 ms. In the prefill-only Jittor profile, operator count fell 1,827 -> 1,491, kernel count 955 -> 871, GPU busy time 3.80 -> 3.09 ms, and model-forward wall 29.25 -> 27.62 ms. This is a real but modest gain and does not close the performance gap.

Validation: `python adapters/tests/vllm/test_flash_attn.py` in the Jittor CUDA environment: 8 tests passed. The new unit case checks GQA output against a NumPy reference and exclusion of chunked, multi-request and one-token cases. The shim venv has no `pytest`, so the file's `unittest` entry point was used. `git diff --check` passed. The graph A/B used one batch shape and does not prove dynamic-batch or long-context production stability.

Raw JSON evidence lives outside the checkout at `/home/chk/jittor-lab/_state/vllm-compare-20261012/prefill-{detail,cupti,graph-baseline,graph-direct}-j.json`, `prefill-detail-o.json` and `prefill-cupti-o.json`. The native graph baseline with the earlier full protocol is `/home/chk/jittor-lab/_state/bare-model/chk-refactor2-integration/vllm-oracle-tp1-b1b4-r1.json`; do not treat it as a same-run A/B for this candidate.

Next measured target: a fused CUDA variable-length prefill attention operator that consumes fresh Q/K/V and writes KV cache without constructing per-layer gather/matmul/softmax graphs, followed by a same-GPU interleaved benchmark against native vLLM. A 90% throughput claim requires end-to-end, same-protocol validation, not extrapolation from kernel speed.
