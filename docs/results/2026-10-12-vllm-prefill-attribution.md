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

The warmed instrumented model-forward prefill was about 29.25 ms on Jittor versus about 11.19 ms on native PyTorch. Jittor's device was busy for 3.80 ms, launching 955 kernels from 1,827 Jittor operators. Host decomposition was 16.43 ms Python/graph construction, 10.92 ms operator launch, 1.18 ms executor planning, and under 0.5 ms device wait. Native PyTorch's warmed profile reported about 2.27 ms summed CUDA kernel time (the profiler's `self_device_time_total` double-counts events; use its kernel table total instead); FlashAttention's 28 prefill launches took about 0.21 ms device time, versus Jittor's 56 batched matmuls taking about 0.77 ms alone. The model GEMMs were similar: 112 `mm` calls and ~1.64 ms GPU on both sides. These earlier cross-runtime profiles used different Transformers versions (Jittor 5.5.3, PyTorch 5.17.0); treat their numerical comparison as exploratory, not version-matched proof. The follow-up below controls this mismatch.

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

## Controlled attention/cache ablation (follow-up)

This follow-up starts from the diagnostic-branch commit `90fe4657e` and tests causality rather than relying on profiler labels. The diagnostic `JT_PREFILL_ATTENTION_BYPASS=1` replaces **both** backends' 28 per-layer attention outputs with zeros and skips their KV-cache updates. Everything else in the vLLM first-token step remains active. Because decode would consume an invalid cache, the runner rejects this mode unless `--output-tokens 1`. The intervention's output is not a correctness result. Both backends produced token `1879` before intervention and `46944` after it, and both recorded exactly 252 forward and 252 cache-update bypass calls (9 requests × 28 layers) per process.

Same physical RTX 4090 (GPU 5), same 128-token prompt, FP16 Qwen3-0.6B, vLLM 0.24.0, Transformers 5.5.3, TP1 eager, prefix cache disabled, 2 warmups + 7 measured requests per fresh process. Two fresh processes were run per case; the version-matched PyTorch reruns occurred after the Jittor runs, so the pass numbers below are descriptive rather than randomized temporal pairs. No other compute process occupied GPU 5 during the runs. The native PyTorch environment used a temporary, read-only package overlay for Transformers 5.5.3 and its compatible tokenizer; its installed vLLM and Torch packages were unchanged. All eight reports have the same benchmark script hash, prompt tokens, model, dtype and shape; all completed with clean worker shutdown. Times are median offline first-token delivery, not pure model or GPU kernel times.

| Pass | Jittor original | PyTorch original | Jittor bypass | PyTorch bypass | Gap removed by attention/cache bypass | Remaining gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.015 ms | 10.245 ms | 16.075 ms | 8.411 ms | 11.106 ms | 7.664 ms |
| 2 | 29.748 ms | 10.486 ms | 15.493 ms | 8.299 ms | 12.069 ms | 7.194 ms |

About 11.1–12.1 ms (59–63%) of the cross-runtime gap is associated with replacing the attention/cache **path** in this intervention, not necessarily the FlashAttention GPU kernel itself: removing the path also removes its CPU graph construction and submissions. Jittor saved 12.94–14.26 ms; PyTorch saved only 1.83–2.19 ms. **Do not interpret the displayed 7.19–7.66 ms as the cost of the remaining real model path.** The replacement itself calls `output.fill_(0)` 28 times. On Jittor CUDA this goes through `_copy_into` and imposes substantial graph publication/copy cost; the two frameworks do not pay the same replacement cost. Zero attention also changes network values, so this is diagnostic intervention, not a valid accelerated implementation or a numerical-parity claim. The previously measured opt-in direct-QKV path retained correctness but saved only ~1.8 ms TTFT, consistent with removing only part of the attention/cache overhead.

Matched raw reports: `/home/chk/jittor-lab/_state/vllm-compare-20261012/ablation-j-{baseline,bypass}-1tok{,-r2}.json` and `ablation-o-{baseline,bypass}-matched-r{1,2}.json`. The older `ablation-o-*-1tok*.json` used Transformers 5.17.0 and is excluded from the table. Each report records configuration, prompt and output tokens, timing distribution, bypass hit counts where enabled, and clean worker shutdown. The benchmark helper now permits one output token (ITL is `null`) and fails closed if attention bypass is requested with more than one output token. With only two fresh processes per case and seven measured requests per process, the numerical interval is diagnostic rather than a formal performance acceptance run.

## Non-attention follow-up: correct the bypass artifact

On physical GPU 0, with the same matched software versions and 2 warmups + 7 measured first-token requests, the Jittor-only diagnostic option `JT_PREFILL_BYPASS_ZERO_CODE=1` writes zero to the *same output buffer* using a CUDA code op instead of `fill_(0)`. This still excludes attention and cache and is **not** a production optimization. Both Jittor bypass variants and PyTorch returned token `46944`, and each invoked 28 bypassed attention layers per request.

| Attention/cache replaced with zero | Median first-token delivery |
| --- | ---: |
| Jittor, old `fill_(0)` replacement | 15.130 ms |
| Jittor, direct CUDA zero replacement | 11.847 ms |
| PyTorch, `fill_(0)` replacement | 8.458 ms |

Thus the artificial Jittor zero-fill path accounts for 3.28 ms of the apparent bypass residual in this same-GPU comparison. The remaining gap is 3.39 ms, **not** the earlier 7.2–7.7 ms estimate. This is still an intervention measurement, not a clean decomposition of the original prefill. A separately synchronized phase trace (itself perturbing TTFT) found Jittor/PyTorch median model-forward 5.709/6.467 ms, `prepare_inputs` 0.552/0.245 ms, `prepare_attn` 0.086/0.049 ms, and `sample_tokens` 2.716/0.571 ms. These phase medians must not be added or subtracted from uninstrumented TTFT. The Jittor model-forward cProfile with the old replacement spent 15 ms across 140 `fill_(0)` calls in five forwards; its 28-layer fused-normalization, rotary and dispatch paths were also visible, but this cProfile perturbs execution and cannot prove those are the remaining 3.39 ms. The better zero replacement and phase trace together point primarily to sampling and host/metadata/scheduler work *outside* model forward, once attention/cache have been excluded.

Two opt-in sampling changes were rejected after actual generation A/B, not retained in the adapter: a guarded direct plain-greedy argmax produced 94.47 -> 92.43 tok/s; replacing vLLM's temporary FP32 `empty_like(...).copy_(...)` with an equivalent cast/clone produced 94.47 -> 90.47 tok/s. A completed-prefill count-cache candidate kept the token identical but moved median TTFT 28.740 -> 28.802 ms. The `copy_` shows up as a synchronization/drain point in cProfile, but eliminating that call does not eliminate the underlying pending work; the measured end-to-end results reject it as a speed fix. A diagnostic attempt to select `FULL` CUDA Graph for prefill was automatically downgraded by vLLM to `FULL_DECODE_ONLY` because this FlashAttention metadata builder reports `UNIFORM_BATCH`; therefore that experiment did not capture prefill and is not a performance result.

For **real, non-ablated** 128-prompt + 32-output TP1 eager generation on the same GPU 0, vLLM 0.24.0 and Transformers 5.5.3, Jittor/PyTorch median TTFT was 28.537/10.531 ms, median ITL 9.971/10.032 ms, and aggregate throughput 94.467/99.398 tok/s (Jittor is 95.0% of PyTorch). All 32 generated token IDs matched. This establishes that the large eager deficit is concentrated in prefill/first token, **not** steady-state eager decode throughput.

A fresh fixed-batch-1 `FULL_DECODE_ONLY` CUDA Graph run on the same GPU, versions and shape found Jittor/PyTorch median TTFT 28.922/11.256 ms, median ITL 3.157/2.308 ms, and aggregate throughput 248.571/383.947 tok/s (64.7% of PyTorch), again with all 32 tokens identical. This is *not* 90% Graph throughput parity. Wrapping runner phases to attribute the Graph gap is too invasive: a light `JT_PHASE_TRACE_ASYNC=1` run increased Jittor ITL from 3.157 to 6.770 ms, while the analogous PyTorch trace was near its baseline. The Jittor subphase wrapper similarly gave 6.720 ms. Consequently the Graph phase spans cannot establish how much of the 0.849 ms uninstrumented ITL gap is replay, sampling or metadata. A lower-perturbation CUDA/CPU timeline or separate A/B interventions are required before changing replay internals. The present work does not reach 90% *TTFT* or Graph-throughput parity; no unverified sampling change is left enabled.

Raw reports under `/home/chk/jittor-lab/_state/vllm-compare-20261012/`: `ablation-{j-fill,j-zero-code,o-fill}-gpu0.json`, `ablation-j-zero-code-phase-gpu0.json`, `residual-phase-o.json`, `plain-greedy-{baseline,candidate}-j-eager32.json`, `fast-sampling-copy-j-eager32.json`, `prefill-count-{baseline,candidate}-j.json`, `oracle-matched-gpu0-eager32.json`, `{jittor,oracle}-matched-gpu0-graph32.json`, `{jittor,oracle}-matched-gpu0-graph32-phase.json`, and `jittor-matched-gpu0-graph32-phaselight.json`. Only the diagnostic zero-code bypass and forward-profiler switch remain in the benchmark helper; the failed production candidates were rolled back.

## Guarded prefill and Graph-decode follow-up

The existing fused one-token cache-write/paged-decode CUDA kernel was previously
unreachable during vLLM CUDA Graph capture: the adapter required host-side
`starts` and `lengths`, which are deliberately absent while recording a graph.
For a full batch-1 decode, `num_actual_tokens`, `max_query_len`, and static
metadata tensor sizes establish the one-token condition without reading GPU
metadata back to the CPU. The adapter now auto-selects this fused path only
for that capture shape. The existing `JITTOR_VLLM_FUSED_CACHE_DECODE=0` disables
it; `=1` retains the explicit eager opt-in. The guarded direct-Q/K/V full
prefill path remains opt-in with `JITTOR_VLLM_DIRECT_FULL_PREFILL=1`, because
its model-general correctness has not been established. Chunked,
prefix-attending, multi-request, and one-token prefill decline its guard.

Same GPU 0, Qwen3-0.6B FP16, TP1, vLLM 0.24.0, Transformers 5.5.3,
128-prompt + 32-output, `FULL_DECODE_ONLY`, batch 1, 2 warmups and 7 measured
requests per process. The baseline has both candidate paths disabled; the
fused-only A/B was run in baseline/candidate/candidate/baseline order. Times
below are medians in ms, except throughput.

| Path | TTFT | ITL | Whole request | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline, first | 29.459 | 3.185 | 129.172 | 248.008 |
| Fused decode, first | 29.742 | 3.125 | 126.759 | 251.156 |
| Fused decode, reverse | 29.234 | 3.152 | 127.875 | 250.200 |
| Baseline, reverse | 29.941 | 3.185 | 128.926 | 247.688 |
| Final default (Graph batch-1 fusion only) | 28.494 | 3.129 | 126.235 | 253.271 |
| Both guarded paths, Qwen3 prefill opt-in | 27.568 | 3.152 | 125.901 | 253.935 |

All seven measured requests in each run produced the same 32 token IDs.
Batch 4 did **not** benefit from the fused Graph path: its baseline/candidate
ITL was 3.711/3.737 ms and whole-request time 145.700/147.587 ms, so the
automatic Graph guard is restricted to batch 1. A separate 64-output-token
run with fused Graph decode and the Qwen3 prefill opt-in matched native
PyTorch on every token in all three
measured requests. Its Jittor/PyTorch median whole-request time was
228.149/157.729 ms, TTFT 28.883/11.246 ms, and ITL 3.151/2.319 ms.

The ~2-3% batch-1 improvement is real in these fixed-shape A/B runs but far
from 90% PyTorch throughput. This does not establish dynamic-batch Graph
stability or isolate the remaining ~0.83 ms/token Graph decode deficit.
The benchmark launcher used `JITTOR_VLLM_TEXT_ONLY_BENCH=1` to skip an
unrelated torchvision/MiniMax import in the installed shim environment; it
changes startup only and is off by default. No third-party package was edited.
The adapter's 8 existing attention tests, a new shape-guard test, and the
FP16/BF16 fused-cache kernel reference test passed; `git diff --check` passed.

Raw JSON: `graph-fusion-{baseline-j-20261012-c,candidate-j-20261012-a,candidate-j-20261012-b,baseline-j-20261012-d,final-default-b1-20261012,default-b1-20261012,baseline-b4-20261012,candidate-b4-20261012,default-b1-64tok-20261012,oracle-b1-64tok-20261012}.json` under `/home/chk/jittor-lab/_state/vllm-compare-20261012/`.

## Guarded plain-decoder prefill default (second follow-up)

The previous direct-Q/K/V full-prefill path is now selected by default only
when the vLLM attention implementation has ordinary causal decoder semantics:
no sliding window, ALiBi, logit soft cap, attention sinks, KV sharing,
quantized KV cache or distributed context parallelism. Metadata must also
exclude cascade and prefix-LM cases. The existing full single-request guard
still excludes chunked prefill and prefix-cache hits. An explicit
`JITTOR_VLLM_DIRECT_FULL_PREFILL=0` restores the old paged prefill route;
`=1` retains the original opt-in behavior for experiments. This is a
Qwen3-0.6B validation, not proof for every vLLM model/configuration.

On the same physical GPU 0 and 128+32 Graph-decode protocol above, two
fresh-process baseline/candidate pairs were run in ABBA order. Baseline
explicitly set `JITTOR_VLLM_DIRECT_FULL_PREFILL=0`; candidate used the new
default. All 7 measured requests in every process emitted exactly the same
32 token IDs.

| Pass | Path | Median TTFT | Median request | Output tok/s |
| --- | --- | ---: | ---: | ---: |
| 1 | Baseline | 28.946 ms | 128.038 ms | 249.961 |
| 1 | Guarded direct prefill | 28.177 ms | 125.030 ms | 255.917 |
| 2 | Guarded direct prefill | 28.386 ms | 125.357 ms | 253.359 |
| 2 | Baseline | 30.374 ms | 131.034 ms | 244.215 |

The process-to-process spread means the exact gain is not a fixed constant;
these pairs support roughly 0.8–2.0 ms lower TTFT and 2–4% higher output
throughput, not anything near PyTorch parity. A separate 128+64 default-path
run produced identical token IDs to both the earlier opt-in Jittor run and
the version-matched native PyTorch run across all three measured requests.
Its median request was 234.900 ms (272.375 tok/s); this was not a same-time
PyTorch A/B and must not be used to claim a new cross-framework ratio.

A fresh native PyTorch run immediately after the second Jittor candidate used
the same GPU, prompt, model, Transformers 5.5.3, script and 2+7 protocol. Its
median TTFT/ITL/request were 11.716/2.326/84.243 ms and throughput was
379.026 tok/s, versus that Jittor candidate's 28.386/3.113/125.357 ms and
253.359 tok/s. All 32 output tokens matched in all requests. Jittor reached
66.8% of native PyTorch output throughput in this matched configuration;
the remaining gap is ~16.67 ms TTFT and ~0.79 ms/token Graph decode. The
native report is `prefill-auto-oracle-new.json` in the same state directory.

An additional idea, skipping the host copy of CUDA slot mapping, was tested
with the same model and a one-output-token request. Its 7-request TTFT median
was 28.161 ms with the old copy and 28.063 ms without it, below reliable
resolution of this experiment. The proposed code was removed, leaving no
extra shape-dependent cache-write branch in the adapter.

Unit coverage now checks both the direct full-prefill GQA math/shape guard
and the plain-decoder feature guard. All 10 attention adapter tests pass with
the Jittor torch shim; `git diff --check` passes. Raw reports are
`prefill-auto-{baseline-new,candidate-new,candidate-new-r2,baseline-new-r2}.json`,
`prefill-auto-default-64tok.json`, and `slot-host-{baseline,candidate}-1tok.json`
in the same state directory. The current bottleneck remains the ~17 ms
first-token gap and ~0.8 ms/token Graph-decode gap versus matched PyTorch;
the guarded prefill route addresses only a small fraction of it.
