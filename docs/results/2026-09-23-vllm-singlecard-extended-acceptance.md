# Single-card vLLM: unified regression, quality, lifecycle and HTTP

- Status: the stated eager FP16 cases are accepted; broader boundaries below
  remain open. This is not a general production-readiness declaration.
- Owner: Codex; reviewed 2026-09-23.
- Production baseline: `9853c6a` (parent `6658645`). The GPU checkout retains
  its existing worktree; 1212 production file hashes plus the three matrix
  runners were checked against the local source, rather than treating its
  older Git HEAD as the tested source identity.
- Runtime: RTX 4090, vLLM 0.24.0, transformers 5.5.3, native PyTorch
  2.11.0+cu130, independent Jittor shim environment, TP=1, FP16, eager,
  FLASH_ATTN. Jittor preserves its CUDA factory default; PyTorch preserves
  its CPU factory default. Actual model computation is CUDA in both.
- Review when: source, model weights, dependency versions, precision,
  attention backend, scheduling or execution mode changes.

Raw artifacts are unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-singlecard/20260923/final-acceptance/`.
The previous [correctness report](2026-09-23-vllm-singlecard-correctness.md)
describes earlier kernel, generator, host-metadata and penalty repairs.

## One production version, two model matrices

`matrix-r3/` contains fresh runs under both independent environments after
the shape-hint repair. Both models complete state/cache, seeded random
generation with five returned logprobs, batch, penalties and long-input cases.

| Per model and backend | Qwen3-0.6B | OPT-125m |
| --- | ---: | ---: |
| Requests / output tokens | 70 / 1492 | 70 / 1492 |
| Greedy requests / tokens exactly equal across backends | 58 / 1232 | 58 / 1232 |
| Seeded random requests exactly equal across backends | 7 / 12 | 4 / 12 |
| Repeated seeds reproducible within each backend | yes | yes |
| Cache cold / repeat / reset hit tokens | 0 / 624 / 0 | 0 / 624 / 0 |

Qwen long inputs contain 1024 and 3072 tokens. OPT uses 512 and 1536,
with its supported 2048-position engine limit. Each long request generates
128 tokens. The matrix comparator checks options, prompts, lengths and all
greedy tokens; it reports stochastic differences explicitly. These counts
are test outcomes, not a model accuracy percentage.

## A newly covered OPT logprob interface

OPT random generation with returned logprobs, and independently OPT prompt
logprobs for quality evaluation, exposed a missing
`torch._dynamo.decorators.mark_unbacked`. The legacy sampler calls this
before counting the sampled token's rank; earlier plain generation did not
exercise this path.

`compat/tests/torch/test_dynamo_shape_hints.py` failed **5 CPU cases** and
**8 combined CPU/CUDA cases** before implementation. The independent
PyTorch reference passes seven cases, with one intentionally shim-only check
skipped. The compatibility repair registers the owned decorators module and
preserves default eager dimension annotations without changing tensor values,
shape or placement. Symbolic specialization options fail explicitly; fidelity
is approximate, and no Dynamo compilation capability is claimed.

After the repair, CPU checks pass 5/5; the GPU run passes all eight new cases
plus ten existing FX-pickler boundary cases (**18 passed**). Real OPT random
logprobs and full prompt-logprob evaluation then pass. The final two-model
matrix was rerun after the repair. Red and green evidence is retained under
`mark-unbacked-*.log`, the initial `opt/jittor-random.*`, and
`quality/opt-jittor-smoke-r2.*` / `*-r3.*`.

## Full WikiText-2 raw test quality comparison

The fixed Salesforce/wikitext test split revision is
`b08601e04326c79dfdd32d625aee71d232d685c3`; its parquet SHA256 is
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
The protocol joins all 4358 text rows with two newlines, disables added special
tokens, and uses 1024-token windows with stride 512. Each target is scored once;
the first corpus token is excluded. `prompt_logprobs=0` returns the actual
target-token probability. Prefix caching is disabled.

The comparator verifies model-weight hashes, tokenizer/config hashes, dataset,
token sequence, windows, precision, runner SHA and dependency versions. Both
runtimes assert CUDA parameter, KV-cache and forward-output placement.
Engineering limits were fixed before GPU evaluation: absolute mean NLL drift
at most 0.01 nats/token and maximum token NLL drift at most 0.5 nats.

| Model | Scored tokens | PyTorch PPL | Jittor PPL | Mean NLL difference | Maximum token NLL difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 299077 | 19.88605095 | 19.88608862 | +0.000001894 | 0.174006 |
| OPT-125m | 287643 | 26.10361741 | 26.10320455 | -0.000015816 | 0.054906 |

Both pass. Tokenizers differ between models, so compare each model across
backends, not the two models' PPL directly. This is language-model NLL/PPL for
the declared protocol, not QA accuracy, universal numerical equivalence or a
proof of identical stochastic generation distributions. Instrumented quality
evaluation time is not a throughput benchmark.

Evidence: `quality/*-oracle-full-r2.*`, `*-jittor-full-r3.*`, and
`*-full-comparison-r3.json`. A metadata-only harness failure from reading
the shim's absent `torch.__file__` was retained and corrected; both oracle
full evaluations were repeated under the same corrected runner SHA.

## Continuous requests, cancellation and KV pressure

Each backend completes 328 requests in the continuous scenario (eight isolated
references plus 320 queued mixed-length requests), 16 completed requests in
the cancellation scenario, and eight in the preemption scenario.

| Model | Continuous requests vs isolated outputs | Cancel and recover | Preempt and recompute |
| --- | --- | --- | --- |
| Qwen | Strict assertion remains failed: Jittor 2/320 differ; PyTorch 13/320 differ | pass; all 768 tokens equal across backends | pass; all 1024 tokens equal across backends |
| OPT | pass; all 4264 tokens equal across backends | pass; all 768 equal | pass; all 1024 equal |

Cancellation covers an active request and a waiting request: no later output,
no retained scheduler request, unchanged surviving requests, and successful
new requests. KV pressure uses actual constrained blocks. Both models/backends
observe a request preempted after 48 output tokens, computed position reset
from 240 to 0, followed by rescheduling and recomputation. Pressure outputs
equal isolated references. The harness was corrected after a retained failure
to recognize recovery through either a subsequent new-request entry (v2) or
cached-resumed entry, with actual scheduled tokens and ordering evidence.

Qwen's continuous strict assertion was not weakened. A fresh PyTorch engine,
no previous user requests and prefix caching disabled reproduce the different
token for mixed batches. At output index seven, token 576 leads 220 by
0.015625 in the isolated run; in the mixed run their FP16 logits tie and
argmax selects the smaller token ID. Observational hooks preserve all outputs,
and compared requests have identical token prefixes at the divergence.
Fresh Jittor first-batch execution likewise reproduces its two differing
outputs without prior requests or prefix-cache reuse. These are evidence for
batch-dependent numerical paths; historical state contamination is not needed
to produce these examples. The first differing model operator is still not
localized, and all possible state bugs are not excluded.

All bounded runs drain their requests and KV usage. Memory observations are
retained but are not an indefinite leak-free guarantee. Evidence includes
`stability/paired-summary.json`, `qwen-native-turnover-diagnosis.json`,
`qwen-jittor-turnover-r2-diff.json` and fresh-engine controls. Multi-hour soak
testing and arbitrary scheduling/length coverage remain open.

## Real HTTP serving

Both isolated runtimes launch the actual AsyncLLM/multiprocess OpenAI-compatible
server, bound only to localhost. `http_acceptance.py` checks four initial
requests, twenty serial requests, and twenty requests at concurrency four:
**44 requests and 1408 output tokens per backend**. HTTP 200, SSE completion,
finish reason, output token IDs and usage counts all pass. Every request's
token IDs and text match across backends, including concurrent requests.

A long ZMQ IPC path prevented the first Jittor server launch. A separate bind
probe reproduces long-path failure and short-path success; a short
`VLLM_RPC_BASE_PATH` resolves this environment limit without production changes.
Cold JIT exits/logs are retained separately. Servers and their child processes
were stopped after validation. See `http/comparison.json`, both client JSONs,
server logs and external launch scripts.

This is a bounded service/concurrency check, not production load certification
or HTTP disconnect-cancellation coverage. SSE chunks can contain multiple
tokens, so inter-chunk delay is not reported as individual-token ITL. The
first concurrent Jittor wave includes compilation; its aggregate speed is not
a valid warm-performance comparison.

## Warm performance

Six fresh processes run sequentially on the same physical GPU: three per
backend, each with three warmups and 21 measured generations for batch sizes
one and four. Every prompt contains 128 input tokens, with 32 generated tokens;
prefix caching is disabled. Initialization, warmups, diagnostic hooks and JSON
writes are outside timed generations. Both backends use identical options,
prompt token IDs, script SHA and the same GPU UUID. All **315 measured requests
and 10080 generated tokens per backend** match exactly.

Throughput below is the median of three process-level aggregate token rates.
TTFT/ITL are pooled medians of observed offline `engine.step` token delivery;
they include host/engine work and are not isolated GPU kernel timing or HTTP
latency. Each step was checked for at most one new token per request.

| Batch | Backend | Output token/s | TTFT ms | ITL ms |
| --- | --- | ---: | ---: | ---: |
| 1 | PyTorch | 58.65 | 33.60 | 16.49 |
| 1 | Jittor | 19.50 | 137.35 | 49.03 |
| 4 | PyTorch | 210.10 | 37.75 | 18.85 |
| 4 | Jittor | 74.67 | 144.42 | 52.50 |

Jittor throughput is **33.2% / 35.5%** of the native reference at batch one /
four. Process rate ranges are 57.41–61.13 versus 19.19–20.31 token/s at batch
one, and 209.00–218.36 versus 69.72–74.72 at batch four. The performance gap
remains open. This protocol has longer inputs than the old short-prompt
benchmark, so its ratios are a new baseline rather than a measured speedup
from those historical numbers.

The first selected GPU had another user's process and that attempt was
excluded. `latency-r2/` contains the accepted six runs, before/after GPU
snapshots, logs and `comparison.json`; no other process was observed on the
chosen GPU during these runs. Other users still occupied other GPUs on the
shared host. Profiling is run separately after all timed processes finish;
its instrumented durations must not be substituted for these measurements.

## Profiling and optimization priorities

After the benchmark, separate cProfile captures use three warmups and three
profiled generations per batch. Tokens remain equal to the warm references.
Batch one captures 12.60 million function calls; batch four captures 13.11
million. Precision-policy lookup runs about one million times, dtype lookup
about 255000 times, installation-context lookup about 343000–351000 times,
and kernel selection about 81500 times per capture. Both captures contain
1911 `sync_all` calls, 1545 adapter UVA snapshots and 9210 Triton device-copy
calls. These are measured call counts, not estimates of avoidable work.

The high-frequency Python frontend and dispatch paths are optimization
candidates. cProfile disproportionately affects frequent calls, and nested
cumulative times must not be added. Native synchronization time includes
execution/waiting; a CUDA timeline is needed to distinguish device computation,
copies, synchronization and CPU submission gaps. Observed NumPy conversions
come from attention lengths, input metadata and output readback, and do not
establish that model math ran on CPU. The captured bridge `_compile` calls
perform cache lookup; no upstream Triton compilation or Python compiler launch
was observed, but C++ internal compilation is not independently excluded.

No performance repair is claimed. Any reduction of repeated dtype/dispatch
work must preserve installation ownership and mutable precision policies.
Changes to UVA snapshots or synchronization require targeted update/aliasing
regressions before real model and uninstrumented benchmark reruns, because
those copies previously repaired stale sampling state. Evidence is under
`profile/jittor-warm-r1/`, including pstats, caller counts and `findings.json`.

## Advanced modes and remaining coverage

Latest `9853c6a` real-engine probes still fail without fallback:

- FP8: importing `torch.library.wrap_triton` fails before the FP8 kernel path.
- CUDA Graph: the platform graph-pool call resolves to a missing/non-callable
  `torch.cuda.graph_pool_handle`.
- Compilation: `torch.fx._lazy_graph_module._use_lazy_graph_module` is missing.

These are first blockers, not evidence that adding one symbol will implement
the mode. `advanced/advanced-summary.json` retains explicit options and full
exceptions. AWQ/GPTQ, broader models/dtypes/contexts, multi-GPU and sustained
service operation remain outside this accepted scope. The user-deferred CPU
default experiment remains unchanged.

## Repository checks and reproduction

Torch CPU core plus new shape-hint/FX boundaries: **227 passed, 6 skipped**.
The full structure gate reports **1364 passed, 4 failed, 8 skipped**; failures
are the unchanged baseline interpreter-launch/collection-policy violations
in `test_executor_python_threads.py` and `test_h3_decode_thread_race.py`.
Layout, generated manifests and whitespace checks pass. The full repository
is not declared green.

Use the commands in the
[vLLM runbook](../../agent/skills/vllm-torch-compat/SKILL.md), with separate
environments, caches and result paths. Per-model matrix comparisons use
`compare_singlecard_acceptance.py RESULT_DIR --cases state random batch
penalties long`. Quality and lifecycle cases must be independently compared
with the binary PyTorch oracle. Preserve failed artifacts when rerunning.
