# vLLM multi-GPU TP retest and CUDA Q/K norm + RoPE fusion

Date: 2026-10-02

## Scope

- Checkout: `d7436b0c863dc9f2ceba381fbad67129596bec0c` plus the existing dirty worktree.
- Model: `/home/chk/models/Qwen3-0.6B`, FP16, eager vLLM 0.24.0.
- Physical GPUs: RTX 4090 3,4 for TP2 and 3,4,5,6 for TP4. GPUs 1,2 were occupied and excluded.
- Protocol: input 128 tokens, output 32 tokens, greedy decode, batches 1 and 4, 3 warmups and 21 measured rounds.
- PyTorch oracle used native torch 2.11.0+cu130 with no Jittor shim. Jittor workers used the torch shim and the same physical GPUs.

## Formal results

Output tokens per second (higher is better):

| TP | Batch | PyTorch | Jittor before | Jittor final | Final vs before | Final / PyTorch |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 78.220 | 33.396 | 35.107 | +5.12% | 44.9% |
| 2 | 4 | 307.574 | 125.054 | 128.477 | +2.74% | 41.8% |
| 4 | 1 | 78.010 | 28.037 | 29.117 | +3.85% | 37.3% |
| 4 | 4 | 307.145 | 100.577 | 100.857 | +0.28% | 32.8% |

TP2 median ITL improved from 28.989 to 27.381 ms at batch 1 and from 29.678 to 29.188 ms at batch 4. TP4 batch 1 improved from 33.030 to 31.795 ms. TP4 batch 4 is intentionally left on the split kernel path after two fused-batch runs regressed to 97.519 and 96.504 tok/s.

The Qwen3-0.6B model does not scale from TP2 to TP4 on these PCIe 4090s: the extra collective and process overhead is larger than the compute saved by sharding this small model. This is a performance result, not a correctness or placement failure.

## Change

- Added an inference-only CUDA kernel that combines per-head Q/K RMSNorm and NeoX RoPE in one launch, avoiding an intermediate Q/K write/read and one launch per decoder layer.
- Routed the public serving primitive through CUDA after the existing ACL hook.
- Enabled batched CUDA fusion for TP1/TP2. TP3+ remains fused for the single-token path but uses the split path for batches after repeatable TP4 batch-4 regressions.
- Added a CUDA GQA numerical test covering unequal Q/K head counts and nonconsecutive positions.

The synchronized kernel diagnostic measured roughly 0.020 ms fused versus 0.27-0.28 ms split for token counts 1, 4 and 128. End-to-end gains are smaller because attention, GEMMs, sampling and collectives remain.

## Validation

- `tests.nn.test_serving_ops` and `adapters.tests.vllm.test_layers`: 18 tests passed, including the CUDA FP16 GQA test against a NumPy reference.
- PyTorch TP2, PyTorch TP4, Jittor TP2 and Jittor TP4 all reported the expected rank/device placement and clean worker shutdown.
- Greedy token IDs matched exactly between PyTorch and final Jittor for TP2/TP4 and batches 1/4.
- `git diff --check` passed.

## Evidence

- Baselines: `/home/chk/jittor-lab/_state/vllm-tp/20261002/baseline/`
- Final TP2 batch 1: `/home/chk/jittor-lab/_state/vllm-tp/20261002/fused/jittor-tp2-final-b1.json`
- Final TP2 batch 4: `/home/chk/jittor-lab/_state/vllm-tp/20261002/fused/jittor-tp2-final-b4.json`
- Final TP4 batch 1: `/home/chk/jittor-lab/_state/vllm-tp/20261002/fused/jittor-tp4-final-b1.json`
- Final TP4 batch 4: `/home/chk/jittor-lab/_state/vllm-tp/20261002/fused/jittor-tp4-final-b4.json`
- Phase profile: `/home/chk/jittor-lab/_state/vllm-tp/20261002/profile/current-jittor-tp2.json`
- Kernel diagnostic: `/home/chk/jittor-lab/_state/vllm-tp/20261002/qk_fusion_bench.py`

## Follow-up: collective holder lifetime and direct attention output

The remaining TP2 decode gap was profiled with the same model and GPUs. Both
backends execute 57 all-reduces per decode token. The synchronized median was
about 0.081-0.089 ms in Jittor and 0.041 ms in PyTorch, so communication
explains only about 2.3 ms of the roughly 13.6 ms model-forward gap. The other
large fixed costs are the per-layer paged-attention/cache path and sampling.

Two production changes were retained:

- For TP1/TP2, native NCCL all-reduce now publishes its result with the public
  `out_hint()` lifetime boundary instead of allocating a clone and then
  rebinding it with `update()`. The communicator caches this decision. TP3+
  keeps the old holder publication because a TP4 batch-4 trial regressed.
- The CUDA paged-decode kernel accepts an existing output tensor and writes
  directly into vLLM's reusable attention output storage. Prefill, partial
  output shapes, and unsupported backends retain the old result-and-copy path.
  This avoids one setitem/copy kernel after every decoder layer without
  replacing the vLLM-owned holder.

The revised TP2 formal result (3 warmups, 21 measurements) is:

| Batch | PyTorch | Previous Jittor final | All-reduce result | Direct attention final | Total gain | Final / PyTorch |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 78.220 | 35.107 | 43.054 | 43.625 | +24.26% | 55.77% |
| 4 | 307.574 | 128.477 | 164.528 | 169.388 | +31.84% | 55.07% |

Median TP2 ITL in the final run was 22.160 ms for batch 1 and 22.644 ms for
batch 4. The final attention step added about 1.3% and 3.0% throughput over the
already optimized all-reduce formal result. Greedy output tokens remained
identical and shutdown removed every worker and shared-memory object.

TP4 was rerun on physical GPUs 0,3,4,6 because GPU 5 became occupied by another
user. Its 21-round direct-output result was 27.678/99.302 tok/s for batches 1/4.
Because this is a different PCIe placement it is not directly comparable with
the earlier 3,4,5,6 formal result. A matched seven-round A/B on 0,3,4,6 measured
29.852/99.129 tok/s with direct output and 29.077/100.038 tok/s with the copy:
+2.66% at batch 1 and -0.91% at batch 4, effectively neutral at TP4. Tokens,
rank placement, and lifecycle checks passed in both runs.

Rejected experiments are important here. Skipping the greedy FP32 sampling
copy reduced throughput by about 2.2-2.4%, so it was reverted. Rebinding the
attention holder with `output.update(attended)` looked 2-6% faster in one short
run, but a repeated batch-4 run killed a worker around decode token 20. It was
also reverted. The retained direct-output kernel preserves the holder identity
and passed a CUDA FP16/BF16 numerical and object-identity test plus 10 adapter
tests. The six public paged-attention fallback tests also passed after the
oneDNN test dependency was built.

Additional evidence:

- All-reduce formal TP2: `/home/chk/jittor-lab/_state/vllm-tp/20261002/final-cached-formal-tp2/result.json`
- All-reduce formal TP4: `/home/chk/jittor-lab/_state/vllm-tp/20261002/final-cached-formal-tp4/result.json`
- Direct-output formal TP2: `/home/chk/jittor-lab/_state/vllm-tp/20261002/direct-out-formal-tp2/result.json`
- Direct-output formal TP4 (GPUs 0,3,4,6): `/home/chk/jittor-lab/_state/vllm-tp/20261002/direct-out-formal-tp4-altgpus/result.json`
- Matched TP4 copy/direct probes: `/home/chk/jittor-lab/_state/vllm-tp/20261002/output-slice-probe-tp4-altgpus/result.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261002/direct-out-probe-tp4-altgpus/result.json`
- Final TP2 collective/attention profile: `/home/chk/jittor-lab/_state/vllm-tp/20261002/final-profile-tp2/result.json`

## Follow-up: lazy-graph synchronization interval

On 2026-10-03, physical GPUs 3 and 4 were occupied by another user, so the
next matched A/B and oracle comparison used otherwise-idle GPUs 0 and 2. These
numbers must not be mixed with the earlier 3,4 results when calculating a
change. The retained change raises the attention adapter's lazy-graph boundary
from 16 to 32 layers. Sixteen forced nearly two device synchronizations per
token for this 28-layer model; 32 still caps graph growth while normally
requiring only one boundary per decode step.

A matched seven-round A/B on GPUs 0,2 measured:

| Batch | 16-layer boundary | 32-layer boundary | Change |
|---:|---:|---:|---:|
| 1 | 43.659 tok/s | 45.430 tok/s | +4.06% |
| 4 | 168.308 tok/s | 170.655 tok/s | +1.39% |

The retained default then completed 3 warmups and 21 measured rounds. A native
PyTorch oracle was run immediately afterward on the same physical GPUs and
protocol:

| Batch | PyTorch | Jittor | Jittor / PyTorch | PyTorch/Jittor gap |
|---:|---:|---:|---:|---:|
| 1 | 75.364 tok/s | 44.821 tok/s | 59.47% | 1.68x |
| 4 | 296.650 tok/s | 171.602 tok/s | 57.85% | 1.73x |

Median Jittor ITL was 21.532 ms at batch 1 and 22.500 ms at batch 4, versus
13.264 and 13.393 ms in PyTorch. Greedy output tokens matched exactly for both
batches. Both backends reported the expected TP rank placement and passed
worker/shared-memory cleanup. Final regression covered nine sampling/attention
adapter tests plus the CUDA decode test and six public paged-attention tests;
all 16 passed. A separate 128-output-token stress run completed every request
at both batch sizes and also passed process/shared-memory cleanup; it measured
44.113 and 169.973 tok/s for batches 1 and 4.

Two additional micro-optimizations were rejected after real-model A/B tests:

- A custom CUDA greedy argmax measured about 0.0135 ms versus 0.0444 ms for
  the existing cast-plus-argmax microbenchmark, but reduced end-to-end TP2
  throughput by 40-45%. Publishing the result with `out_hint()` did not fix
  it. Consuming logits through a new custom op disturbed Jittor's graph and
  tensor-lifetime behavior, so the experiment and its tests were removed.
- Scheduling KV-cache scatter and paged decode as two CUDA launches in one
  `jt.code` op passed FP16/BF16 numerical and holder-identity tests, but the
  matched real-model run was slower: 43.056 versus 43.659 tok/s at batch 1 and
  163.287 versus 168.308 at batch 4. It was fully reverted. The multiple-output
  cache/output lifetime and scheduling cost outweighed removal of one graph
  node; separate ordered cache and attention ops remain the faster path.

Evidence:

- 16-layer matched probe: `/home/chk/jittor-lab/_state/vllm-tp/20261003/split-kv-probe-tp2-gpu02/result.json`
- 32-layer matched probe: `/home/chk/jittor-lab/_state/vllm-tp/20261003/sync32-probe-tp2-gpu02/result.json`
- Final 32-layer TP2: `/home/chk/jittor-lab/_state/vllm-tp/20261003/sync32-formal2-tp2-gpu02/result.json`
- Same-GPU PyTorch oracle: `/home/chk/jittor-lab/_state/vllm-tp/20261003/oracle-formal-tp2-gpu02/result.json`
- 128-token stress run: `/home/chk/jittor-lab/_state/vllm-tp/20261003/sync32-stress-tp2-gpu02/result.json`
- Rejected fused KV/decode probe: `/home/chk/jittor-lab/_state/vllm-tp/20261003/fused-kv-probe-tp2-gpu02/result.json`
- Rejected custom sampling probes: `/home/chk/jittor-lab/_state/vllm-tp/20261003/fused-greedy-probe-tp2/result.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261003/fused-greedy-hint-probe-tp2/result.json`

## Follow-up: static inference dispatch and parameter/cache lifetime

Later on 2026-10-03, GPUs 3 and 4 became idle and were used for another
same-GPU TP2 comparison. GPUs 1,2,5,6 remained occupied by other users, so a
safe TP4 rerun was not possible. The protocol remained Qwen3-0.6B, input 128,
output 32, greedy decoding, batches 1/4, three warmups and 21 measured rounds.

An instrumented decode run attributed about 26.2 ms/token to model forward,
1.6 ms to sampling, and 1.35 ms to input/attention-metadata preparation. Both
ranks still issued 57 all-reduces per token at about 0.055-0.063 ms median;
communication alone did not explain the forward gap. The same profile observed
1,792 RoPE cache matches, all identity hits.

Two production changes were retained:

- `UnquantizedLinearMethod.apply` validates each loaded CUDA/no-grad vLLM layer
  once and then calls the already-selected cuBLAS relay directly. This removes
  repeated shape, placement, dtype and backend-capability dispatch from every
  projection. Biased or unsupported calls retain vLLM's original path, and
  `JITTOR_VLLM_DIRECT_CUBLAS=0` provides an exact benchmark fallback.
- Qwen3 attention now caches its loaded Q/K norm parameters, TP geometry,
  fused-kernel capability and the already-matched RoPE cache per layer. CUDA
  and no-grad remain dynamic guards. `JITTOR_VLLM_QK_STATIC_CACHE=0` disables
  this cache for matched comparisons.

The first matched seven-round A/B (direct cuBLAS only) measured:

| Batch | Generic Linear path | Direct cuBLAS | Change |
|---:|---:|---:|---:|
| 1 | 43.900 tok/s | 47.857 tok/s | +9.01% |
| 4 | 167.438 tok/s | 176.762 tok/s | +5.57% |

With direct cuBLAS enabled on both sides, the Q/K static-cache A/B measured:

| Batch | Cache off | Cache on | Change |
|---:|---:|---:|---:|
| 1 | 46.695 tok/s | 48.549 tok/s | +3.97% |
| 4 | 174.916 tok/s | 184.794 tok/s | +5.65% |

The final 21-round result and the native PyTorch oracle, run immediately on
the same physical GPUs, were:

| Batch | PyTorch | Jittor final | Jittor / PyTorch | PyTorch/Jittor gap |
|---:|---:|---:|---:|---:|
| 1 | 76.734 tok/s | 48.958 tok/s | 63.80% | 1.57x |
| 4 | 305.056 tok/s | 185.854 tok/s | 60.92% | 1.64x |

Median final Jittor ITL was 19.717 ms at batch 1 and 20.776 ms at batch 4,
versus PyTorch's 12.991 and 13.070 ms. Relative to the matched generic-path
probe at the start of this follow-up, the combined final result gained 11.52%
and 11.00%. Greedy output token IDs matched PyTorch exactly. All workers and
shared-memory objects were cleaned up after every recorded run.

A final instrumented run confirmed that the change reduced median model
forward from about 26.2 ms to 24.0-24.2 ms per token. Input/attention metadata
preparation stayed near 1.36 ms and sampling near 1.55 ms; the RoPE trace fell
from 1,792 redundant identity matches to zero. The remaining same-GPU PyTorch
profile is still about 14.8-15.1 ms in model forward. No further sampling or
combined KV/attention rewrite was retained because the earlier real-model A/B
tests showed those graph-boundary changes regress or destabilize throughput.

Regression validation covered 55 vLLM adapter tests and nine native
Q/K-RMSNorm-RoPE, KV-cache and public paged-attention tests; all 64 passed.
`git diff --check` also passed.

Evidence:

- Generic/direct cuBLAS A/B: `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/baseline.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/direct.json`
- Direct-cuBLAS 21-round run: `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/direct-formal21.json`
- Q/K cache A/B: `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/qk-cache-off7.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/qk-cache-on7.json`
- Final combined 21-round run: `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/qk-cache-formal21.json`
- Same-GPU PyTorch oracle: `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/oracle-formal21.json`
- Instrumented hotspot run: `/home/chk/jittor-lab/_state/vllm-tp/20261003/sync32-profile-tp2-gpu34/result.json`
- Final instrumented run: `/home/chk/jittor-lab/_state/vllm-tp/20261003/direct-cublas-ab/final-profile.json`

## Follow-up: attention graph views and TP4 collective publication

On 2026-10-03/04, a first-layer comparison localized the largest remaining
per-layer gap to the complete attention wrapper: about 0.395 ms in Jittor
versus 0.150 ms in PyTorch.  The two synchronized CUDA operations inside that
wrapper accounted for only about 0.070 ms for KV scatter and 0.065 ms for
paged decode.  The residual cost was therefore dominated by graph/view and
publication work around the kernels rather than by the paged-decode kernel
alone.

Two additional production changes were retained:

- The vLLM attention adapter no longer creates `query[:tokens]`,
  `key[:tokens]`, and `value[:tokens]` graph views when the tensors already
  contain exactly `tokens` rows.  Partial/padded inputs retain the slices, and
  `JITTOR_VLLM_REUSE_FULL_TOKEN_VIEWS=0` restores the previous behavior.
- TP3+ all-reduce publication now allocates an empty holder before updating it
  with Jittor's out-of-place NCCL result.  The previous holder cloned the
  input even though all of its contents were immediately overwritten.  TP1/2
  still return the NCCL result directly.  Set
  `JITTOR_VLLM_EMPTY_ALL_REDUCE_HOLDER=0` to restore the TP3+ clone.

The full-token-view change had repeatable matched seven-round gains:

| TP | Batch | Views off | Views on | Change |
|---:|---:|---:|---:|---:|
| 2 | 1 | 47.056 tok/s | 51.990 tok/s | +10.49% |
| 2 | 4 | 177.253 tok/s | 196.178 tok/s | +10.68% |
| 4 | 1 | 46.622 tok/s | 49.211 tok/s | +5.55% |
| 4 | 4 | 158.745 tok/s | 167.036 tok/s | +5.22% |

The TP4 empty-holder A/B then measured 49.211 to 49.806 tok/s at batch 1
(+1.21%) and 167.036 to 168.563 tok/s at batch 4 (+0.91%).  A 21-round run
of the retained path produced 49.671 and 168.065 tok/s.  A matching 21-round
legacy-holder rerun could not be completed because all seven GPUs became
occupied by external jobs on 2026-10-04; the failed attempt stopped during
worker initialization and produced no measurement samples.

The latest formal/same-GPU comparisons are:

| TP | Batch | PyTorch | Jittor | Jittor / PyTorch | PyTorch/Jittor gap |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 76.734 tok/s | 51.738 tok/s | 67.42% | 1.48x |
| 2 | 4 | 305.056 tok/s | 196.041 tok/s | 64.26% | 1.56x |
| 4 | 1 | 77.198 tok/s | 49.671 tok/s | 64.34% | 1.55x |
| 4 | 4 | 304.042 tok/s | 168.065 tok/s | 55.28% | 1.81x |

TP2 formal ITL is 18.619/19.604 ms for batches 1/4.  TP4 formal ITL is
19.216/22.938 ms.  All recorded Jittor/PyTorch greedy token IDs matched
exactly, and every completed engine run passed worker/shared-memory lifecycle
checks.  TP4 remains slower than TP2 on this 0.6B model because four RTX 4090s
have no hardware CUDA P2P path and the extra collectives outweigh the smaller
per-rank GEMMs, especially at batch 4.

Two experiments were rejected and removed:

- Calling the CUDA paged-decode entry point directly regressed TP2 by 2.70%
  and 2.53% for batches 1 and 4.
- Skipping the FP32 processed-logits copy for unmodified greedy sampling kept
  token IDs exact but regressed TP2 by 1.20% and 0.93%; BF16 argmax was slower
  in the real graph.

The new collective/greedy focused tests pass (6/6), the full-token view adapter
tests passed on CUDA before the GPUs were claimed (7/7), and `git diff --check`
passes.  A later CPU-only attempt ran 63 adapter tests: 45 passed and 18 CUDA
or oneDNN-dependent cases could not run without a visible GPU/cmake; these were
environment failures rather than assertion mismatches.  The prior nine native
CUDA serving-kernel tests remain valid because this follow-up did not modify
their kernels.

Evidence:

- Attention profiles: `/home/chk/jittor-lab/_state/vllm-tp/20261003/detail-profile-jittor.json`, `/home/chk/jittor-lab/_state/vllm-tp/20261003/detail-profile-oracle.json`, and `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-subphase-jittor.json`
- TP2 view A/B and formal run: `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/full-views-off7.json`, `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/full-views-on7.json`, and `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/full-views-formal21.json`
- TP4 view A/B: `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/tp4-full-views-off7.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/tp4-full-views-on7-rerun.json`
- TP4 holder probe/formal: `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/tp4-empty-holder7.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/tp4-empty-holder-formal21.json`
- Same-GPU TP4 oracle: `/home/chk/jittor-lab/_state/vllm-tp/20261003/attention-wrapper-ab/tp4-oracle7.json`
- Rejected sampling A/B: `/home/chk/jittor-lab/_state/vllm-tp/20261003/sampling-raw-ab/off7.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261003/sampling-raw-ab/on7.json`

## Follow-up: single-GPU attribution and rejected scheduler/kernel changes

On 2026-10-04, shared GPU 0 initially reported 0% utilization (an unrelated
process held about 0.9 GiB) and was used to separate the remaining framework
overhead from NCCL/P2P effects.  The protocol was again Qwen3-0.6B FP16 eager,
input 128, output 32, greedy decoding, two warmups and seven measured rounds.
The narrow spread of that run makes it useful diagnostic evidence, but it is
not an exclusive-GPU formal result.

| Batch | PyTorch | Jittor | Jittor / PyTorch | PyTorch/Jittor gap |
|---:|---:|---:|---:|---:|
| 1 | 102.718 tok/s | 58.070 tok/s | 56.53% | 1.77x |
| 4 | 399.583 tok/s | 217.853 tok/s | 54.52% | 1.83x |

Median Jittor ITL was 16.508/17.531 ms for batches 1/4, versus PyTorch's
9.704/9.975 ms.  Output token IDs matched exactly and lifecycle cleanup
passed.  This confirms that the largest remaining deficit is present without
tensor-parallel communication.

A synchronized all-layer decomposition corrected an earlier first-layer-only
extrapolation.  Per decode step, Jittor/PyTorch time attributed to all 28
layers was approximately 14.95/7.39 ms in self-attention, 3.45/2.47 ms in the
MLPs, 1.88/0.97 ms in input RMSNorm, and 1.99/1.18 ms in post-attention
RMSNorm.  Input preparation, attention metadata preparation, and sampling
also remained slower at median 1.00/0.26, 0.41/0.06, and 1.61/0.63 ms.  These
timings synchronize around each module and are attribution measurements, not
representative throughput timings.

Three new changes were implemented and functionally exercised against the
real model, but none was retained:

- Direct Jittor CUDA replacements for `prepare_pos_seq_lens` and the ordinary
  non-speculative `combine_sampled_and_draft_tokens` path kept token results
  correct and measured 57.90 tok/s at batch 1, showing no clear improvement
  over the earlier 58.07 baseline.
- Moving the rolling 32-layer graph synchronization to the end of Qwen's
  model forward completed at both batch sizes; a conditional form also passed
  generation and cleanup checks.
- Reducing hidden-state RMSNorm from 1024 to 256 threads per row also completed
  at both batch sizes with correct generation.

At 00:28, before the latter comparisons, another user's benchmark started on
GPU 0.  It later exposed two child CUDA processes using about 16.9 and 3.7 GiB
and up to 86% SM.  GPUs 1-6 were simultaneously occupied by training jobs.
Consequently the later 55.81/206.23 and 56.91/207.38 measurements, as well as
a 55.93/207.67 restored-source rerun, are contention-tainted and must not be
used as code A/B results.  The candidate changes were removed because no
reliable positive result was established, not because those contaminated
numbers prove a regression.

No production code from these three experiments was retained.  The worktree
was restored to the preceding full-token-view/collective-holder state and
`git diff --check` passes.

Evidence:

- Single-GPU Jittor/PyTorch: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/jittor-current7.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/oracle-current7.json`
- Single-GPU detailed traces: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/jittor-detail.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/oracle-detail.json`
- All-layer traces: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/jittor-coarse.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/oracle-coarse.json`
- Rejected input-kernel smoke run: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/input-kernels-smoke2.json`
- Rejected conditional model-end sync: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/model-end-sync-conditional7.json`
- Rejected 256-thread RMSNorm: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-current/rms256-7.json`

## Follow-up: precompiled-op hypothesis and fused cache/decode experiment

The claim that repeated Python-side `jt.code` construction was the dominant
remaining cost was tested directly rather than assumed.  For RMSNorm, 5,000
lazy-op constructions were timed in seven rounds without executing the CUDA
kernel.  The existing cached-`jt.code` path took a median 8.360 us per call;
a native op compiled once with `compile_custom_op` took 7.318 us, a saving of
only 1.042 us.  Even multiplying that by all 56 hidden-state RMSNorm calls in
a 28-layer decode step gives about 0.058 ms, far below the measured 7-10 ms
single-GPU gap.  Precompiling the wrapper is therefore not the principal fix.

A more substantial serving-op candidate fused the one-token KV-cache scatter
with paged decode attention.  The current token is read directly from the
incoming K/V tensors, while independent grid blocks write contiguous K/V data
for the next token; this avoids a cross-block visibility dependency.  Against
the original two-kernel path, FP16 tests at batch 1/head-dim 64 and batch
4/head-dim 128 produced bit-identical attention output and complete cache
contents (`max_abs=0`).  A synchronized component microbenchmark measured
about 38.7 us for the two calls and 27.8 us for the combined call, or 1.39x.

That isolated gain did not survive real-model A/B.  With Qwen3-0.6B FP16,
input 128/output 32, two warmups and five rounds, the first uncontended
baseline/candidate pair measured 57.198/57.185 tok/s at batch 1 and
213.657/209.528 tok/s at batch 4.  Median ITL was identical at 16.737 ms for
batch 1 and worsened from 17.877 to 18.195 ms for batch 4.  A second reverse
order pair was partly contaminated by new GPU activity and also showed no
candidate advantage.  The fused route consequently remains guarded by
`JITTOR_VLLM_FUSED_CACHE_DECODE=1` and is disabled by default.

This result narrows the diagnosis: neither cached-`jt.code` object creation
nor the standalone cache/decode launch pair explains the overall PyTorch
gap.  The next useful measurement must capture unsynchronized whole-step CPU
submission time and the actual CUDA timeline separately; module-by-module
synchronization exaggerates small-kernel costs and cannot distinguish Python
adapter work, lazy-graph publication, and GPU execution.

Evidence:

- Baseline A/B run: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-fused-ab-baseline1.json`
- Fused candidate A/B run: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-fused-ab-candidate1.json`
- Reverse-order candidate/baseline: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-fused-ab-candidate2.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-fused-ab-baseline2.json`

## Follow-up: whole-step submission and synchronization

An uninstrumented-per-module stage recorder was run in separate Jittor and
PyTorch processes.  It adds no synchronization inside `engine.step`, so it
does not amplify every small kernel as the earlier module decomposition did.
For batch 1, median Jittor/PyTorch decode steps were 17.268/9.775 ms.  The
scheduler (0.035/0.035 ms), scheduler update (0.021/0.018 ms), and remaining
host work (0.119/0.116 ms) were effectively identical.  The gap was localized
to model execution at 14.729/9.183 ms and sampling at 2.385/0.424 ms.  Batch 4
showed the same result: 15.275/9.493 ms in model execution and 2.601/0.431 ms
in sampling.

The accompanying cProfile pass found 1,239 `sync_all` calls and 597 targeted
`sync` calls across three batch-1 generations.  The attention adapter's
rolling 32-layer boundary accounted for 84 of the global synchronizations and
127.7 ms of self time.  The Triton bridge issued every one of the 597 targeted
synchronizations, even when all kernel operands were already materialized.
The regular decode trace contained six bridge launches per step:
`prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`,
`gather_block_tables`, `compute_slot_mappings`,
`get_num_sampled_and_rejected`, and `post_update`.

Two changes were retained:

- The Triton bridge now builds a submission target only from operands whose
  Jittor Vars are not finished.  Materialized persistent serving metadata has
  a stable allocation and is already ordered on the shared per-thread CUDA
  stream, so rebuilding its execution plan did no work.  A CUDA regression
  test verifies that finished operands skip `jt.sync`, while the existing
  vector-add test continues to cover fresh operands and numerical output.
- The attention adapter no longer globally synchronizes every 32 layers by
  default.  Sampling's bridge submission already materializes the logits graph
  once per decode step.  `JITTOR_VLLM_LAYERS_PER_SYNC=32` remains available as
  a conservative override; the serving default is 1,000,000 layers, which
  preserves a rare safety boundary without draining every token.

With both changes, the stage diagnostic reduced batch-1 median decode from
17.268 to 16.784 ms.  `execute_model` fell from 14.729 to 13.110 ms; some of
the deferred wait moved into sampling (2.385 to 3.519 ms), leaving a real net
reduction of about 0.48 ms per step.  Targeted `sync` calls fell from 597 to
402 and rolling attention `sync_all` calls disappeared.

The final uninstrumented, default-configuration Qwen3-0.6B run used FP16
eager execution, input 128/output 32, two warmups and seven measured rounds:

| Batch | Adjacent old default | New default | Gain | Median ITL old/new |
|---:|---:|---:|---:|---:|
| 1 | 57.304 tok/s | 58.503 tok/s | +2.09% | 16.717/16.371 ms |
| 4 | 214.300 tok/s | 218.527 tok/s | +1.97% | 17.839/17.458 ms |

The seven-round ranges did not overlap.  A separate input-128/output-128 run
completed for batches 1 and 4 with matching greedy outputs and clean worker/
shared-memory shutdown, providing a longer lazy-graph/lifetime check.  Against
the earlier same-GPU PyTorch reference, the new default reaches 56.95% of
PyTorch at batch 1 and 54.69% at batch 4.  The remaining gap is still mostly
inside model execution and the four-to-six serving metadata/sampling bridge
launches; it is no longer consistent with Python scheduler overhead.

Evidence:

- Whole-step Jittor/PyTorch: `/home/chk/jittor-lab/_state/vllm-tp/20261004/whole-step-jittor/report.json` and `/home/chk/jittor-lab/_state/vllm-tp/20261004/whole-step-oracle2/report.json`
- Pending-only whole-step diagnostic: `/home/chk/jittor-lab/_state/vllm-tp/20261004/whole-step-pending-only/report.json`
- Adjacent old default: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-layer-sync32-adjacent7.json`
- Final new default: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-default-million-final7.json`
- Long-output stability: `/home/chk/jittor-lab/_state/vllm-tp/20261004/tp1-pending-no-layer-long128.json`
