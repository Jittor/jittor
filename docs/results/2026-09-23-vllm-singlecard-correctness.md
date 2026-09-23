# vLLM single-card correctness follow-up

- Status: Qwen regression, bounded attention diagnosis and OPT greedy/seeded inference verified; cross-framework random-token equality is not claimed.
- Date: 2026-09-23.
- Baseline: `44d7e04176813e705fda39bc2c6cb58139753980`, branch `chk`, plus the accompanying changes. `origin/chk` was fetched and remained `61294cd14673ba60f4072542a2e73ea2c8f8c509`.
- Owner: vLLM adapter and Torch compatibility maintainers.
- Review when: model arithmetic, sampling, host metadata, generator state, KV cache,
  or adapter dependencies on public compatibility APIs change.

## Scope and configuration

This follows the [initial acceptance](2026-09-23-vllm-uva-topk.md). It verifies
single-card FP16 eager inference, not training, HTTP serving, distributed,
quantized, graph or compiled execution. Both isolated environments use
vLLM 0.24.0 and transformers 5.5.3 on RTX 4090. The independent reference is
PyTorch 2.11.0+cu130; Jittor 1.3.11.0 uses CUDA toolkit 12.2. Both runners assert
which runtime supplies Torch. Model weights are the same local Qwen3-0.6B and
OPT-125m artifacts used by the prior acceptance. These observations do not
establish that another developer's earlier version or environment failed.

Preserve each backend's established default: Jittor enables CUDA; native
PyTorch keeps its CPU factory default. Actual model parameters, KV cache and
recorded model outputs are checked for CUDA placement. Equal global defaults
are not a prerequisite for comparing real GPU inference. No forced CPU default,
argmax replacement for random sampling, or numerical clamping is used.

Raw artifacts are **unversioned** under
`$JITTOR_LAB_ROOT/_state/vllm-singlecard/20260923/`. Current runs are in
`restored-default/`; older default-device experiments retain their original
paths. Concurrent tasks use separate devices and Jittor caches. Source hashes
identify the final worktree; first-build time and diagnostic runs are not
performance evidence.

## Qwen request and generation regression

`request_state_acceptance.py` uses max_num_seqs=4, max_model_len=1536,
max_num_batched_tokens=512, chunked prefill and prefix caching. Each backend
completed **51 requests / 692 output tokens**, with all seven internal assertions
passing. Actual scheduler records show turnover of queued requests, completion
notifications, new and existing requests in the same step, and four active
requests.

| Check | Result |
| --- | --- |
| Eight isolated greedy requests and three batches, including reversed order | Batch outputs equal isolated outputs within each backend |
| Mixed input/output lengths, including 512/661/1024-token inputs | Completed; repeated mixed seeded batches reproduce within each backend |
| Cross-backend greedy comparison | All 43 requests / 560 tokens exactly equal |
| Cross-backend random requests in that matrix | 5 of 8 complete outputs equal; the 3 differences repeat the same seeded batch case |
| 640-token prefix, cold/hit/reset | Cached tokens 0 → 624 → 0; all three 24-token outputs equal within and across backends |
| Separate random generation: seeds 10, 10, 11, 12, 32 tokens each | Both backends repeat seed 10 exactly and produce three distinct outputs; seeds 11/12 match across backends, seed 10 first differs at output index 6 |
| Separate 1024/3072-token inputs, 128 new tokens each | All 256 generated tokens exactly equal across backends |

The last two checks use `acceptance.py`, with GPU placement checks enabled.
The mixed-batch seed case also differs between isolated and batched execution
in native PyTorch, so that difference alone does not establish state leakage.
The tests found no state leakage within their scope; abort, preemption under KV
exhaustion, indefinite stress and HTTP load are not covered. These counts are
finite output comparisons, not model-task accuracy percentages.

Artifacts: `restored-default/qwen-matrix/{oracle,jittor}-{state,random,long}.json`,
their logs and `comparison.json`. Preliminary results before the final legacy
buffer repair are preserved under `qwen-matrix/before-legacy-buffer-fixes/`.
`compare_singlecard_acceptance.py` checks completion, configurations, prompts,
greedy tokens and seed contracts, and reports stochastic differences explicitly.

## Random divergence: identical-input sampling replay

`sampling_trace.py` forces the same 16-token reference prefix for
"Write a short story about a cat." Settings are temperature=0.8, top-k=40,
top-p=0.9, seed=10. It records the original sampler's choice before substituting
the reference token. Forced outputs are a diagnostic intervention, **not**
successful-generation parity evidence.

At every step, mapping, temperatures, seeds, positions, top-k and top-p match.
Raw model logits already differ at step 0; maximum absolute differences across
the 16 steps range from 0.015869 to 0.031250. The first sampled-token divergence
is zero-based step 6, position 13:

| Observation | Jittor | PyTorch |
| --- | ---: | ---: |
| Top-four probability mass after temperature and top-k, float64 diagnostic | 0.8997726773 | 0.9002196644 |
| Candidates retained by top-p=0.9 | 5 | 4 |
| Sampled token | 8171 | 8251 |

Both captured raw-logit sets were then independently replayed through both
runtimes. Across **32 input steps**, temperature results, filtered logits/masks
and sampled token IDs are exactly equal for the same inputs. Already-filtered
replay agrees too, and each replay reproduces its original capture's choice.
This identifies the top-p boundary mechanism without altering the sampler.
The compact real-logit fixture `test_top_p_boundary.py` also passes both cases
on each runtime against an independent float64 nucleus reference.

Fresh artifacts: `restored-default/sampling/`, including both captures, four
cross-replays and `comparison.json`.

## First model-operator difference: attention precision

`model_numerics_trace.py` captures the first real request. Embedding, first
input RMSNorm, QKV projection, Q/K normalization and rotary-position outputs
are bitwise equal across runtimes, as are all eight captured first-layer
parameter tensors. The first attention receives **identical Q/K/V**. Its output
first differs: **1827 / 16384 elements**, maximum absolute difference
**0.00048828125**.

`replay_attention_numerics.py` runs those same saved Q/K/V inputs on real CUDA:
Jittor paged attention and the oracle's FlashAttention 2 each exactly reproduce
their own model capture. An independent NumPy float64 causal/GQA attention
formula gives:

| Error against float64 reference | Jittor | Oracle FlashAttention 2 |
| --- | ---: | ---: |
| Maximum absolute error | 0.000240366 | 0.000279775 |
| RMS error | 0.0000227094 | 0.0000232542 |
| Elements different from the reference rounded to FP16 | 18 | 1827 |

The evidence supports different attention numerical paths rather than a
mathematical error demonstrated in this Jittor operator. Jittor is closer to
the high-precision reference for this particular input. Emulating FP16 rounding
of intermediate exponentials reduces the oracle mismatch to 94 elements;
this supports an intermediate-precision explanation but is not an exact
reimplementation of its kernel. No precision-degrading production change was
made merely to force token parity.

Hooks can alter Jittor fusion, so their effect was checked: **all 151936 raw
logits match the corresponding capture without internal module hooks exactly**,
for both runtimes, including comparison with the fresh sampler captures.
`compare_model_numerics.py --sampling-dir ...` enforces that check and exact
operator replay. This evidence concerns the first prefill attention/input. It
does not prove every later decode operator correct, establish universal model
accuracy, or promise bitwise stochastic generation across implementations.

Artifacts: `restored-default/numerics/{oracle,jittor}.{json,npz,log}`,
`{oracle,jittor}-replay.{json,npz,log}`, `comparison.json`, and instrumentation
checks. Diagnostic synchronization invalidates these runs as performance data.

## Explicit exponential generator

The old explicit-generator exponential API rejected execution. Expanded tests
first produced **10 failed / 11 passed** before repair. Native
`jittor.ops.random.CounterGenerator` now provides a Philox4x32-10 counter stream
through native CPU/CUDA kernels. The frontend validates device/rate/dtype and
uses inverse-CDF exponential sampling with in-place/view writeback.

State is reserved at lazy-graph construction, so later reseeding or evaluation
order cannot change an already-created draw. CPU ByteTensor snapshots preserve
the native stream and historical NumPy factory stream. CUDA offsets support
vLLM's four-word rewind. Empty draws do not advance state; pickle/deepcopy
restore independent locks and state. Clone tests initially failed before repair.

Final related regressions: **30 GPU passes; 27 CPU passes and 2 CUDA-only skips**.
Native known-answer, high-seed-bit, lazy-state and invalid-state tests: **6 CPU
and 6 CUDA passes**. Independent PyTorch tests cover matching public contracts;
the final factory-offset boundary test also passes on the oracle.

The generator is not PyTorch-bitwise and does not claim a unified PyTorch RNG
sequence. Historical factory draws remain a separate NumPy stream. After such
draws, CUDA offset APIs explicitly reject the unsupported stream instead of
pretending an exponential counter can rewind it; full-state restore remains
supported. Device-native support for every generator consumer is not claimed.

## OPT metadata failures and repair

Restoring the normal CUDA default exposed two legacy-runner metadata contracts,
separate from the deferred CPU-default experiment:

1. `optimistic_seq_lens_cpu` was created by an unqualified `torch.zeros`, then
   used as an output of CPU arithmetic. The Jittor default placed it on CUDA.
   A failing minimal test established the mismatch; the adapter now moves only
   this private host buffer to CPU, preserving the global CUDA default.
2. `CpuGpuBuffer.np` and twelve `InputBatch` host arrays assumed `.numpy()` was
   writable shared storage. It is a snapshot in this frontend: a host update
   `[0, 6, 6, 6]` was followed by a GPU copy of `[0, 0, 0, 0]`. Full/partial
   transfer tests failed before repair; the twelve real InputBatch alias tests
   also failed before repair. The adapter now uses the public `jt.Var.data`
   writable view and reacquires it after CPU tensor writes replace storage.

The local `_make_prompt_token_ids_cpu_tensor` helper had the same detached
NumPy write problem. Two nonempty prompt fixtures failed before repair; zero
length and empty-batch behavior were retained. The adapter now populates its
returned CPU tensor through the public writable view, including vocabulary-size
padding. The metadata suite passes **22 tests** on the shim; the independent
oracle passes its **20 real-helper tests**. Tests cover alternating NumPy/tensor writes,
partial upload/download, repeated updates and the no-NumPy option. This is a
bounded vLLM helper adaptation, not a new general `torch.Tensor.numpy` aliasing
promise. A previously retained ndarray can still refer to old storage after a
Jittor tensor assignment; inspected vLLM callers reacquire the relevant views.
No attention mathematics was changed to hide empty sequence metadata.

OPT now completes real CUDA generation on both isolated runtimes, with 148
model parameter tensors, 12 KV-cache tensors and all observed forward outputs
on CUDA. **All 32 greedy tokens exactly match the fresh independent oracle.**
Seeds 10/10/11/12 each generate 32 tokens: both runtimes reproduce the repeated
seed and produce three distinct sequences. **None of the four random sequences
is exactly equal across runtimes**; the explicit RNG streams are not bitwise
aligned. This establishes the tested seed/isolation and execution contracts,
not stochastic token parity or distributional equivalence for the entire model.
Artifacts: `restored-default/opt/`, including preserved red tests, final JSONs,
logs and `comparison.json` with artifact hashes.

A further real OPT test uses repetition=1.1, presence=0.2, frequency=0.3 and
greedy 32-token generation. The oracle completes; the shim explicitly fails on
missing `torch.ops._C.apply_repetition_penalties_`. The retained
`acceptance.py --case penalties` is a failing feature-boundary reproducer, not
a passed acceptance case. KI-VLLM-003 records this unimplemented private op.
The host metadata helper is repaired, but penalty-enabled inference is **not**
accepted and no no-op substitute was introduced.

After these adapter changes froze, the complete Jittor Qwen state/random/long
matrix was rerun; the results above remained unchanged. The independent oracle
uses the unchanged binary runtime and the paired reference runs from this turn.

## Deferred CPU-default experiment

A later diagnostic forced `torch.set_default_device("cpu")` after enabling
Jittor CUDA. In this frontend that sets global `use_cuda` to 0; Qwen subsequently
met a CPU causal mask while scores were on GPU. The normal runners no longer
apply that override, as requested by the user. KI-VLLM-002 retains the failure
and reproduction conditions. No paged-attention repair is claimed for that
configuration; passing the established CUDA configuration does not close it.

## Repository checks and remaining scope

- CUDA Torch core selection: **213 passed, 2 skipped**. The earlier 9 loss-test
  failures traced to ambiguous generated CUDA `isnan/isinf` overloads with
  GCC 12. A new float32 predicate case failed before selecting the CPU/CUDA
  namespace explicitly; complete predicate tests then passed **6 CPU + 6 CUDA**.
- Optional-pointer, packed-mask, stochastic sampler, UVA, logprob and top-p
  boundary regression: **31 passed**, no skips.
- Layout, generated-manifest and whitespace checks: passed.
- Final metadata/UVA/adapter structure regression: **45 passed**. Controlled
  lifecycle/arming/ownership tests in their own host-only process: **11 passed**.
  An initial combined invocation gave 50 passes and 2 import-ownership fixture
  errors; real-library tests and fake-module lifecycle tests must be isolated.
  Neither production ownership checks nor test assertions were weakened.
- Full structure run: **1367 passed, 5 failed, 4 skipped**. Four failures refer
  only to unchanged baseline files `tests/core/test_executor_python_threads.py`
  (interpreter launch policy) and `tests/integration/test_h3_decode_thread_race.py`
  (Torch namespace/collection side effects). The fifth, process-mode collection,
  stopped because a cold child rebuilt `jit_utils` and requested restart; pytest
  wrapped that SystemExit as collection exit 2. The warm rerun of that case plus
  documentation, packaging and flag-scope checks passed **38 tests**, leaving
  the four unchanged baseline policy failures. The full repository is not
  claimed green.

Penalty-enabled OPT, multi-GPU, quantization, CUDA Graph/compilation and performance optimization are
not part of this closure. The prior performance report remains historical:
these changed sources have not been benchmarked again. No standard language
model task-accuracy benchmark was run.

## Reproduction

After activating each isolated backend environment, preserve its default device
and run with the same model path and fixed settings:

```bash
python agent/skills/vllm-torch-compat/request_state_acceptance.py \
  --backend "$BACKEND" --model "$QWEN_MODEL" --output "$RESULT_DIR/$BACKEND-state.json"
python agent/skills/vllm-torch-compat/acceptance.py \
  --backend "$BACKEND" --case random --model "$QWEN_MODEL" --logprobs 5 \
  --check-gpu-placement --output "$RESULT_DIR/$BACKEND-random.json"
python agent/skills/vllm-torch-compat/acceptance.py \
  --backend "$BACKEND" --case long --model "$QWEN_MODEL" \
  --check-gpu-placement --output "$RESULT_DIR/$BACKEND-long.json"
python agent/skills/vllm-torch-compat/compare_singlecard_acceptance.py "$RESULT_DIR"
```

For numerical diagnosis, capture with `sampling_trace.py`, cross-replay both
NPZs under both runtimes, then run `compare_sampling_traces.py`. Capture internal
boundaries with `model_numerics_trace.py`, replay attention with
`replay_attention_numerics.py`, and run `compare_model_numerics.py NUMERICS_DIR
--sampling-dir SAMPLING_DIR`. See each tool's `--help` for artifact arguments.
