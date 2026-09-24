# vLLM: remove compute-policy work from metadata reads

- Status: correctness accepted and redundant callbacks reduced in real-model
  captures; no accepted end-to-end speedup yet because GPU occupancy varied.
- Owner: Codex; reviewed 2026-09-24.
- Production change: `3401bf005016f5aa75359126385dc15830088b18`; the same 1164-file
  manifest was rechecked on the GPU checkout after synchronization.
- Baseline: `dbb0379aec4843e908df4970065d866ce93aa272`. The fetched `origin/chk`
  remains ancestor `61294cd14673ba60f4072542a2e73ea2c8f8c509`; no upstream
  changes were overwritten. Before execution, 1164 production source hashes
  on the GPU checkout matched the local baseline.
- Scope: Qwen3-0.6B and OPT-125m, vLLM 0.24.0, FP16 eager, one RTX 4090.
  Existing Jittor CUDA factory defaults are preserved.
- Review when: bindings, frontend policy, model weights, vLLM, precision,
  execution mode or scheduling changes.

Raw evidence is unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-performance/20260924/`.
The [previous acceptance report](2026-09-23-vllm-singlecard-extended-acceptance.md)
contains the last accepted isolated performance baseline.

## Cause and bounded change

The binding generator previously created a complete `PyTensorFrontendScope`
for every VarHolder entry, including pure metadata getters. That selects
frontend result type, autograd and placement policy, then calls Python to
read mutable precision settings even when the native method only reads a
field. In a warm three-generation batch-one capture (96 output tokens), the
precision callback ran 1017420 times; dtype access alone contributed 255468
calls. The new capture reproduces the previous profile's counts.

Only audited metadata declarations now carry `frontend_metadata`: `dtype`,
`shape`, `ndim`/`dim`, `device_id`, `placement_backend`, `numel` and `nbytes`.
The generator omits frontend scope only when every overload of that binding
has the annotation. This is not a blanket exemption for getters or scalar
return values. Materialization, data access, mutation and Tensor-returning
operations retain their scope.

Metadata values are still read from the current native holder on every call.
Nothing caches a Tensor's dtype/shape, mutable precision settings, or an old
installation context. Thus holder replacement can change metadata immediately,
while actual computation continues to select current policies. No GPU math,
sampling algorithm, UVA synchronization or temporary-buffer lifetime is changed.

## Reproduction and regression

Before production edits, CPU behavior tests failed 16 cases and passed three;
each group of eight metadata reads made eight redundant policy callbacks.
The generator tests failed two cases and passed two, including the mixed
overload control. A further Parameter control is included in the final suite.
On the unchanged CUDA source, 17 behavior cases fail and three pass.

After the edit, local CPU behavior/generator checks pass all 24 cases without
skips. They cover Tensor subclasses, Parameter result type, holder dtype/shape
rebinding, no new live Vars or forced materialization, mutable computation
policy and nested/exception restoration. CPU core plus these checks passes
236 cases with three skips. Existing parser and CPU oracle checks pass 26;
14 TF32/precision checks cannot enter their bodies in the CPU-only build,
which lacks `cuda_allow_tf32`, so these require the real CUDA build.
The real CUDA suite subsequently passes **47 cases with no skips**, including
the metadata and generator cases, public TF32 setters, deferred matmul/conv
precision, RNN precision and restoration controls. Evidence is retained in
`metadata-cuda-red.log` and `metadata-cuda-green.log`.
The repository structure gate reports 1364 passed, four failed and eight
skipped. The four failures are the unchanged interpreter-launch/collection
violations in `test_executor_python_threads.py` and `test_h3_decode_thread_race.py`,
also present in the previous baseline. Layout, manifests and whitespace pass;
the entire repository is not declared green.

Both real-model matrices complete on `3401bf0`: **70 requests and 1492 output
tokens per model**, covering state/cache, seeded random generation with
logprobs, batches, penalties and long inputs. All **140 requests / 2984 tokens**
match the previously accepted Jittor outputs exactly, including random cases.
The independent native-reference comparator also passes the declared greedy
scope: 58 requests / 1232 tokens per model. Native matrix references are reused
from the accepted `matrix-r3` run with unchanged options and dependencies;
this is a correctness comparison, not a new native timing measurement.
Evidence is in `matrix/candidate-token-comparison.json` and per-model reports.

The first OPT initialization is retained under `matrix/opt-init-failure-r1/`:
another process freed GPU memory during vLLM's memory profiling, changing free
memory from 13.72 to 22.05 GiB and triggering its consistency assertion. A fresh
run completes without production changes or relaxing the assertion.

## Stage diagnosis and measurement limits

`stage_timing.py` runs the same prompts and eager options as
`latency_acceptance.py`. It warms the model, measures disjoint scheduler,
model execution, sampling and result-update host spans, then removes its
wrappers before a separate cProfile pass. No per-step CUDA synchronization
is added. Both passes assert unchanged generated tokens and save their source
hashes. These are host wall times, including existing waits, not GPU kernel
times and not the uninstrumented throughput gate.

Both baseline runtime captures complete, with matching options, prompt IDs,
script hashes and tokens. Their decode host medians are retained as diagnostic
data only: GPU occupancy changed during the session. Most observed time is
inside model execution and sampling; scheduler time is comparatively small.
There are no `parameter_new`/`parameter_init` calls in the warm baseline
captures, so repeated Parameter construction is not an established decode
bottleneck. Repeated temporary factories, same-dtype cast calls and duplicate
kernel-support validation are separate candidates for subsequent measurement.

After synchronization to `3401bf0`, both runtimes are captured again with the
same script, options, prompts and per-batch token outputs. Each capture contains
three generations (96 output tokens at batch one, 384 at batch four).

| Batch | Precision callbacks before | After | Reduction | Total Python/profile calls before | After |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1017420 | 362652 | 64.36% | 12600001 | 11290465 |
| 4 | 1065696 | 386841 | 63.70% | 13105837 | 11748127 |

These call reductions are deterministic evidence of removed work, not a
throughput percentage. Dtype reads remain 255468 / 255477 respectively, and
installation-context lookups remain 342660 / 350760: this change does not cache
their values or remove their ownership validation. Parameter constructor counts
remain zero. The shared-device host timings below are retained to guide the
next experiment; they are **not an accepted speedup claim**.

| Batch | Runtime / source | Prefill step median ms | Decode step median ms |
| --- | --- | ---: | ---: |
| 1 | PyTorch before | 18.91 | 18.35 |
| 1 | Jittor before | 97.03 | 61.43 |
| 1 | PyTorch after | 16.45 | 17.10 |
| 1 | Jittor after | 77.91 | 43.50 |
| 4 | PyTorch before | 18.09 | 19.18 |
| 4 | Jittor before | 103.73 | 62.90 |
| 4 | PyTorch after | 16.99 | 19.12 |
| 4 | Jittor after | 82.36 | 47.09 |

Phases refer to scheduled prefill/decode work, not token delivery latency;
these are not TTFT/ITL. After the edit, the observed decode model-execution
medians are 39.87 / 43.00 ms versus native 16.34 / 18.12 ms for batches one /
four. Sampling and repeated dtype/context/dispatch work remain useful follow-up
targets. CPU and GPU timeline evidence is still needed before changing buffer
reuse or correctness synchronization. `profile-comparison.json` checks all four
capture identities and tokens; `stage-source-identities.json` ties the phase
labels to the verified source manifests.

Two formal benchmark attempts were stopped when other users entered the
selected GPU. `baseline/excluded.txt` and `baseline-r2/excluded.txt`, process
snapshots and `gpu-observations.jsonl` preserve why they are excluded. Shared
GPU measurements must not be presented as an accepted optimization speedup.
