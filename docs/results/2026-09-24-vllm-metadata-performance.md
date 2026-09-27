# vLLM: remove compute-policy work from metadata reads

- Status: correctness accepted; the 2026-09-25 uncontended-GPU comparison
  measures 11.8–12.5% higher throughput (8.6–9.0% relative to native drift).
- Owner: Codex; reviewed 2026-09-27.
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

## 2026-09-25 retry and next isolated reproducers

At local `aa6f750`, `origin/chk` still has no new commits. The production
baseline remains `3401bf0`. A new attempt prepared three independent processes
per backend and source phase, with continuous two-second GPU/process-group
observations, source-hash checks, interruption cleanup and automatic restoration
of the two original/candidate files. An idle card acquired another user's task
during the first native process, so only our process group was stopped and the
run was excluded. A retry on another card was rejected by the occupancy check.
Both early attempts were excluded. A later retry completed all 12 processes
on the same physical GPU with no foreign process observed on that GPU. All
1164 production source hashes match the restored candidate. Five injected
monitoring/cleanup failure cases pass after fixing the external experiment
supervisor.

Raw scripts, excluded attempts and status are under
`$JITTOR_LAB_ROOT/_state/vllm-performance/20260925/isolated-round{1,2}/` and
`isolated-status.json`. The successful later retry is in `isolated-round2/`.
The source-switching script must not be resumed after further production edits without reconciling
its source manifests. It preserves accepted rounds and archives interrupted
ones; it does not terminate other users' processes.

While waiting, the next candidates were verified without changing production:

- Factory final casts: before the next patch, both CPU and real CUDA suites
  report **41 failed, 12 passed** across 53 cases.
  The failures observe redundant same-dtype casts, after checking values, dtype
  and placement; they do not mean 41 independent numerical bugs. Controls cover
  gradients, default-dtype changes, like inheritance and holder replacement.
  Native `normal` can promote a tensor mean after creating its typed random
  input, and `ones_like`/`tril`/`triu` still need some final conversions. Therefore
  accepting a dtype argument is insufficient grounds to remove every cast.
- KV-update capability: the patched attention forward already writes KV, but
  the backend still advertises a separate update. Two host contracts fail and
  three pass in the installed vLLM 0.24.0 environment with CUDA hidden. The
  passing checks cover source call guards, current-step write-before-read
  ordering and the absent-implementation boundary. The proposed change needs
  owned publication/rollback and real CUDA cache regression before acceptance.
- Full RMSNorm validation is repeated in selection and execution. Sharing a
  result would require preserving current dtype, shape, autograd and precision
  eligibility; no cross-call cache or buffer reuse was introduced.

The original reproducers, recoverable patches/audit and red logs are retained
in the unversioned `factory-cast/` and `kv-update-capability/` directories of the
same run. The KV capability candidate remains unmodified. The factory follow-up
is recorded below, separately from the metadata measurement.
An independent CPU check also reproduces an existing `empty_like` default
gradient-flag mismatch against binary PyTorch; see KI-COMPAT-005 in the
[issue ledger](../../agent/manuals/known-issues.md).

### Accepted metadata timing comparison

Three independent processes per backend and source phase each perform three
warmups and 21 measurements for batch 1 and 4, with 128 input and 32 output
tokens, FP16, eager execution and prefix caching disabled. Initialization and
compilation are excluded. All 1,260 measured requests / 40,320 output tokens
match across the 12 processes. This is offline engine token delivery timing.

| Phase/backend | Batch | Output token/s | TTFT median / p95 (ms) | ITL median / p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original Jittor | 1 | 17.01 | 158.58 / 167.69 | 56.41 / 59.92 |
| Metadata-optimized Jittor | 1 | 19.14 | 143.24 / 152.08 | 50.53 / 55.66 |
| PyTorch before | 1 | 51.18 | 39.54 / 44.04 | 19.13 / 21.61 |
| PyTorch after | 1 | 53.00 | 37.63 / 41.41 | 18.63 / 20.74 |
| Original Jittor | 4 | 64.61 | 164.71 / 172.83 | 59.88 / 62.88 |
| Metadata-optimized Jittor | 4 | 72.23 | 149.40 / 156.61 | 53.62 / 57.05 |
| PyTorch before | 4 | 198.42 | 40.40 / 43.47 | 19.99 / 21.38 |
| PyTorch after | 4 | 203.52 | 38.75 / 41.23 | 19.30 / 20.76 |

Throughput uses the median of process rates; latency distributions pool measured
requests/tokens. Jittor's raw gain is 12.50% / 11.78% at batch 1 / 4. Native
PyTorch improves by 3.56% / 2.57% between phases. Dividing the Jittor ratio by
the corresponding native ratio gives an **8.64% / 8.98%** relative gain. This
normalization is a sensitivity check, not proof that every source of timing
drift is removed: other GPUs and the host remain shared. Optimized Jittor still
reaches only 36.11% / 35.49% of paired native throughput on these inputs.

Per-process rates, complete distributions and source/GPU observations are in
`isolated-round2/{baseline,candidate}/comparison.json` and
`isolated-round2/cross-phase-comparison.json`. The previously excluded runs
remain archived. Do not mix these numbers with short-prompt legacy benchmarks.

### Factory final-cast follow-up

After the metadata comparison completed, commit `17d2cfd` limits its production
change to `compat/torch/installers/factories.py`: it skips the constructor's final cast
when the current output dtype already equals the requested dtype. Both names
are normalized for comparison. Actual-result queries preserve tensor-argument
promotion cases; no tensor dtype, default dtype or installation context is
cached. No parameter construction or output-buffer reuse is introduced.

The retained `compat/tests/torch/test_factory_redundant_cast.py` suite changes
from 41 failed / 12 passed to **53 passed on CPU and 53 on real CUDA**. Related
CPU factory/dtype/generator/autograd tests report 100 passed and two skipped
(unavailable native-Torch references). One test that explicitly requests CUDA
cannot run on the CPU-only host; its initial failure is retained and its GPU
verification passes (one case, no skips). The layout and package-manifest checks
pass. The structure gate remains at 1364 passed, four unchanged baseline
failures and eight skips; no new structure failure is introduced.

The real Qwen and OPT matrices each complete 70 requests / 1492 output tokens
(state, random with logprobs, batch, penalties and long inputs). All 140 requests /
2984 tokens match pre-patch Jittor, including seeded random output. Each model's
58 greedy requests / 1232 tokens also match the retained native reference;
those old native artifacts are used for correctness only. Results are in
`factory-cast/matrix/` and `factory-cast/matrix-comparison.json`.

The next formal timing comparison uses the accepted metadata candidate as its
baseline and changes only the factory file (full 1164-file manifest checked).
Its initial preflights were blocked by foreign GPU processes. The September 27
comparison below completes the formal timing protocol; stable speedup remains
unconfirmed. Runs and occupancy evidence live in `factory-perf/`. The KV
capability and RMSNorm candidates remain unmodified.

A separate real-model cProfile capture on the shared GPU preserves the same
generated tokens and confirms 5664 direct constructor casts disappear per
batch-size capture. Precision callbacks decrease by 5664, but dtype queries
and installation-context lookups each **increase by 5664**. Parameter
construction remains zero. This exposes the cost of the safe actual-result
check; fewer casts alone do not establish a net speedup. Shared-device timings
are excluded. See `factory-cast/profile-comparison.json` and
`factory-cast/stage-diagnostic/`. Two formal preflights were rejected by the
GPU occupancy check. `17d2cfd` has correctness acceptance. The September 27
measurements below are positive at the median but do not establish a stable
performance gain; further changes must still be evaluated independently.


### Bounded parameter and snapshot lifetime audit

A later retry starts from clean `08643f3` / production `17d2cfd`; fetched
`origin/chk` remains `61294cd`. All four GPUs initially have foreign workloads.
One formal preflight is rejected; two later idle-window attempts start but are
excluded when foreign processes enter the selected GPU. No factory timing run
is accepted. Production remains unchanged during this audit.

`lifetime-round1/probe.py` runs Qwen on real CUDA in both independent runtimes.
After warming batch sizes 1, 4 and 2, it repeats that cycle four times: **28
measured requests / 448 tokens per runtime**, excluding seven warmup requests.
Every repeated output matches, including the native/Jittor comparison. All
**226 model parameters and 28 KV cache tensors** retain their Python identities,
storage addresses, shapes, dtypes and devices at the sampled batch boundaries.
No Parameter constructor occurs in the measured generation profile.

Jittor's post-request live/held tensors stay at **719**, live ops at **12**,
and reported allocated memory at **8,743,686,656 bytes** across all 13 samples.
Reserved memory grows once by 5 MiB after a warmed shape is reused, then stays
flat. Native allocated memory is also constant. These backend-specific
accounting values are not a cross-framework memory-efficiency comparison.
Each sample has zero unfinished requests. This bounded check does not prove
indefinite leak freedom, unchanged contents of every model weight, or behavior
under precision changes. KV contents are expected to change during generation;
only cache identity/storage stability is asserted.

The next observed lifetime-related hotspot concerns **per-request sampling
metadata, not model weights**. In the previous real-model profile, 99 engine
steps call 14 sampler metadata snapshots each (1386 total). The adapter must
finish the CPU source read before it can be overwritten, so each snapshot
currently synchronizes CUDA. Temperature/top-k/top-p/seed, penalties, bias,
bad-word and logprob metadata are submitted even during ordinary decode.
The existing empty staged-write queues already return early and are not bugs.

Host control-flow tests retain **four failed performance contracts and six
passing correctness controls**, under `parameter-lifetime-review/`. Controls
cover immediate source mutation, same NumPy object with new contents, and old
snapshot retention beyond pool rotation. They do not demonstrate a numerical
failure. Identity-only caching or removing the source-read barrier would
invalidate these guarantees. Any next optimization must detect real content
updates and preserve old consumers; a read-only sampler scope cannot silently
be generalized to writable UVA buffers.

Raw scripts, profiles, source extracts and reports are unversioned under
`$JITTOR_LAB_ROOT/_state/vllm-performance/20260925/{lifetime-round1,parameter-lifetime-review}/`.
These instrumented shared-GPU runs are not performance measurements.

The subsequent real-CUDA `sampling_snapshot_trace.py` pass records the bytes,
dtype and shape of each sampler submission, delegates to the unchanged
production transfer, and retains the first GPU snapshot from each array. In
14 requests / 224 output tokens (two cycles of batches 1/4/2), all **14 arrays**
are submitted 102 times: **1428 submissions create 1428 new GPU objects**.
After excluding the first observation of each array, **1409 submissions have
identical CPU contents** and **five have changed contents** (request seeds).
No zero-length submission occurs in this path, so the host empty-input cases
are not the current priority. All 14 retained initial GPU snapshots still equal
their original CPU values after subsequent requests, and every output token
matches the unhooked run. This confirms a reduction opportunity while also
showing why an identity-only cache would miss actual updates. It does not yet
prove that every GPU consumer or external plugin treats all snapshots as
read-only. Data and assertions are in `lifetime-round1/sampling-trace*.json`.

No further production optimization was layered onto the factory candidate
during the lifetime audit. Its subsequent timing comparison is below.


### 2026-09-27 factory comparison: positive medians, substantial variation

At clean local `fad53b6`, fetched `origin/chk` is still `61294cd`. The previously
blocked factory comparison completes on the same physical GPU. Two interrupted
candidate processes are excluded and archived when foreign GPU processes enter;
completed unaffected rounds are retained. Every included process has continuous
occupancy observations without a foreign compute process on the selected GPU.
Other GPUs and the host are shared, so this is not a fully isolated machine.

To avoid a cross-day comparison, both source variants are measured today:
`17d2cfd` first, followed by the pre-factory-patch production `3401bf0`. Only
`compat/torch/installers/factories.py` differs across the 1164-file production
manifests. Three fresh processes per backend/variant each use three warmups and
21 measurements per batch, FP16/eager, 128 input tokens, 32 output tokens, no
prefix cache. All **1260 measured requests / 40320 tokens** agree. Afterwards,
the candidate is restored and its full production manifest verified.

| Variant/backend | Batch | Median process token/s | TTFT median / p95 (ms) | ITL median / p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Before factory change, Jittor | 1 | 17.70 | 143.51 / 173.19 | 50.45 / 66.96 |
| Factory candidate, Jittor | 1 | 18.61 | 141.91 / 190.11 | 49.84 / 64.04 |
| Before factory change, PyTorch | 1 | 50.64 | 37.38 / 51.88 | 18.57 / 23.99 |
| Factory candidate, PyTorch | 1 | 52.20 | 38.03 / 42.96 | 18.73 / 21.68 |
| Before factory change, Jittor | 4 | 69.95 | 149.17 / 183.87 | 53.37 / 65.24 |
| Factory candidate, Jittor | 4 | 73.25 | 147.40 / 167.67 | 52.45 / 75.66 |
| Before factory change, PyTorch | 4 | 202.45 | 37.96 / 43.37 | 19.11 / 22.89 |
| Factory candidate, PyTorch | 4 | 203.38 | 38.18 / 41.31 | 19.25 / 21.27 |

Raw throughput ratios are +5.16% / +4.72% at batches 1 / 4. Dividing by the
corresponding native ratio yields +2.01% / +4.24%, a drift sensitivity check,
not a causal correction. Individual Jittor process rates overlap considerably:

- Batch 1 before: 17.70, 18.60, 15.27; candidate: 18.61, 19.29, 17.89 token/s.
- Batch 4 before: 74.01, 69.95, 61.92; candidate: 74.30, 73.25, 66.25 token/s.

Median TTFT/ITL changes are only about 1–2%, and tails do not improve uniformly:
single-request TTFT p95 and batch-four ITL p95 worsen in this sample. Thus the
measurement protocol is complete and medians favor the candidate, but **a
stable net speedup is not established**. Do not add this percentage to the
previous metadata improvement or promise lower tail latency. No further
production optimization is introduced in this measurement session.

Raw candidate runs remain under
`$JITTOR_LAB_ROOT/_state/vllm-performance/20260925/factory-perf/candidate/`;
today's baseline, source manifests, restoration and cross-phase comparison are
under `20260927/factory-ab/` beneath the same performance state root. The latter
links the candidate directory, rather than copying an old baseline. These are
unversioned artifacts; all run options, token IDs, per-process rates and latency
distributions are retained. Production remains `17d2cfd`; CUDA Graph, compiled,
quantized and multi-GPU modes are outside this comparison.
