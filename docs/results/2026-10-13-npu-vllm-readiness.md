# Jittor vLLM: ACL/HCCL readiness and CUDA cross-model checks (2026-10-13)

## Scope and status

Worktree: `chk-refactor2-integration`, branch `prefill-profile-20261012` on
`cscg-qh00`. The CUDA path runs vLLM 0.24.0; the only previous Ascend engine
evidence in this repository used vLLM 0.20.2 plus an *external, unversioned*
`vllm-jittor-npu` platform/worker plugin. That plugin is not in this checkout.
The ACL/HCCL changes below began as preparation. Qwen3-0.6B TP1 and TP2 eager
vLLM requests on Ascend 910B3 now each reproduce 64 PyTorch oracle tokens
under the ACL-default per-layer metadata readback. The earlier fast path
diverged after six tokens. **This is a correctness milestone, not production
readiness:** decode is slow, TP2 shutdown needs a longer grace period,
and broader prompt/model coverage is still in progress.
`cscg-hw00` became reachable at
2026-10-12 22:54 UTC. Its account had neither vLLM nor the previously used
external platform/worker plugin, so a separate Python 3.11/vLLM 0.24.0
environment and a fresh plugin integration are being built. Device compute
and HCCL have now been tested alongside the TP1 engine result below.

## ACL request-state and attention metadata

- `adapters/jittor_adapters/vllm/buffers.py`: CUDA-only native metadata kernels
  now explicitly reject ACL, where `use_cuda` is a compatibility alias.
  Under ACL, vLLM 0.24 request-state staging uses owned device tensors instead
  of CUDA UVA; small input metadata, block tables, slot mappings, token
  combination, computed-token counters and basic post-update have non-Triton
  fallbacks. The large all-token-ID and output-bin tensors are updated by
  touched slices/cells only, not copied in full each generated token.
  Context parallelism and sampling beyond plain greedy fail closed. These
  paths use CPU/device round trips and are not intended as final NPU fast paths.
- `adapters/jittor_adapters/vllm/backend.py`: ACL selects 128-token KV pages,
  which the current ACL paged-decode kernel supports; CUDA keeps the upstream
  block-size policy.

## HCCL TP bootstrap

- `compat/torch/installers/distributed.py`: dynamic `torch.distributed` setup
  now has an HCCL branch using Jittor's MPI-free `JT_HCCL_*` env/file
  rendezvous, `setup_hccl(no_mpi=True)` and a separate `hccl_init()` call
  before vLLM creates process groups. HCCL `all_gather` now uses the
  requested process-group communicator, not the NCCL-only generic helper.
  It sets local rank from
  `ASCEND_RT_VISIBLE_DEVICES`, publishes HCCL collectives and preserves the
  NCCL path.
- `python/jittor/distributed/process_group.py`: an HCCL world is now
  recognized as initialized.
- `backends/comm/hccl/src/hccl_wrapper.cc`: BF16 maps to
  `HCCL_DATA_TYPE_BFP16`, available in CANN 9.0 HCCL; this previously raised
  unsupported-dtype in TP BF16 all-reduce.
- `backends/comm/hccl/ops/hccl_reduce_op.h`: the constructor is kept on one
  line because Jittor's wrapper generator uses a line-bound regex. The
  previous multiline declaration prevented this op from being generated at
  all, before a real HCCL collective could run.

## Verification and remaining gates

The mock HCCL/NCCL bootstrap test passes (6 cases,
including global rank 1 isolated to one visible local NPU),
block-size policy test passes (1 case), ACL NumPy block-table/slot reference
and Torch-shim view-copy tests pass (7 cases), and the existing native process
group test passes (6 cases). `git diff --check` passes. These mock tests do
not execute a CANN/HCCL device kernel; the hardware checks below do.

The first six hourly `cscg-hw00` SSH checks (latest failure at
2026-10-12 21:54 UTC) timed out after the proxy jump; access was restored
at 22:54 UTC. The login account has eight Ascend 910B3 devices. Cards 1, 4,
and 7 were occupied by other users at inspection; hardware tests used free
cards 0 and 2/3 only. In isolated worktrees based on old `chk` and the CUDA
integration HEAD `c65396e5`, Jittor ACL computed `[1,2,3]*2+1` as
`[3,5,7]` on device 0. Two HCCL ranks on devices 2 and 3 each all-reduced
a BF16 scalar from rank+1 to exactly `3.0` and exited normally. These tests
ran against CANN 9.0.0, including the constructor/codegen and BF16 fixes
above. They establish one-card compute and two-card communication; the short
vLLM TP1/TP2 results below are separate tests, not long-run validation.

The original `chk` worktree was left untouched. The new integration worktree
is `npu-vllm-qh-integration-20261013`; a separate Python 3.11.17 venv at
`/beegfs/home/tuser028/venvs/vllm-jittor-npu-20261013` has the vLLM 0.24.0
wheel installed without its native PyTorch/CUDA dependency set. The old NPU
Python 3.9 env cannot run vLLM 0.24.0. The external plugin is absent from
the accessible account. A new opt-in `vllm.platform_plugins` entry point now
registers `JittorNpuPlatform`, reusing vLLM's GPU worker through the Torch
shim's ACL-backed CUDA API, forcing eager/no CUDA Graph, selecting 128-token
ACL FlashAttention pages and the HCCL distributed backend. Entry-point
discovery and syntax checks pass. The vLLM 0.24 engine now resolves Qwen3,
starts its worker, initializes a one-rank HCCL world, constructs the V1
GPUModelRunner and reaches `load_model`; missing platform dtype, compute-unit
and memory-accounting methods were filled from the actual worker errors.
Qwen3-0.6B weights and tokenizer were downloaded to
`/beegfs/home/tuser028/models/qwen/Qwen3-0.6B`. On NPU 0, the first eager
vLLM request generated IDs `[7281, 11, 10339, 304]`, exactly matching a
native PyTorch CUDA vLLM run with the same prompt, BF16 model and greedy
sampling. The first request took 82.84 s including ACL operator compilation;
the next clean-process request with cached kernels took 6.24 s for four
tokens. Both workers shut down normally. TP2 with devices 2/3 also generated
the same four IDs in 8.66 s. One TP2 worker needed the framework's SIGTERM
after its graceful-exit timeout, but no worker remained and both cards were
released. This demonstrates only short TP1/TP2 smoke correctness.

A separate TP1 64-token run took 7.03 s after engine initialization. It
matched the native PyTorch CUDA vLLM oracle for the first six token IDs, then
diverged at token 7 and eventually repeated token 264. Using the exact
PyTorch-generated prefix as a fresh vLLM prompt, NPU next-token predictions
matched the oracle at prefix lengths 0, 4, 5, 6, 7, 10, 20 and 40. Thus the
failure is associated with continuous incremental decode state rather than
the same model on a freshly prefilled sequence; this does not yet identify
which stateful component is wrong. An independent BF16 KV-write/paged-decode
attention check against NumPy passed at lengths 7, 11, 16, 17, 32, 64,
100 and 128: cache values were exact and maximum attention error was below
0.0007. A separate incremental test also appended one token at a time from
length 11 through 70; cache values stayed exact and sampled attention checks
had error below 0.0006. The complete model's decode state/metadata remains
under investigation. Diagnostic per-layer rereading of attention metadata
made an 8-token NPU request match PyTorch, whereas reading runner input
metadata, submitting/synchronizing the full graph once per step, and even
device-synchronizing every attention layer did not. Disabling the fast ACL
paged-decode kernel and using the generic reference attention still produced
the same wrong first eight tokens, taking 38.14 s instead of roughly 7 s;
the fast attention kernel alone is therefore not the explanation. A trial
stream fence after custom asynchronous KV memcpy likewise failed the 64-token
check and was reverted. Reading any ONE of slot mapping, query starts or
sequence lengths at every attention layer made an 8-token request match
the PyTorch sequence. The native PyTorch oracle at the first mismatch chooses
token 1128 with log-probability -0.353 versus -2.228 for its second choice;
the NPU's wrong token 624 is not in PyTorch's top 20. This is not a near-tie
BF16 rounding difference. With per-layer query-start rereads, a TP1 64-token
request matched every PyTorch oracle token (12.69 s request time, versus about
7 s for the incorrect old path). The same diagnostic path passed TP2 64-token
in 18.36 s. It is now the ACL default, gated by
`JITTOR_VLLM_NPU_SAFE_METADATA_READBACK`; without any diagnostic environment
variables, TP1 and TP2 64-token runs again matched all oracle IDs in 12.65 s
and 17.35 s. TP2's exit code was zero and both cards were freed, but one
worker required framework SIGTERM after the graceful-exit grace period.
Plain `jt.sync`, device-wide/layer sync, a lazy CPU copy, and a device clone
followed by sync all failed to replace the actual host readback. The cause of
this graph-lifetime sensitivity and a fast barrier remain open. The NPU and CUDA
timings are on different devices and must not be treated as a speed ratio.

The ACL-default TP1 path also crossed a KV page boundary: 130 input token IDs
(13 repeats of the 10-token probe prompt) plus 32 greedy generated tokens,
with 128-token KV blocks. All 32 generated IDs matched native PyTorch CUDA
vLLM exactly; the NPU request took 10.55 s after load, versus 0.35 s on a
different CUDA device. This is a correctness comparison, not comparable
hardware throughput. ACL-default TP2 on devices 2/3 also matched all 32 IDs
for the same 130-token prompt in 16.02 s. The TP2 request exited zero and
freed both cards, though one worker again needed framework SIGTERM on shutdown.
A separate TP2 four-token run with
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=20` let both workers exit gracefully in
about five seconds; the prior warning appears to be a too-short default
grace period rather than a persistent orphan. The local NPU test launcher now
defaults to 15 seconds, pending its own confirmation run.

## CUDA control results

SmolLM2-360M-Instruct, 128 prompt + 16 generated tokens, TP1, FP16,
decode-only CUDA Graph, 1 warmup + 3 measurements on physical RTX 4090 GPU6:

| Backend | Median full request | Median TTFT | Median ITL | Output throughput |
| --- | ---: | ---: | ---: | ---: |
| Jittor | 73.484 ms | 31.865 ms | 2.748 ms | 217.3 tok/s |
| PyTorch | 43.876 ms | 11.990 ms | 2.082 ms | 368.2 tok/s |

All 16 greedy token IDs matched in all three measured requests. Jittor is
59.0% of PyTorch output throughput for this narrow matched test; this is not
a general-model claim. Raw reports:
`/home/chk/jittor-lab/_state/vllm-compare-20261013/smollm-{jittor,torch}-gpu6.json`.

An opt-in experiment that advertised full prefill CUDA Graph support to vLLM
was first a no-op because `use_acl` aliases `use_cuda` on CUDA builds. After
correcting that guard, actual 128-token FULL graph execution failed with
`cudaGetLastError`; the experiment was **removed**, leaving decode-only Graph
unchanged. The failed report is
`smollm-jittor-full2-gpu6.json` in the same state directory. No speedup or
correctness claim is made for FULL graph.

Three additional local models were downloaded from the HF mirror as explicit
model files (no ONNX artifacts): SmolLM2-360M-Instruct,
Qwen2.5-0.5B-Instruct and TinyLlama-1.1B-Chat-v1.0. They were compared on
physical GPU6, TP1, FP16, 128 prompt + 16 generated tokens, decode-only CUDA
Graph, 1 warmup + 3 measurements. Each measured request emitted exactly the
same 16 greedy token IDs as native PyTorch:

| Model | Jittor request / tok/s | PyTorch request / tok/s | Jittor/PyTorch throughput |
| --- | ---: | ---: | ---: |
| SmolLM2-360M | 73.484 ms / 217.3 | 43.876 ms / 368.2 | 59.0% |
| Qwen2.5-0.5B | 84.240 ms / 189.6 | 38.941 ms / 404.1 | 46.9% |
| TinyLlama-1.1B | 86.253 ms / 184.8 | 56.325 ms / 284.1 | 65.1% |

Qwen2.5 uses vLLM's V2 model runner and initially failed at the missing
`weak_ref_tensor` graph alias. A guarded compatibility implementation now
allows it only with `JITTOR_VLLM_EXPERIMENTAL_STRONG_GRAPH_ALIAS=1`; this is an
owning identity rather than vLLM's non-owning alias, so longer graph-lifetime
testing is needed before enabling it by default. TinyLlama's first Jittor
attempt failed during vLLM memory profiling because another process freed
GPU memory on the shared card; the successful rerun above did not reproduce
that environment failure. An accidental PyTorch eager run is excluded from
the same-Graph table.

For Qwen3-0.6B prefill, the cuDNN FP16 GQA fused candidate is registered but
declines the actual `(Q heads=16, KV heads=8, sequence=128, head_dim=128)`
shape with `No execution plans support the graph`. The loaded cuDNN runtime
version is **8.9.7**, so Jittor falls back to composite attention. An
CUDA full-prefill path casts fresh Q/K/V to FP32,
uses Jittor's tiled fused GQA kernel, then casts the output back to the model
dtype. The isolated 128-token call measured 0.389 ms composite FP16 versus
0.046 ms fused FP32 on GPU2; max absolute output difference was 0.00141.
This micro result is not an end-to-end estimate.

In two candidate / one baseline fresh-process runs on GPU6 (128 prompt + 32
output, Graph decode, 1 warmup + 3 measurements), the candidate's median
TTFT was 20.113 and 20.024 ms; baseline was 28.961 ms. Median request was
119.993 and 119.775 ms versus baseline 132.672 ms, approximately 10% faster.
All measured 32-token outputs matched the native PyTorch oracle exactly.
After the validation below, this path defaults to `auto` only for FP16 full
prefill of 2–512 tokens with head dimension at most 128. Setting
`JITTOR_VLLM_F32_FUSED_PREFILL=0` restores the original path; `=1` extends
the experiment to BF16 or longer sequences. Reports are `qwen3-{f32prefill-jittor-gpu6,
baseline-jittor-gpu6,f32prefill-jittor-gpu6-r2}.json` in the state directory.

The same opt-in prefill path improved the other three models under the
128+16 protocol, with all measured greedy token IDs unchanged:

| Model | Jittor baseline request / TTFT | Fused-FP32 candidate request / TTFT |
| --- | ---: | ---: |
| SmolLM2-360M | 73.484 / 31.865 ms | 65.994 / 23.049 ms |
| Qwen2.5-0.5B | 84.240 / 25.931 ms | 75.730 / 18.007 ms |
| TinyLlama-1.1B | 86.253 / 23.533 ms | 80.186 / 17.511 ms |

Their candidate reports are `smollm-f32prefill-jittor-gpu6.json`,
`qwen25-f32prefill-jittor-gpu6.json`, and
`tinyllama-f32prefill-jittor-gpu6.json`. Qwen2.5 still requires the separate
experimental strong alias switch. The candidate only changes full-prefill
attention; ordinary decode attention, dynamic batch and NPU paths are not
covered by these results.

The harness now accepts `--prompt-variant science|math|code` (default science
preserves the previous prompt) to avoid validating only one repeated sentence.
A later dtype guard briefly regressed the candidate: standalone Jittor printed
`float16`, while the vLLM Torch-shim worker printed `torch.float16`. A trace
showed the experimental path was *not selected* in the first math 64-token
run. The guard now normalizes the optional `torch.` prefix. On GPU2, math
128+64 with the effective old path measured 30.458 ms TTFT and 234.678 ms
request; the corrected candidate measured 19.866 ms and 222.060 ms, with all
64 token IDs identical to the native PyTorch oracle. A separate code prompt
at 32+64 measured 29.549/226.048 ms old versus 20.699/218.887 ms candidate,
again with all 64 tokens matching the oracle. The native code-prompt request
was 158.377 ms, or 404.23 tok/s; the candidate reached 294.38 tok/s
(72.8% of native output throughput). These timings exclude engine loading.
The failed GPU6 diagnostic run terminated before inference because another
process released 1.65 GiB while vLLM profiled memory; it is not used for
performance conclusions.

Further GPU2 checks with the corrected guard and exact greedy output:

| Prompt / length | Old Jittor TTFT / request | Fused candidate TTFT / request | Native PyTorch request |
| --- | ---: | ---: | ---: |
| Math, 128+64 | 30.458 / 234.678 ms | 19.866 / 222.060 ms | Same 64-token oracle used for correctness |
| Code, 32+64 | 29.549 / 226.048 ms | 20.699 / 218.887 ms | 158.377 ms |
| Science, 512+16 | 31.477 / 89.497 ms | 20.303 / 78.884 ms | 48.917 ms |

The 512-token result increases Jittor output throughput from 178.38 to
203.45 tok/s, but this is still only 62.1% of PyTorch's 327.68 tok/s.
Code-prompt throughput is 294.38 versus PyTorch 404.23 tok/s (72.8%).
Enabling the default `auto` policy without an environment override was also
checked on Qwen3 science 128+16: 21.209 ms TTFT, 70.120 ms request, 16
tokens matching the PyTorch oracle. The policy unit suite passed 11 tests.

An asynchronous phase trace on the same GPU is diagnostic only: it adds
substantial overhead (Jittor traced request 125.423 ms versus its untraced
70.120 ms), so its wall times must not be interpreted as production speed.
Median traced CPU submission per step was 0.342 ms for input preparation,
0.132 ms for attention metadata, and 0.911 ms for sampling, compared with
PyTorch's 0.210, 0.045, and 0.537 ms. The trace records model forward only
for prefill, not CUDA Graph decode, so it cannot apportion the remaining
decode GPU gap. Another low-overhead GPU timeline is needed before attributing
the decode deficit to a specific kernel.

Qwen2.5-0.5B on GPU2 with a math prompt, 128 input + 64 output tokens,
shows why prefill alone is insufficient. With the default short-FP16
prefill optimization, Jittor's median TTFT/request was 17.415/257.914 ms
(248.25 tok/s), versus PyTorch 9.612/127.547 ms (501.75 tok/s). All
64 greedy token IDs matched for each of the three measured repetitions.
Qwen2.5 still requires the experimental strong graph alias switch; this
is a correctness/performance probe, not a production-stability claim.

An A/B trial of 16 instead of 8 warps in the CUDA paged-decode kernels
kept the Qwen2.5 64-token outputs identical, but did not show a stable speed
gain: 16-warp median request 254.311 ms; repeated 8-warp median 251.742 ms
on the same GPU. The experimental kernel change was reverted; the default
remains 8 warps.

## Decode attribution and host-copy experiment

Matched Qwen2.5 science/maths decode traces changed the diagnosis. CUPTI
reported Jittor GPU busy 30.651 ms over one 128+16 request, including
18.899 ms across 736 calls of its dominant cuBLAS GEMV kernel and 2.723 ms
across 360 fused paged-decode kernels. Native PyTorch invoked the **same**
GEMV kernel 736 times for 18.927 ms. The main decode discrepancy is thus
not a slower GEMV calculation. A separate low-overhead host trace of the
same model/protocol found median per-step timings:

| Stage | Jittor | PyTorch |
| --- | ---: | ---: |
| `_prepare_inputs` | 1.430 ms | 0.330 ms |
| `sample_tokens` | 1.716 ms | 0.165 ms |
| Graph replay host call | 0.007 ms | 0.012 ms |

These are nested host-stage timings, not values to add indiscriminately to
request wall time. The trace itself slightly perturbs timing. The profiler
shows that V2 sampling constructs an output event every step and Jittor's
compatibility `Event.record()` blocks on `sync_all`; `_prepare_inputs`
performs many small fixed-buffer copies. Removing the event wait without a
real completion fence changed the fourth generated token and was reverted.
A real CUDA event preserved all 64 greedy token IDs in three requests but
did not improve request time (260.094 ms versus 251–258 ms baseline), so that
more complex experiment was also reverted. The saved failed/correct reports
are `qwen25-async-event-math64-jittor-gpu2.json` and
`qwen25-nativeevent-math64-jittor-gpu2.json`.

The opt-in `JITTOR_TORCH_DIRECT_HOST_COPY=1` path in Torch `Tensor.copy_`
instead sends a CPU source directly into a fixed-address CUDA destination
when dtype, shape, contiguity and storage ownership permit. It avoids making
an intermediate GPU Var and a second D2D copy, excludes graph capture and
ACL, and falls back to the old path otherwise. A two-update probe preserved
the destination address and expected values. For Qwen2.5 math 128+64 on GPU2,
all three 64-token requests matched native PyTorch and shutdown was clean.
Median request was 239.984 ms in the first candidate run, 260.452 ms in the
interleaved original-path run, and 245.673 ms in the repeated candidate run.
This is a reproducible roughly 6% request improvement, not enough to close
the approximately 2x throughput gap. Reports have `qwen25-directhostcopy-`
prefix in the state directory. The path remains opt-in pending multi-model
and multi-shape validation.

The same direct-host-copy setting also completed Qwen3-0.6B math 128+64 on
GPU2 with all 64 tokens equal to its PyTorch oracle in each measured request
and a clean shutdown. Its median request was 223.136 ms, close to the
earlier non-copy-path 222.060 ms, so no Qwen3 speed benefit is claimed.
SmolLM2-360M and TinyLlama-1.1B also each completed three 128+16 requests
with all greedy tokens equal to their respective PyTorch oracles; their
medians were 62.280 and 79.343 ms on GPU2. The new CUDA Graph regression
test verifies two CPU-source updates keep the CUDA graph input pointer fixed
and each replay observes the new data (`test_direct_host_copy.py`: 1 pass).
Fixed batch=4 Qwen2.5 math 128+16 with Graph shape 4 also completed three
measured rounds: all 12 request outputs (16 tokens each) exactly matched
native PyTorch. Jittor's median batch request was 86.277 ms / 741.3 tok/s;
PyTorch's was 40.866 ms / 1564.7 tok/s. Correctness at batch=4 is not a
claim of stable dynamic-batch shape switching or multi-GPU support.
An interleaved Jittor original-path batch=4 run measured 88.734 ms / 721.5
tok/s, versus the direct-host-copy 86.277 ms / 741.3 tok/s, a smaller
approximately 2.7% request-time gain under a quieter load (~46/128 CPUs).
During these latter checks, `cscg-qh00` had load average around 262 on 128
logical CPUs due to other users' jobs. The alternating Qwen2.5 improvement
is consistent, but sub-10% timing claims need a quiet-machine repeat.

An opt-in direct-host-copy TP=2 Qwen3-0.6B check on physical GPUs 4 and 5
did generate three 128+16 requests; every 16-token greedy sequence matched
the TP=1 native PyTorch oracle. Median request time was 67.779 ms (236.1
output tok/s per round). Its isolated rank caches triggered first-time
Jittor-core and NCCL builds, and initialization took 383.60 seconds under
NFS stalls; these are excluded from request latency. vLLM then force-terminated
both worker processes after the shutdown grace period (exit -15), although
its shared-memory segments were removed. The harness marks this run **failed**
because clean shutdown is required. At that stage, TP=2 numerical correctness
was observed but lifecycle stability was not yet established. Raw report:
`qwen3-tp2-direct-copy-gpu45-retry-20261013.json`.

The same opt-in copy path was then tested on one GPU with **one engine**
capturing shapes 1 and 4, switching batches `1→4→1→4`, one warmup and two
measured rounds at each stop. Both Jittor and native PyTorch completed and
shut down cleanly. All 20 measured request outputs (16 tokens each) matched
the corresponding PyTorch output exactly. There was no illegal address in
this run, but a single successful sequence is not proof that the old
intermittent shape-switch bug is gone. On GPU2 the latter batch=1/4 median
request times were Jittor 65.837/84.856 ms and PyTorch 46.680/51.100 ms;
the first batch=1 Jittor pair was noisy (median 86.838 ms). Reports:
`qwen3-{directcopy,oracle}-shapes1414-gpu2.json`. The copy path remains
opt-in pending longer stress and TP=2 clean-exit verification.

Low-overhead Qwen3 host trace on GPU2 (128+16) showed only a small direct-copy
effect: median `prepare_inputs` 0.368→0.314 ms/step, while median full request
was 68.175→68.148 ms and `sample_tokens` stayed about 2.26 ms/step. A
Qwen2.5 V2-runner paired trace showed a larger but still limited improvement:
`_prepare_inputs` 1.472→1.290 ms/step, request 75.095→71.737 ms, with
`sample_tokens` nearly unchanged at 1.71 ms/step. These traced timings are
diagnostic, not an isolated quiet-machine throughput claim. One earlier
Qwen2.5 direct-copy + trace run segfaulted after its warmup; a baseline run
and repeated direct-copy run then completed normally, so its cause is not
isolated. This intermittent failure is another reason to keep the copy and
Qwen2.5 strong-graph-alias paths experimental rather than default.

A longer Qwen3 same-engine shape-switch stress used `1,4` repeated five times
(nine transitions). It completed 25 warmup and 75 measured requests with clean
shutdown. Every measured 16-token greedy output matched the corresponding
native PyTorch batch=1 or batch=4 oracle (75/75). This raises confidence in
the tested direct-copy + dual-graph path, but cannot prove that the historical
intermittent illegal address is impossible for other shapes or lifetimes.
Raw report: `qwen3-directcopy-shapes-stress-gpu2.json`.

## TP=2 CUDA Graph communicator lifetime fix

A warmed-cache TP=2 A/B on the same physical GPUs 4/5 showed that the
direct-host-copy experiment did not solve the multi-card lifecycle issue:
Jittor's 128+16 median request was 68.177 ms with the copy path versus
68.734 ms without it, while native PyTorch measured 45.271 ms. The native
PyTorch worker exit codes were 0; both Jittor variants were force-terminated
with -15. Increasing the Jittor shutdown grace to 30 seconds did not help.
These are output-only inference times and exclude initialization.

Instrumenting worker teardown showed that Python worker, model-parallel and
distributed cleanup each returned in under 0.2 seconds. Native stacks for
both ranks then showed the Jittor NCCL static finalizer waiting inside
`ncclCommDestroy`. Per-group tracing identified group 2, the first two-rank
TP communicator, as the blocking point. TP=2 eager inference destroyed every
group and exited cleanly. Calling native NCCL shutdown earlier or replacing
destroy with abort also blocked, whereas explicitly resetting the captured
CUDA Graph before normal teardown allowed `ncclCommDestroy` to complete.
This establishes a graph/communicator lifetime ordering problem, not a slow
NCCL computation. Temporary NCCL and harness tracing/abort experiments were
reverted.

The retained fix is `patch_v2_model_runner_graph_teardown` in
`adapters/jittor_adapters/vllm/buffers.py`: on Jittor V2 runner shutdown it
syncs, resets each captured graph executable once, clears shape-specific
keepalives, then calls vLLM's original shutdown. Its idempotency/order unit
test passes. After reverting all temporary native NCCL changes, TP=2 Qwen3
decode-only Graph completed three 128+16 requests with 16/16 token IDs per
request matching the PyTorch TP=2 oracle, and both workers exited normally.
Median request was Jittor 67.516 ms (236.98 tok/s) versus PyTorch 45.271 ms
(353.69 tok/s), or 67.0% of PyTorch output throughput. This fixes teardown,
not the remaining performance gap. Reports:
`qwen3-tp2-finalgraphfix-gpu45.json` and `qwen3-tp2-oracle-gpu45.json`.

The fixed batch=4 TP=2 Graph case also exited normally. Three measured
128+16 requests (four sequences each) matched native PyTorch for all 64
generated IDs per request. Median request latency was 100.437 ms for Jittor
versus 56.212 ms for PyTorch (55.97% of PyTorch output throughput). The
same-engine TP=2 shape sequence `1,4,1,4` then captured both shapes, completed
one warmup and two measured requests per shape occurrence, matched native
PyTorch for all eight measured requests, and exited cleanly. The two-shape
Graph teardown unit test now checks that distinct executables are reset before
the original runner shutdown, in addition to the alias/idempotency case.
These checks cover the specified Qwen3/FP16/128+16/greedy path; they do not
prove all dynamic-batch and multi-model cases safe. Reports:
`qwen3-tp2-b4-finalgraphfix-gpu45.json`,
`qwen3-tp2-b4-oracle-gpu45.json`,
`qwen3-tp2-b1-4-1-4-finalgraphfix-gpu45.json`, and
`qwen3-tp2-b1-4-1-4-oracle-gpu45.json`.

## TP=2 sampling/readback attribution and rejected async experiment

An unsynchronized host-stage trace at batch=4, TP=2, 128+16 showed per-rank
steady decode medians (last 30 calls): Jittor `execute_model` 1.1–1.25 ms,
`prepare_inputs` 0.62–0.65 ms, `sample_tokens` 2.46–2.60 ms, and nested
`AsyncOutput.__init__` 1.64–1.94 ms. Native PyTorch on the same GPUs measured
0.49–0.50, 0.21, 0.51, and 0.05 ms, respectively. These are overlapping host
spans, not additive GPU times. In particular, Jittor's synchronous token
readback can be waiting for outstanding GPU work, so the 1.6–1.9 ms cannot
all be attributed to memory-copy overhead. Reports:
`qwen3-tp2-b4-lighttrace-gpu45.json` and
`qwen3-tp2-b4-oracle-lighttrace-gpu45.json`.

A standalone 100-iteration pinned CUDA D2H side-stream prototype returned
correct values with 0.011 ms median enqueue time, but wiring that mechanism
into vLLM did not improve whole-request latency. On a single GPU, batch=4,
two measured requests took 98.84/99.65 ms with the opt-in async readback
versus 91.41/90.70 ms with the existing synchronous path; all generated IDs
matched. The server later became heavily loaded and an additional trace was
aborted while blocked in NFS I/O, so this small A/B is not a definitive
performance ranking. Because it gave no demonstrated end-to-end win and adds
stream/lifetime complexity, the adapter experiment was removed; the prototype
is retained only in diagnostic state, not the serving code. Reports:
`qwen3-tp1-b4-asyncd2h-r3-gpu4.json` and
`qwen3-tp1-b4-baseline-gpu4.json`.

## Experimental batch=4 native metadata kernels

The V2 Qwen3 batch=4 `prepare_inputs` cProfile showed 66 Triton bridge
dispatches in 32 calls: `prepare_pos_seq_lens` and
`combine_sampled_and_draft_tokens` each took the original Triton path because
the existing native CUDA metadata implementation was gated to batch=1. The
batch=4 fused block-table/slot kernel was already enabled. The diagnostic
profiler was corrected to profile only methods present on the active V2/legacy
runner, avoiding an empty `pstats.Stats` failure.

`JITTOR_VLLM_NATIVE_METADATA_BATCH4=1` now opts batch=4 into the existing
per-request native CUDA position and token-combine kernels. Other batch sizes
and the default path are unchanged. With 128 prompt + 16 greedy generated
tokens, FP16, decode-only CUDA Graph, and the same physical GPU within each
paired test:

| Model / TP | Baseline median request | Candidate median request | Result |
| --- | ---: | ---: | --- |
| Qwen3-0.6B / TP1 | 90.09 ms (3 runs) | 87.69 ms (3-run second A/B) | 2.7% faster |
| Qwen3-0.6B / TP2 | 92.50 ms (3 runs) | 89.02 ms (3 runs) | 3.8% faster |
| SmolLM2-360M-Instruct / TP1 | 88.37 ms (2 runs) | 81.40 ms (2 runs) | 7.9% faster |
| TinyLlama-1.1B-Chat / TP1 | 94.85 ms (2 runs) | 97.75 ms (2 runs) | 3.1% slower |
| Qwen2.5-0.5B-Instruct / TP1 | 89.84 ms (2 runs) | 90.54 ms (2 runs) | within noise; strong Graph alias also required |

Every candidate request in this table produced the same 16 token IDs per
sequence as its baseline and, for all four models, the native PyTorch oracle.
The TP2 candidate also passed a same-engine `1,4,1,4` graph switch: eight
measured requests were exact versus PyTorch, and both workers exited cleanly.
The first TP2 baseline recheck was invalidated during vLLM initialization when
another process released GPU memory; the successful immediate retry above is
the paired comparison. Model-specific regressions and the shared server's
variable load are why the switch remains opt-in, not a default optimization.
Raw reports have `native-meta` in their filenames under the same state
directory as the other comparisons.

A longer-output Qwen3 check on the same physical GPU 6 tested whether this
batch-4 metadata path benefits repeated decode, not just first-token work.
With FP16, 128 input + 64 greedy output tokens, batch=4, TP1, decode-only
Graph, one warmup and two measured requests per process, the interleaved
baseline/candidate pairs were 266.832/250.933 ms and 267.413/251.525 ms
median full batch time. The candidate therefore saved about 15.9 ms, or
5.9–6.0%, in both pairs. All eight 64-token sequences per variant exactly
matched each other and native PyTorch. PyTorch's same-GPU median was
171.370 ms (1493.8 aggregate tok/s); candidate Jittor reached 1017.8–1020.2
tok/s, about 68.2% of that throughput, versus 957.3–959.4 tok/s baseline.
The switch remains opt-in because the earlier TinyLlama and Qwen2.5 cases
did not benefit. Reports:
`qwen3-tp1-b4-math64-native-meta-{baseline,candidate}-gpu6-{r2,r3}-20261013.json`
and `qwen3-tp1-b4-math64-native-meta-oracle-gpu6-20261013.json`.

The same 128+64, batch=4, TP1, FP16, decode-only Graph protocol was run on
GPU6 for the other three local models (one warmup, two measured requests per
process). All candidate outputs matched native PyTorch token-for-token:

| Model | Baseline | Native-metadata candidate | PyTorch | Candidate / PyTorch throughput |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-360M | 248.84 ms | 234.29 ms | 139.85 ms | 59.7% |
| TinyLlama-1.1B | 324.86 ms | 307.45 ms | 223.22 ms | 72.6% |
| Qwen2.5-0.5B | 280.76 ms | 284.56 ms | 139.96 ms | 49.2% |

The first two gain about 5–6%, while Qwen2.5 regresses about 1.4%; these
are small, shared-server A/B samples, so the switch stays opt-in. Qwen2.5
also needs `JITTOR_VLLM_EXPERIMENTAL_STRONG_GRAPH_ALIAS=1` to pass its V2
Graph path, so its result is not a production-default configuration. Raw
reports are `smollm2-`, `tinyllama-` and `qwen25-tp1-b4-math64-native-meta-*`
in the same state directory.

A matched physical GPU2 check of Qwen3-0.6B at TP1, batch=1, FP16,
128 input + 64 output, decode-only Graph, one warmup and three measured
requests per process found:

| Batch-1 path | Median request | Output throughput | Median TTFT | Median ITL |
| --- | ---: | ---: | ---: | ---: |
| Jittor, native metadata off | 236.59 ms | 270.1 tok/s | 20.77 ms | 3.36 ms |
| Jittor, native metadata on | 220.87 ms | 289.8 tok/s | 19.62 ms | 3.16 ms |
| Native PyTorch | 157.55 ms | 405.2 tok/s | 12.07 ms | 2.30 ms |

The current default batch-1 native metadata path improved Jittor throughput
6.6% in this A/B but remained at 71.5% of PyTorch. Every one of the three
64-token sequences in both Jittor variants exactly matched its PyTorch
counterpart. A prior adjacent GPU6 A/B found 237.11 ms off versus 224.95 ms
on (5.1% faster); its PyTorch run landed on another physical card and is not
used for the throughput ratio. Reports have
`qwen3-tp1-b1-math64-native-meta-{baseline,candidate,oracle}-gpu2-20261013.json`.

A light step trace on the same GPU2 and batch-1 128+64 workload retained
similar request times (Jittor 222.04 ms, PyTorch 158.42 ms). Median recorded
decode scopes, in milliseconds, were:

| Scope | Jittor | PyTorch |
| --- | ---: | ---: |
| `execute_model` | 0.830 | 0.409 |
| `prepare_inputs` | 0.356 | 0.215 |
| `prepare_attn` | 0.126 | 0.045 |
| `sample_tokens` | 2.242 | 0.395 |
| `AsyncOutput.__init__` (within sampling) | 1.657 | 0.054 |
| CUDA Graph replay submission | 0.009 | 0.012 |

These are host-observed scopes, not independent additive costs: Jittor's
sampled-token `numpy()` synchronizes and can absorb outstanding GPU time,
whereas PyTorch's pinned async output copy returns before completion. The
trace therefore localizes the synchronization boundary but does not prove
that all 1.6 ms is removable CPU overhead. Reports are
`qwen3-tp1-b1-math64-lighttrace-{jittor,oracle}-gpu2-20261013.json`.

## Request-level CUDA timeline and further rejected experiments

One profiled Qwen3/FP16/TP1/batch=4 128+16 request on physical GPU 4
recorded 99.50 ms Jittor wall time, 45.23 ms GPU busy time, and a 97.63 ms
span from first to last kernel. The Jittor profiler attributed 45.47 ms to
Python/graph construction, 33.36 ms to device waits, 14.88 ms to operator
launch, and 4.03 ms to Graph launch (host categories need not be additive).
The native PyTorch CUDA profiler on the same GPU recorded 59.10 ms wall time,
44.09 ms CUDA-kernel busy time, and a 57.28 ms kernel span. These are separate
profiler implementations and single samples, so not a controlled microsecond
ranking of kernels; nevertheless the almost equal aggregate kernel times and
roughly 40 ms longer Jittor kernel span support a host submission/idle-gap
problem, not a model GEMM throughput problem. Raw reports:
`qwen3-tp1-b4-deviceprofile-jittor2-gpu4.json` and
`qwen3-tp1-b4-deviceprofile-oracle3-gpu4.json`.

Two opt-in transfer experiments were removed from the serving tree after A/B:

- A same-stream pinned asynchronous sampled-token D2H readback gave TP1
  batch=4 times 88.58/88.43/87.84 ms and 93.33/89.33/88.81 ms in two runs,
  versus adjacent synchronous baseline 92.22/90.13/89.34 ms. The small TP1
  gain did not generalize: TP2 same-engine `1,4,1,4`, with native batch-4
  metadata in both variants, was generally 2–5 ms slower than the existing
  readback, although all eight requests exactly matched the PyTorch oracle
  and shutdown was clean. This remains a rejected experiment, not a default.
- A pinned direct NumPy H2D upload for the persistent query-start buffer
  reduced wrapper work locally, but Qwen3 batch=4 end-to-end was 89.50/89.14
  ms, indistinguishable from nearby baseline, and it bypassed Jittor's
  dependency tracker. It was removed.

An opt-in native `post_update` kernel was also tried for greedy sampling's
request-state update. cProfile reduced that function from about 13–14 ms to
about 2 ms over 32 calls, but whole-request TP1 batch=4 was only
89.64/88.66/87.63 ms versus nearby baseline 92.22/90.13/89.34 ms. TP2
`1,4,1,4` showed no convincing improvement and occasional slower batch-4
requests; all tokens matched the oracle and shutdown was clean. Because the
local CPU gain did not transfer robustly to end-to-end latency, this native
path was removed. One intermediate profile failed during vLLM memory sizing
when another GPU process released memory; the successful retry, not that
failed run, informs the comparison. Raw reports have `asyncsame`, `asynch2d`,
or `nativepost` in their filenames.

## Follow-up: single-card CUDA gaps and second NPU model

On the same physical CUDA GPU 2, Qwen3-0.6B FP16 batch=1 128+64 decode
profiling recorded Jittor 255.77 ms and PyTorch 177.33 ms with instrumentation.
GPU busy time was 136.07 ms versus 148.93 ms, while the first-to-last-kernel
span was 254.96 ms versus 175.89 ms. The implied idle intervals were
118.89 ms versus 26.96 ms. Thus the remaining request gap is principally
*between* kernels, not Jittor GEMV compute. The unprofiled same-GPU median
was 220.87 ms Jittor versus 157.55 ms PyTorch; profiler overhead must not be
treated as production latency. Reports:
`qwen3-tp1-b1-math64-lighttrace-{jittor,oracle}-gpu2-deviceprof-20261013.json`.

Host cProfile of 130 `execute_model` calls (warmup and measured) attributed
69 ms to 193 `torch.cuda.synchronize()` calls below `copy_to_uva`. Removing
that sync under an opt-in experiment gave TP1 batch=1 128+64 median
215.05 ms versus adjacent 220.48 ms with sync, with all 64 token IDs equal.
But batch=4 128+16 was 85.59 ms without sync versus 84.52 ms with it, again
with exact IDs. The source is a mutable NumPy view, so the optimization is
**not enabled by default** until lifetime and dynamic-shape tests establish
safety and a reproducible win. A separate tiny-index cache regressed the
batch-1 median from 220.48 to 222.48 ms and was removed. The old indirect
`to('cpu').numpy()` readback took 226.84 ms in the same batch-1 workload,
supporting the current direct `numpy()` default. Reports carry `nosyncuva`,
`uvasync`, `smallcache` or `indirectd2h` in the filename.
An isolated CUDA source-lifetime probe also constructed 256 device tensors
from NumPy arrays and immediately overwrote the NumPy sources; all 256 device
snapshots retained the original values after synchronization. This rules out
that simple source-reuse failure, but not all dynamic-batch lifecycle cases.

On Ascend 910B3, Qwen2.5-3B-Instruct BF16 also loaded and generated through
the Jittor vLLM 0.24 adapter. TP1 produced 16 tokens matching native
PyTorch-NPU Hugging Face greedy inference exactly (vLLM request 9.14 s;
the Hugging Face loop 1.62 s is a different serving path, not a comparable
vLLM benchmark). TP2 generated 16 tokens in 15.21 s and exited gracefully,
but differed from TP1/reference starting at token 9. That alternate token
was exactly the reference's second choice, only 0.25 logits below first;
this is consistent with a BF16 near-tie under TP reduction and is not yet
an exact-correctness pass for the second model. Qwen3-0.6B remains the model
with exact 64-token TP1 and TP2 checks. Setting the TP2 worker shutdown grace
to 15 seconds let both workers exit gracefully in a separate four-token test.

An ACL diagnostic reread every second attention layer, instead of every
layer, reproduced all Qwen3 oracle IDs for 16-token TP1 and 64-token TP1/TP2
requests. It took 12.22 s TP1 versus prior 12.65 s, but 17.82 s TP2 versus
prior 17.35 s, so it brought no consistent speedup. The stride experiment
was removed; the validated per-layer readback remains default. Three host
metadata policy tests pass on CUDA/fake ACL. The full 14-case attention test
file on real ACL had one pre-existing mapped-matmul float32/float16 dtype
error, outside those metadata tests; it is not claimed as fully green.

The ACL readback itself was then optimized without reducing the number of
correctness fences: an actual `query_start_loc.numpy()` at each later layer
still flushes the needed work, while avoiding the extra lazy CPU Var from
`query_start_loc.cpu().numpy()`. Qwen3 TP1 and TP2 short-prompt 64-token
requests remained exact versus the PyTorch oracle and took 11.25 s and
16.31 s, compared with prior 12.65 s and 17.35 s. A 130-token input plus
32-token output also remained exact across a KV page boundary. On the same
NPU and in one engine, the second/third requests with that prompt took
3.070/3.052 s using the old readback and 2.669/2.676 s using direct readback:
roughly 12.7% faster after warmup. The first request is dominated by
compilation and is excluded from that warm comparison. Direct readback is
now the ACL default; setting `JITTOR_VLLM_NPU_DIRECT_METADATA_READBACK=0`
restores the old path. Host metadata policy tests pass (3 cases), and default
configuration hardware regression plus TP4 remain pending at this note's
time of writing.

### ACL automatic replay isolation (later validation)

The per-layer metadata readback proved to be a workaround for Jittor's
automatic module graph replay, not an inherent requirement of ACL attention.
With `JT_AUTO_GRAPH_REPLAY=0` and
`JITTOR_VLLM_NPU_SAFE_METADATA_READBACK=0`, Qwen3-0.6B BF16 passed the native
PyTorch token-ID oracle for TP1, TP2, and TP4 with 64 generated tokens; all
workers exited gracefully. It also passed a 130-token prompt plus 32 output
tokens across the 128-token KV page boundary and four concurrent TP1 requests
of 16 tokens each. These isolate auto replay as the likely source of the stale
dynamic-KV behavior, but do not identify the exact graph-alias defect.

The opt-in NPU platform now defaults `JT_AUTO_GRAPH_REPLAY=0` and omits the
per-layer host readback in that mode. Setting `JT_AUTO_GRAPH_REPLAY=1` keeps
the prior readback safeguard by default. The qh CUDA path is unchanged. Five
host-metadata policy tests pass; default-path hardware regression is in
progress. First-request NPU times include significant JIT and are not used as
steady-state throughput comparisons.

The fourth CUDA cross-model Graph trial, Qwen2.5-0.5B, requires vLLM's
`weak_ref_tensor` operation. Jittor's owning identity substitute remains
explicitly opt-in with `JITTOR_VLLM_EXPERIMENTAL_STRONG_GRAPH_ALIAS=1` because
its lifetime semantics differ. Under that switch, TP1 batch1 128+64 ran
three measured rounds on GPU2, all 64 IDs matching PyTorch each round;
median Jittor request was 266.1 ms / 242.9 tok/s versus PyTorch 128.8 ms /
496.9 tok/s. The 48.9% throughput ratio is another example of host-side
Graph integration still lagging despite correct outputs. This fixed-shape
trial is insufficient to enable the strong alias by default.

The default NPU no-auto-replay configuration subsequently passed Qwen3 TP1
and TP2 64-token probes and a TP4 16-token probe, all with graceful worker
shutdown. In one TP1 engine, batch sizes `1,4,1,4` each generated 16 exact
oracle tokens per request and exited normally; request times were
9.730/6.062/1.343/2.953 s respectively (the first includes JIT). A repeated
130-token prompt plus 32-token output test also matched the PyTorch oracle
and took 9.197/2.666/2.663 s in one engine. Compared with the previous
direct-readback safeguard's warmed 2.669/2.676 s, disabling auto replay is
primarily a correctness/lifetime simplification, **not a demonstrated
throughput improvement**. The NPU path remains eager and materially slower
than desired; these NPU seconds must not be compared with CUDA seconds as a
framework speed ratio.

The CUDA Qwen2.5 strong-alias experiment also completed a same-engine
batch-size sequence `1,4,1,4`, each with 128 input and 16 output tokens.
All ten request outputs (1+4+1+4) exactly matched the native PyTorch CUDA
vLLM outputs. Jittor request times were 74.77/88.26/72.87/89.26 ms versus
PyTorch 40.17/42.46/38.94/40.18 ms on the same GPU2. This tests four shape
transitions without the earlier illegal address, but a short successful
sequence does not settle long-run lifetime safety. The strong alias remains
opt-in and the speed ratio is still poor.

### NPU host profiling and independent environment audit

The isolated Jittor-NPU vLLM environment is
`/beegfs/home/tuser028/venvs/vllm-jittor-npu-20261013` (Python 3.11,
vLLM 0.24). The pre-existing pure Jittor environment is
`/beegfs/home/tuser028/qwen35-jittor/.venv` (Python 3.9), and the
pre-existing pure PyTorch-NPU environment is
`/beegfs/home/tuser028/venvs/torch-npu-bench` (Python 3.9,
torch/torch-npu 2.7.1). The host shell has no `conda` executable; those two
pure environments were left untouched. The Jittor-NPU vLLM package is
already installed in its separate environment, not in either pure environment.

An optional host-only trace of warmed Qwen3-0.6B TP1, batch 1, 16-output-token
steps (without cProfile) measured median `execute_model` 51.14 ms,
`model_forward` 44.50 ms, `_prepare_inputs` 4.70 ms, and `sample_tokens`
30.73 ms. These nested intervals must not be added. cProfile on a later
warmed run found nearly all sampling host time in the compatibility event's
`jt.sync_all(True)` during vLLM's `_to_list` output transfer: 1.024 s in 32
calls, about 32 ms per call. This is primarily a forced execution/drain of
lazy model work, not 32 ms of argmax CPU computation; cProfile itself raised
model-forward host timings and is not used for throughput reporting.

A separate native vLLM-Ascend comparison venv was created at
`/beegfs/home/tuser028/venvs/vllm-ascend-oracle-20261013`, without changing
the pure environments. A verified PyPI source archive was built for
`vllm==0.20.2` with `VLLM_TARGET_DEVICE=empty`; `vllm-ascend==0.20.2rc1`,
Torch/Torch-NPU 2.10, and user-local CANN 9.0 NNAL ATB were installed. The
published wheel metadata still conflicts (`vllm` advertises Torch 2.11 while
the Ascend plugin advertises Torch 2.10), so this remains an experimental
comparison environment. Nevertheless, an actual native TP2 Qwen2.5 BF16
vLLM-Ascend engine now loads and generates. It is a useful correctness oracle,
but not yet a fair performance oracle against the Jittor vLLM 0.24 adapter:
versions, compilation state and request protocol differ. No system-level
CANN or pure Python environment was changed.

### Qwen2.5 TP2 near-tie investigation

The earlier Qwen2.5-3B-Instruct BF16 TP2 token-9 mismatch was against an
**unsharded TP1** native PyTorch-NPU Hugging Face oracle. The first eight token
IDs agreed. At token 9, native TP1 chose ID 264 over 13482 by 0.25 logit;
Jittor TP1 reported the two tokens tied and chose 264, whereas Jittor TP2
chose 13482 over 264 by 0.125. An opt-in FP32 HCCL-reduce experiment did not
change the TP2 token or reported candidate gap and was reverted.

To distinguish an adapter bug from low-precision sharding order, a separate
native PyTorch-NPU BF16 diagnostic replaced the row-parallel `o_proj` and
`down_proj` linears with two half-width GEMMs followed by BF16 addition on
one card. This simulation selected 13482 over 264 by 0.125 at token 9 and
produced all 16 token IDs equal to Jittor TP2. However, the stronger test is
the real distributed vLLM-Ascend TP2 engine. It produced the native TP1
sequence twice, selecting ID 264 at token 9; its logprobs for IDs 264 and
13482 were equal to six decimal places (-0.748356). Jittor TP2 reported
-0.818558 for 264 and -0.693558 for 13482, a 0.125 gap. This is a very
close BF16 decision, but the simulation is not evidence that Jittor's TP2
path agrees with actual native vLLM. The numerical source of the gap remains
open; the earlier FP32 HCCL-reduce experiment did not change it.

Earlier default-path Jittor TP2 16-token smokes passed the weaker simulated
oracle with requests of 9.851/2.307/2.504 s, first including JIT. TP1
passed the native PyTorch-NPU Hugging Face oracle. Other TP2 models and
features still require validation.

The reusable `agent/skills/vllm-torch-compat/npu_vllm_smoke.py` now asserts
the actual native vLLM-Ascend TP2 Qwen2.5 IDs. This test is intentionally
red at token 9 until the numerical divergence is understood or fixed. Before
updating that oracle, the versioned test passed three Jittor TP1 requests
(7.960/1.803/1.843 s) and three TP2 requests against the weaker one-card
simulation (10.003/2.229/2.361 s). The multiprocess executor needed a
SIGTERM for straggling workers at shutdown in these runs; clean/instant
worker teardown is not asserted.

An isolated NPU output-readback probe changed vLLM `_to_list` to materialize
only the sampled token tensor. Qwen3 TP1 16-token requests remained exact,
with two warmed requests taking 1.328/1.328 s versus adjacent default
1.366/1.362 s. This is a small roughly 2.5% improvement and is **not** a
default serving change: longer and multi-rank correctness and interleaved
performance remain unproven.

A subsequent same-card Qwen3 TP1 64-token check also passed the exact oracle
for all three requests. With the scoped output readback the two warmed
requests took 5.298/5.344 s; the immediately following default-path run took
5.408/5.642 s. This suggests a small 1–5% gain but has only one A/B pair;
multi-rank, batch switching and further alternating runs remain required
before changing the default.

The follow-up Qwen3 TP2, batch-1, 64-token three-request A/B on visible
NPUs 2,3 remained exact against its native PyTorch-NPU token oracle. Default
readback took 14.661/7.443/7.374 s; scoped readback took
14.592/7.563/7.698 s. The warmed TP2 candidate was slower, so this
readback change is not being promoted for multi-card serving. The first
request in each run includes setup/JIT and is not used for warmed throughput.

### CUDA V2 runner output-readback A/B

Qwen2.5-0.5B uses vLLM 0.24's `AsyncGPUModelRunnerOutput` path, where the
Torch compatibility Event records via `jt.sync_all(True)`. A first opt-in
experiment replaced that path with a sampled-Var-only NumPy readback. It
failed the 128+64 math-prompt repeated-output assertion, proving that the
global drain cannot simply be deleted across Jittor's CUDA Graph/lazy
dependencies; the code was reverted. A second opt-in experiment explicitly
called `jt.sync_all(False)` before the sampled-only readback. All three
64-token requests then matched the baseline, but paired medians were
260.80 ms baseline versus 252.98 ms candidate in one run and 252.41 ms
baseline versus 253.63 ms candidate in the next. The apparent first-run
gain was not reproducible, so that code was also reverted. Neither trial
changed the serving default.

### CUDA Qwen2.5 graph host/device split and rejected follow-up trials

For the same GPU2, model, FP16, 128-input/64-output request, batch 1 and
`FULL_DECODE_ONLY` graph, the profiled PyTorch oracle's GPU busy time was
123.54 ms across a 140.03 ms kernel span. Jittor's busy time was 120.84 ms,
but its kernel span was 300.94 ms. This confirms that the 2x end-to-end gap
is primarily gaps between GPU kernels rather than slower arithmetic. Jittor's
host profile for that request attributed 130.09 ms to Python/graph build,
38.42 ms to operator launch, 110.61 ms to device wait and 13.88 ms to graph
launch; these categories should not be summed as a simple causal model.

An unsynchronised stage trace (with cProfile for attribution, not throughput)
found median decode `_prepare_inputs` 2.616 ms in Jittor versus 0.390 ms in
PyTorch, and `sample_tokens` 1.741 ms versus 0.252 ms. The Jittor input
profile includes 3,072 native `_copy_into` calls over 256 traced steps,
totalling 0.507 s. The sampled-output event's `jt.sync_all(True)` accounted
for 0.418 s over 256 calls. These are nested/overlapping intervals, not
additive request-time components.

Two opt-in CUDA trials were correct for three 64-token Qwen2.5 requests but
did not improve adjacent A/B medians and were reverted: a native CUDA Event
with lazy publish measured 252.21 ms versus a 248.99 ms original-path
median; same-device small D2D copies issued on the compute stream measured
257.47 ms versus an adjacent 255.94 ms original-path median. No new default
CUDA fast path was enabled by either trial. The earlier direct-host-copy
opt-in remains available and correctly returned 244.81 ms median versus an
adjacent 248.99 ms baseline (~1.7% gain in this retake); this is smaller than
the earlier ~6% under a different host load, and still needs longer
multi-model A/B before promotion.

Combining direct host copy with skipping the CUDA-only UVA snapshot sync did
not improve Qwen2.5 128+64 further: 244.91 ms median versus 244.81 ms for
direct host copy alone, with three identical 64-token outputs. A stronger
experiment allocated a pinned CPU result and enqueued an actual asynchronous
sampled-ID D2H copy/event after `jt.sync_all(False)` on Jittor's per-thread
compute stream. It matched the PyTorch 64-token oracle in three repeated
requests, but median request time was 257.57 ms versus adjacent original-path
258.20 ms: no meaningful end-to-end gain. The diagnostic monkeypatch was
reverted; the default output path remains unchanged.

Skipping the same-dtype `Tensor.copy_` cast was likewise not stable: one
three-request Qwen2.5 run measured 250.25 ms median against the next baseline
253.91 ms, but the immediately repeated candidate measured 259.84 ms.
All candidate 64-token IDs were exact; the shim edit was reverted because
the apparent gain was not reproducible.

### Native NPU oracle experiment status

Inside its isolated venv only, PyPI `vllm==0.20.2` was built from its
verified source archive with `VLLM_TARGET_DEVICE=empty`, and the matching
`vllm-ascend==0.20.2rc1` AArch64 plugin wheel was installed. The source
wheel advertises `torch==2.11` while the plugin advertises `torch==2.10`;
the experiment uses Torch/Torch-NPU 2.10. Despite that metadata mismatch,
the native TP2 Qwen2.5 engine completed two consecutive 16-token requests
with identical outputs and logprobs, then shut down normally. Its first
request included Ascend JIT and is not a useful latency comparison. A fair
speed comparison needs matching vLLM versions, warmed steady state, and
matched request protocol; none is asserted here.

A second native TP2 engine explicitly loaded the same Qwen2.5 weights as
`float16` instead of BF16. Its 16 generated IDs matched the Jittor **BF16**
TP2 sequence, including token 13482 at position 9. This contrasts with the
native BF16 TP2 run, where candidates 264 and 13482 tied and 264 was chosen.
Changing dtype can therefore flip this near-tie even within native
vLLM-Ascend. The first FP16 startup failed before weight load with a Torch
OpenMP `Invalid thread pool` assertion; a retry with `OMP_NUM_THREADS=1`
and `MKL_NUM_THREADS=1` succeeded and shut down normally. Jittor vLLM 0.24
TP2 with the same model explicitly loaded as FP16 then generated the exact
same 16 token IDs as native FP16 TP2 in two requests. Its first request
included extensive ACL JIT (~50.60 s); the second took 4.17 s, but different
vLLM versions prevent a fair speed ratio. This establishes an FP16
correctness workaround for this prompt, not a fix for BF16 numerical drift.
An opt-in comparison against vLLM's original clone + distributed all-reduce
fallback yielded the same Jittor BF16 TP2 16-token sequence as the adapter's
fast path, so that wrapper is not the cause; its diagnostic edit was reverted.
Full top-five logprobs from native vLLM-Ascend and Jittor then showed the
numerical drift already exists on TP1, before any cross-device collective:
at token 9 native TP1 favored ID 264 over 13482 by 0.375 logit, while Jittor
TP1 tied them and selected 264. Native TP2 tied them; Jittor TP2 favored
13482 by 0.125. All four paths agreed on the first eight generated IDs.
This moves the investigation toward model-forward numerical kernels
(attention/norm/GEMM) rather than a TP2-only HCCL adapter defect.

An opt-in CANN `aclnnAddRmsNorm` prototype now replaces Jittor ACL's two
separate BF16 Add and RmsNorm calls with the same fused operator family used
by native torch-npu. On identical random BF16 `[4,2048]` inputs, the new
path's normalized output and residual both matched
`torch_npu.npu_add_rms_norm` **bit-for-bit** (SHA256 of FP32-expanded output
`b995cdbb...` and residual `e85736de...`); the old two-call Jittor path
matched the residual but not the normalized output (`9a83735c...`). The
opt-in path is guarded by `JITTOR_NPU_FUSED_ADD_RMS_NORM=1` and remains off
by default pending model and performance tests. In a complete Qwen2.5 TP1
BF16 run, two requests still produced the 16 native token IDs, but token 9's
two candidates remained tied in Jittor instead of the native TP1 0.375 gap.
Thus the norm difference is real but is **not sufficient** to explain or fix
the full-model drift. The TP2 BF16 candidate also retained ID 13482 at token
9 (candidate logprob margin 0.125), versus native vLLM-Ascend's tied 264
and 13482. Two candidate requests took 11.672/2.517 s; the earlier
same-card default TP2 warmed request was 2.390 s, so there is no measured
speed gain for this workload. The fused Norm remains opt-in, not default.
Attention isolation then compared the same 16-token Qwen2.5-3B BF16 prompt
against native vLLM-Ascend. With both implementations' attention output
zeroed at every step, all 16 IDs matched. Selectively zeroing only prefill
attention (while preserving Jittor's KV cache writes and running normal
decode) also matched all 16 IDs. Selectively zeroing only decode attention
(with normal prefill) likewise matched all 16 IDs. These are diagnostic
controls, not valid model outputs; they show that attention arithmetic is
sufficient to expose the divergence, but do not by themselves prove which
individual kernel is wrong.

The NPU Qwen2.5 BF16 TP2 mismatch was reproduced against the direct full
prefill shortcut. With `JITTOR_VLLM_DIRECT_FULL_PREFILL=0`, real cache-backed
prefill attention (not a zeroed or bypassed attention) produced the exact
native vLLM-Ascend TP2 16-token sequence in two consecutive requests. At
token 9, Jittor's candidate IDs 264 and 13482 were tied at -0.742657,
matching native TP2's tied ordering (native logprobs -0.748356 each), whereas
the old shortcut favored the wrong ID 13482 by 0.125. Jittor TP1 retained
the same correct 16 IDs as before; its token-9 tie still differs numerically
from native TP1's 0.375 margin, so this is an output-level fix for the
measured TP2 regression, not proof of full logit parity.

`flash_attn.py` now disables direct full prefill by default only when ACL is
active. CUDA keeps its previous default, and `JITTOR_VLLM_DIRECT_FULL_PREFILL=1`
can explicitly re-enable the shortcut for experiments. A policy unit test
and versioned 16-token BF16 TP2 oracle regression were added. The explicit
cache-backed TP2 trial's warmed second request was 2.438 s versus earlier
shortcut trials around 2.23–2.39 s, but these were not adjacent A/B runs;
the correctness fix may trade some NPU throughput and requires controlled
benchmarking. The fresh-process versioned BF16 TP2 regression then passed
two 16-token requests against the real native vLLM-Ascend TP2 oracle with
no override (`oracle_match=true`, process exit 0); the warmed request took
2.733 s. Its executor needed a SIGTERM at shutdown after the grace period,
so clean multi-rank teardown remains a separate readiness item. The policy
test source was added and both changed Python files passed bytecode
compilation. An initial standalone unittest import stalled in the Jittor
cache initialization while another engine was running; after that process
was stopped, the isolated targeted policy unittest was rerun with CANN
sourced and passed (`Ran 1 test ... OK`).

A subsequent adjacent same-card TP2 Qwen2.5 BF16 16-token A/B quantified the
tradeoff: the old direct-prefill shortcut's three warmed requests took
2.390/2.383/2.365 s (median 2.383 s), but all chose wrong ID 13482 at token
9. The corrected cache-backed default's warmed requests took
2.667/2.674/2.641 s (median 2.667 s), all with native-oracle ID 264 at token
9. Thus this workload's output-correct default is about 11.9% slower; it is
not a performance improvement. Both engines still needed worker SIGTERM at
shutdown. An opt-in FP32 direct-prefill arithmetic experiment via
`JITTOR_VLLM_DIRECT_FULL_PREFILL=1` and
`JITTOR_VLLM_ACL_F32_DIRECT_PREFILL=1` preserved the native BF16 TP2
16-token sequence in two requests. At token 9 its candidates 264 and 13482
tied at -0.744046, near native TP2's tied -0.748356. The warmed second
request took 2.552 s, between old incorrect BF16-direct 2.383 s and correct
cache-backed 2.667 s; adjacent repeated A/B is required before claiming a
speedup. This FP32 shortcut is strictly opt-in while other NPU models and
longer prompts remain untested.

The adjacent FP32-direct / cache-backed / FP32-direct follow-up, each with
four requests and the first discarded as warmup, measured warmed medians
2.286 / 2.555 / 2.287 s respectively. All three 16-token sequences in both
FP32-direct engines matched the native TP2 BF16 oracle. This A/B/A supports
roughly 10% lower request latency for FP32-direct on this narrow Qwen2.5
workload, despite the first isolated FP32-direct trial's slower 2.552 s;
shared-machine timing remains variable. A native 64-token TP2 oracle run is
in progress before any default-policy change. That native vLLM-Ascend BF16
TP2 engine completed two identical 64-token requests. The Jittor FP32-direct
candidate then completed two 64-token requests and matched all 64 native
IDs exactly (first mismatch: none). Jittor's warmed request was 9.372 s;
the native warmed request was 5.391 s. This cross-version timing is not a
fair framework throughput ratio, but it shows that even the faster Jittor
prefill candidate does not close the overall serving gap. The adjacent
cache-backed default 64-token control also matched all 64 native IDs and
took 9.341 s warmed, versus 9.372 s for FP32-direct. Thus the 16-token
FP32-direct gain does not produce measurable end-to-end benefit once 64
decode tokens dominate the request; decode is the next performance target.
The versioned Qwen3-0.6B BF16 TP2 default-path non-regression then passed
two 16-token requests against its existing native PyTorch token oracle
(`oracle_match=true`); its warmed request took 1.896 s. Single-card
Qwen3 BF16 also passed two 16-token native-PyTorch-oracle requests on a
single NPU; its warmed request took 1.433 s and its worker exited normally.
Qwen2.5-3B BF16 TP1 then passed its versioned native 16-token oracle in two
requests with the new default, warmed request 1.816 s and clean worker exit.
This and the Qwen2 64-token results justify
keeping the ACL cache-backed path as the safe default while FP32-direct
remains an opt-in short-request candidate.

An unsynchronized host-stage trace installed only after the first 64-token
Qwen2.5 TP2 request measured 64 steps on each rank. Rank 0 medians were
`execute_model` 143.96 ms, `model_forward` 135.44 ms, `_prepare_inputs`
4.92 ms, and `sample_tokens` 1.32 ms; rank 1 was nearly identical. These
scopes nest and must not be summed. Unlike the earlier Qwen3 TP1 short-step
trace, this Qwen2.5 TP2 long decode clearly concentrates elapsed host time
inside model forward, not sampling. A targeted model-forward cProfile is
running to distinguish Python scheduling, synchronization and device work.
That profile on a separate 16-token warmed request recorded 1,168 TP
all-reduces (73 per token) and 1.117 s cumulative in the compatibility
layer's `Var.sync()` calls, out of 3.161 s of cProfile-instrumented forward
time. The profile itself raised median forward latency from the unprofiled
135 ms to 183 ms, so these are attribution figures, not throughput figures.
The native HCCL operator also has four default device/stream sync calls per
collective. `JT_HCCL_COLLECTIVE_SYNC=stream-order` is being tested opt-in while
retaining the Python `Var.sync()` safety boundary; the repository's hardware
checklist explicitly forbids changing the default before device numerical,
race, and ordering tests.
The first opt-in `stream-order` vLLM trial passed the versioned native TP2
Qwen2.5 BF16 16-token oracle across four requests; warmed times were
2.345/2.477/2.491 s, median 2.477 s. The earlier nearby full-sync default
median was 2.555 s, an apparent ~3% improvement too small to promote without
interleaved replication. Python's `Var.sync()` remains active for every
collective, so this switch cannot remove the larger host serialization. The
64-token trial also matched every native BF16 TP2 token, warmed 9.141 s
versus the nearby full-sync control's 9.341 s (~2%). Default stays `full`:
the required 200-iteration device race and ordering checklist has not passed.

An additional, more aggressive vLLM-only NPU opt-in prototype now patches
`DeviceCommunicatorBase.all_reduce` to return the out-of-place Jittor HCCL
result with `out_hint()` instead of entering the generic synchronous
Torch-distributed compatibility path. It is guarded by
`JITTOR_VLLM_NPU_LAZY_ALL_REDUCE=1`, leaves generic distributed defaults
untouched, and has a passing isolated unit test. This deliberately relaxes
per-collective issue/completion ordering and is **not** safe to promote on
a single model smoke. A timeout-guarded 16-token Qwen2.5 TP2 oracle and
subsequent race tests are required; the first hardware trial is running.
The first timeout-guarded trial completed without hanging and passed the
native BF16 TP2 16-token oracle twice, warmed 2.549 s. However, its cProfile
still showed all 1,168 `Var.sync()` calls: the prototype had looked for
`BaseDeviceCommunicator`, while the platform actually instantiates
`DeviceCommunicatorBase`. Thus this first timing is **not** an async result.
The class target was corrected, the opt-in unit test rerun successfully, and
a second timeout-guarded hardware/profile trial passed the actual native
BF16 TP2 16-token oracle twice. In the cProfile top 30, the prior 1,168
`Var.sync()` calls and compatibility all-reduce chain disappeared. The
profile-instrumented 16-token forward total dropped from 3.161 s before
activation to 1.893 s after activation. These cProfile figures are *not*
serving throughput measurements; an unprofiled four-request trial is now
running. That unprofiled run passed the native 16-token oracle in four
requests; warmed times 2.440/2.465/2.464 s, median 2.465 s versus a nearby
safe-default median of 2.555 s (~3.5% lower latency). This is far smaller
than the cProfile attribution difference and insufficient to justify a
default change. A timeout-guarded combination with HCCL's internal
`stream-order` switch is being tested next. Cross-rank issue ordering and
long-sequence race validation are still mandatory for either setting.
The two switches together passed four native-oracle 16-token TP2 requests;
warmed times 2.369/2.396/2.423 s, median 2.396 s. That is about 6% below
the earlier safe-default 2.555 s median but slower than the independently
correct FP32-direct prefill candidate's 2.286–2.287 s medians. Given this
modest gain and the untested cross-rank lazy issue-order semantics, neither
HCCL switch is being promoted. A 64-token/race campaign would be mandatory
before considering production use.

An adjacent CUDA Qwen3 TP2 direct-host-copy A/B on GPUs 4,5 gave baseline
128+16 medians 68.584 ms (three requests) and opt-in direct-copy 68.899 ms
(three requests), with exact oracle IDs and clean shutdown. The small
regression shows this host-copy switch does not generalize to TP2 and remains
opt-in.

Another opt-in CUDA TP2 trial disabled the `copy_to_uva` synchronization
without direct-host-copy. Three fixed-batch Qwen3 128+16 requests measured
67.511 ms median versus the adjacent synchronized baseline's 68.584 ms
(~1.6% lower latency), with clean shutdown. A same-engine 1,4,1,4 shape
switch then completed all ten sequences and matched a separately run native
PyTorch token oracle exactly. Both engines shut down cleanly. The native
oracle was configured `enforce_eager=True` while Jittor used its experimental
CUDA Graph capture, so their observed request times are **not** a fair speed
comparison. A second fixed-batch A/B replicated the gain (69.054 ms with
sync versus 66.587 ms without, three requests each). However, the adjacent
same-engine 1,4,1,4 synchronized baseline took 98.566/91.519/67.548/89.416
ms versus no-sync 97.044/92.928/69.544/90.862 ms: three of four dynamic
shape requests were slower without sync. The switch therefore stays opt-in;
it needs stronger host-source lifetime proof and workload-wide benefit.

## Complete Qwen2.5 TP2 CANN timeline, 2026-10-13

The versioned benchmark helper's TP2 BF16 oracle was corrected from an old
single-device row-parallel simulation to the real native vLLM-Ascend BF16
TP2 sequence. Its initial stale assertion prematurely shut down the first
profiling run; no model regression was involved. The final run attached
`msprof` dynamically to both loaded workers, started before generation,
stopped while the workers were held alive after the second request, and
exported both ranks successfully. Two 16-token Qwen2.5-3B-Instruct BF16 TP2
requests emitted all native token IDs exactly. The first measured 13.812 s
(JIT/cold work), the second 2.879 s; profiler overhead means this is not a
standalone serving benchmark.

Rank 0 and rank 1 each recorded 2,336 all-reduces, exactly 73 per generated
token across the two requests. In the warmed second request, the 1,168
all-reduces span about 2.860 s but sum to only 50.1 ms on rank 0 and
50.3 ms on rank 1 before interval overlap. In a 3.01 s trace window
surrounding that request, rank 0's non-communication operation intervals
have a 124.1 ms union (rank 1: 117.8 ms), and rank 0's communication
operation durations sum to 51.5 ms. These trace-accounting quantities are
not hardware utilization and should not be added naively across streams;
they nevertheless show the device kernels are not occupying most of the
~2.88 s request. `MatMulV2` is the largest non-communication operation type
at 55.1 ms for rank 0 in that window; `IncreFlashAttention` is 8.6 ms.

The same rank-0 host trace records 17,312 node launches taking 448 ms in
aggregate, 16,111 `aclrtLaunchKernelWithHostArgs` calls taking 146 ms,
2,468 `aclrtSynchronizeDevice` calls taking 56.6 ms, and 1,184 collective
API calls named `OpType::ALLGATHER` taking 251 ms during the warmed window.
These API scopes can nest, so their totals are **not** an additive CPU time
breakdown. They explain why merely removing explicit HCCL sync gave only a
3–6% end-to-end improvement: numerous model operations and their host-side
graph/ACL dispatch remain. The next optimization should reduce per-token
task count or replay the stable NPU decode path, with a small correctness
prototype and complete warmed A/B before a default change. No such speedup
is claimed yet. Profile source:
`_state/msprof-complete-fixed-qwen25-tp2-20261013/` on `cscg-hw00`.

## TP2 worker shutdown grace, 2026-10-13 late follow-up

The prior vLLM 0.24 default gives a worker only 5 seconds to exit after its
death pipe closes, then sends SIGTERM. A Qwen2.5-3B TP2 BF16 16-token run
was repeated with a 30-second diagnostic grace and `strace` attached only
after generation. Both workers exited with status 0 about 7 seconds after
shutdown began; the trace shows ordinary thread/memory cleanup, not a
permanent wait. The opt-in `JittorNpuPlatform` now sets a 30-second default
via `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`, while preserving any explicit
caller override and leaving CUDA untouched. This changes only teardown,
not inference execution. The platform policy unit suite passed 4/4.

A fresh, uninstrumented Qwen2.5-3B BF16 TP2 run using the new default
produced the native 16-token oracle in two requests (warmed request
2.602 s). Both workers exited normally in about 7 seconds with no SIGTERM.
A separate fresh Qwen2.5 BF16 TP2 64-token run reproduced all 64 native
vLLM-Ascend IDs exactly, twice in one engine. Its warmed request took
10.269 s; this is a different machine-time sample from the earlier 9.341 s
control and is not evidence of an inference regression. Both workers exited
cleanly about 6 seconds after shutdown started, again without SIGTERM.
The second model, Qwen3-0.6B BF16 TP2, then matched its 16-token oracle in
two requests. The warmed request took 2.094 s and both workers exited
gracefully in about 5 seconds. The different timing from the previous
1.896 s Qwen3 run is not a controlled performance A/B. Three independent
post-change TP2 engines (Qwen2 16, Qwen2 64, Qwen3 16) now have clean exits.
Batch changes, longer stress runs and failure-path teardown remain untested;
this is not yet a production-stability certification.
