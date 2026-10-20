# Native DDP overlap follow-up

Status: partial overall; OVL-01 through OVL-06 accepted on the tested native CUDA/NCCL SHM fallback (2026-10-16). The readiness memory follow-up below removes the former 2.21-times allocator peak in the measured Transformer while retaining device overlap. The 72-case matrix contains **69 PASS, 2 FAIL (COMM-04, INT-07), and 1 UNSUPPORTED (MOD-12)**, with no unrun or resource-wait cases. Investigation of the sampled host's P2P failures is deferred at the user's request; the failures also reproduce independently in PyTorch and are not attributed to Jittor reducer code. The remote refactor branch remains unmerged, and unconditional release is not claimed.

The current scheduling change, memory and timeline evidence are recorded in
[the readiness memory follow-up](#readiness-memory-follow-up-2026-10-16).
The preceding stable-kernel candidate is recorded in
[the final r2 acceptance section](#final-production-r2-acceptance-2026-10-16).
Earlier sections retain the original baselines, failures and diagnostic sequence;
their FAIL/pending statements describe those earlier revisions.

## Initial code and validation (historical)

- Checkout: `integrate/cgq-transformers-native-multicard-target`
- Code revision at launch: `48de31b4157fa128c5e1e19b052fefa3be7ed098`
- Scope: native single-host `optimizer.backward(loss)` plus NCCL DDP; no MPI,
  `mpirun`, or `compat.torch`.
- Focused test: `tests/distributed/test_native_ddp.py` -> `18 passed`.
- The reducer keeps homogeneous reverse buckets, deferred NCCL joins, and
  post-join averaging/unpacking. The optimizer still constructs one complete
  backward graph before this reducer receives its gradient buffers.

## Device evidence

Both runs used the fixed 12-layer, hidden-768 Transformer (85,842,688
parameters), sequence length 512, global batch 8, three warmup updates and four
measured updates on physical GPUs 5 and 9. Each run used an isolated
`HOME`/`JITTOR_HOME`/`TMPDIR`, `JT_BUILD_USE_MPI=0`, `JT_BUILD_USE_NCCL=1`, and
`NCCL_P2P_DISABLE=1` as the validated SHM fallback. No other process was
terminated; startup checks found about 78 and 75 GiB free on the two cards.

The first production reducer trace is
`/home/cgq/jittor-lab/_state/jittor-native-multicard/native-ddp-readiness-20261014-01/nsys/native-ddp-2gpu.sqlite`.
For all eight measured rank/update windows, backward GEMM kernels completed
before the NCCL all-reduce kernels: `overlap_ms=0` and `comm_overlap_pct=0`.
The second trace after the reducer ordering cleanup is
`/home/cgq/jittor-lab/_state/jittor-native-multicard/native-ddp-readiness-20261014-03/nsys/native-ddp-readiness.sqlite`.
It gives the same result for `update_0` through `update_3` on both ranks:
`gemm_n=164`, `nccl_n=1`, `overlap_ms=0`, and `comm_overlap_pct=0`.

The second run returned launcher `rc=0`; every rank reported finite, nonzero
gradients, and both ranks reported the same final parameter SHA
`26829387c198671241c06e972165d0405581dcd14c583d6cd3fb2db300541fed`.
Rank-0 steady-state throughput was 22.65 samples/s (0.3532 s/update), and
rank-1 was 22.62 samples/s (0.3536 s/update), during shared GPU load. This is
a bounded baseline observation, not a formal cross-framework speed claim. The earlier experimental
multiple-`jt.grad` implementation was removed because it did show apparent
timeline overlap but reduced throughput to about 9.5 samples/s by rebuilding
the backward graph once per bucket.

## Synchronous baseline completion

The Jittor trace above and the independent PyTorch 2.6 trace at
`/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl-baseline-20261013-01/nsys/torch-ddp-2gpu.sqlite`
use the same 12-layer, 85,842,688-parameter Transformer, sequence length 512,
global batch 8, two ranks, three warmup updates and four measured updates.
They cover eight rank/update windows each. A retained analysis script and JSON
are under `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl-sync-baseline-20261014-01/`.

Across the Jittor trace, summed GEMM device time is 1216.214 ms and NCCL
all-reduce device time is 1330.331 ms. Each measured update has one NCCL
all-reduce kernel, and every rank/update window completes its GEMM kernels
before NCCL starts; the measured GEMM/NCCL interval intersection is zero.
This establishes the complete-backward-then-reduce baseline and its current
bucket submission dependency.

The PyTorch trace has 13 NCCL kernels per measured rank/update window. Summing
the per-window intersections between the unions of GEMM and NCCL kernel
intervals gives 632.059 ms overlap from 1231.981 ms of NCCL device time
(51.30%) for this capture. This describes the sampled profiler run only; it
does not predict Jittor overlap speedup. The isolated calculation is recorded
by `analyze_nsys.py` with its output in `timeline-summary.json`.

A separate same-model Jittor memory sample on physical GPUs 0/2 returned
`rc=0`, with four measured updates at about 0.2271 s/update and 3890 MiB
maximum sampled process memory per rank. GPU utilization was shared and high;
this run is useful for peak-memory accounting, not a formal performance
comparison. Its raw samples, rank logs and JSON outputs are under
`/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl-sync-baseline-20261014-01/`.

## Acceptance consequence

These captures completed OVL-01 and OVL-02 at the time of the baseline run. The readiness results below supersede that earlier status. The remaining acceptance issue is numerical: the same frozen FP32 comparator rejects the readiness trajectory after several AdamW updates. The default NCCL P2P/CUMEM issue remains independent and unresolved.

## FP16 AMP readiness signature

Status: fixed and verified on the current checkout; this closes the FP16 AMP
readiness-signature failure only. It does not change the status of OVL-03
through OVL-06.

- Checkout: `48de31b4157fa128c5e1e19b052fefa3be7ed098`
- Fetched remote, not merged: `origin/2.0-refactor@3495d443d9d52eb0672f91b22735a7eb585f0c46`
- Failure reproduction: two ranks recorded identical readiness payloads, but
  the signature all-reduce returned `Inf` for second-moment entries of 57,600.
  Under FP16 AMP, the intermediate two-rank sum overflowed before the mean.
- Fix: reduce only the 16 low-magnitude digest nibbles and compare the mean to
  the local signature. The collective remains `mean`; no optimizer or model
  arithmetic changed.
- CPU regression: `tests/distributed/test_native_ddp.py` -> 24 passed.
- Device regression: physical GPUs 2/3, NCCL SHM fallback
  (`NCCL_P2P_DISABLE=1`), FP32 parameters with FP16 autocast and GradScaler,
  global batch 8, three updates and six gradient buckets. Launcher returned
  `rc=0`; each update issued one signature collective plus six bucket
  all-reduces. Global-batch gradient max error was at most `4.77e-7`;
  checkpoint model, optimizer, scaler, resumed update and rank replicas were
  exact (`replica_max_abs=0`, scaler scale `64`).
- Raw evidence, including the pre-fix `Inf` output, is retained under
  `/home/cgq/jittor-lab/_state/jittor-native-multicard/native-ddp-readiness-precision-20261014-01/amp-fp16/`.

The result extends matrix case DT-04. It establishes correctness for this
two-rank AMP readiness scenario; it is not evidence of large-Transformer
communication overlap or default P2P/CUMEM correctness.


## Readiness implementation and acceptance update

Updated: 2026-10-14. The checkout remained `48de31b4157fa128c5e1e19b052fefa3be7ed098`; `origin/2.0-refactor@c1131847f0ed50b0b2cbe52c4f049b253f28d704` was fetched and recorded, not merged. The native readiness implementation and its tests are uncommitted worktree changes. Their targeted source diff SHA-256 is `1a6b0cfd360eec06a7c294699088b0f6c718aae06ad0b89d2606054179286a97`. FP32 tolerance remains `atol=rtol=2e-6`.

- **OVL-03 PASS:** the native 12-layer Transformer path ran through `optimizer.backward/step` on two ranks, with 16 gradient buckets and NCCL SHM fallback. No MPI, `mpirun`, or `compat.torch` was used. The readiness path issues the bucket collectives from the gradient callback; rank collective shapes and final replica hashes agree. The actual CUDA overlap is measured separately by OVL-06.
- **OVL-04 FAIL:** a complete `rc=0` repeat on physical GPUs 2/4 used the frozen 12-layer, hidden-768, 85,842,688-parameter workload, sequence 512, global batch 8, three warmup and four measured updates, fused AdamW with `eps=1e-8`. Local gradients (max absolute difference `2.61e-8`), final check gradients (`2.33e-9`), and each path's independent AdamW recurrence pass. The moments also pass. The cross-path parameter check first exceeds tolerance at update index 2 and ends at `2.7503e-6` max absolute (`1.3752` tolerance ratio), at `blocks.0.qkv.bias[1469]`. Both ranks are identical within each path. Two earlier repetitions also failed (`3.61e-6` and `3.69e-6`).

  The parameter mismatch is consistent with AdamW's `eps=1e-8` sensitivity on near-zero QKV-bias gradients: in an earlier complete step record, a gradient difference of `3.75e-11` at such a coordinate accompanied a `1.12e-6` first-update parameter difference. Each path independently passes its AdamW recurrence oracle, so the evidence does not currently identify an optimizer recurrence defect. The exact source of the small backward rounding difference remains unverified; this is still a failed frozen acceptance case, with no tolerance adjustment. The earlier two-repeat launcher ended at its 300-second job deadline after writing a second pair comparison (`rc=124`); the newer single-repeat run completed all step records and returned `rc=0`. Artifacts are under `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl04-local-grad-numerics-2gpu-20261014-02/`.
- **OVL-05 PASS:** four-rank CUDA/NCCL SHM-fallback lifecycle covered `no_sync` accumulation, clipping, rank-only/unused/empty parameters, AdamW checkpoint restore and resumed update. Launcher `rc=0`; all ranks drained buckets in the same order, checkpoint model/optimizer state matched exactly, resumed update max_abs was zero, and rank replicas stayed exact. Raw outputs and logs are under `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl05-readiness-edge-4gpu-20261014-01/`.
- **OVL-06 PASS:** the same-process, same-input Nsight capture used eight measured rank/update windows. The synchronous negative control had no NCCL kernel inside the backward marker and zero intersection. Readiness had GEMM/NCCL intersections in all eight windows: `50.808 ms / 110.688 ms = 45.902%` of NCCL kernel duration. The host range encloses `optimizer.backward`, including its deferred communication join, so the device intervals establish overlap before that call returns. This is run-specific profiler evidence, not a general speedup claim. Traces and analysis are under `/home/cgq/jittor-lab/_state/jittor-native-multicard/native-ddp-overlap-current-20261014-01/`.

The 72-case ledger now contains 61 PASS, 8 WAITING_RESOURCE, 0 NOT_RUN, 1 UNSUPPORTED and 2 FAIL (COMM-04 and OVL-04). The tolerance contract is unchanged. Overall readiness-overlap acceptance remains blocked by OVL-04; the staged `origin/2.0-refactor` merge remains deferred until the full matrix and overlap gate are complete. See the machine-readable ledger at `/home/cgq/jittor-lab/jittor-native-multicard/scenario-matrix-20261012.json`.

## 2026-10-16 model and optimizer follow-up

The remaining model cases were run on the current checkout `48de31b4157fa128c5e1e19b052fefa3be7ed098` before any `origin/2.0-refactor` integration. On physical GPUs 2/5, with isolated `HOME`, `JITTOR_HOME` and `TMPDIR`, native Jittor DDP used NCCL SHM fallback (`NCCL_P2P_DISABLE=1`) and returned launcher `rc=0` for all three fixed two-rank cases:

- **MOD-04 ViT PASS:** patch embedding, class token, attention, normalization and classifier; one bucket; finite nonzero gradients; rank gradient arrays identical; `replica_max_abs=0`; final parameter SHA `8d4935fcef6e9fbaf5c8fa0014151e64e5f67124dcb4281417fde52a9d55f50a`. All gradient families passed the frozen FP32 tolerance against the single-card Torch reference, with worst ratio `0.00346`.
- **MOD-05 encoder PASS:** padding mask and position buffer; one bucket; rank parameter SHA `ff65855106cdcc5d0299c560f94097a3848f12c2f37990738991a9d922e92849`; `replica_max_abs=0`; all 18 gradient families passed, with worst ratio `0.000916`.
- **MOD-07 encoder-decoder PASS:** encoder, decoder and cross-attention; one bucket; rank parameter SHA `544775b9a2dc66262f50e03172850e98faa78629144c1c3438c4a1ca44a1441d`; `replica_max_abs=0`; all gradient families passed, with worst ratio `0.00116`.

These cases use the fallback transport only and are correctness/model gates, not speed claims. External GPU load was retained throughout. The machine-readable ledger is now `68 PASS / 3 FAIL / 1 UNSUPPORTED`, with no remaining `WAITING_RESOURCE` entries.

To separate the unresolved OVL-04 parameter drift from the fused AdamW kernel, the same two-rank, 12-layer hidden-768, sequence-512, global-batch-8, three-warmup/four-measured-step probe was rerun in an experiment-only copy with ordinary (non-fused) AdamW. The run returned `rc=0`, but readiness versus complete-backward still failed the unchanged FP32 parameter contract: maximum absolute difference `3.3333115e-6`, tolerance ratio `1.66665`; local gradient and check-gradient comparisons passed. This makes an optimizer-kernel recurrence defect unlikely. The earlier single-GPU ablations also observed near-zero QKV-bias variation through ordinary `jt.grad`, no-op callbacks and holder-update callbacks, so the evidence does not isolate a callback-specific defect. The current explanation is small backward rounding differences amplified by AdamW `eps=1e-8`; the exact source remains unverified and OVL-04 stays FAIL. No tolerance or training semantics were changed. Evidence: `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl04-unfused-adamw-20261016-01/`.

Documentation validation: `tests/structure/test_refactor_board_contract.py` passed 3/3 using `JITTOR_TORCH_SHIM=1 PYTHONPATH=/home/cgq/jittor/python`; matrix JSON parses and `git diff --check` is clean for the updated board/report. The targeted gate used a separate cold `JITTOR_HOME` and compiled 228 core translation units; its result record is under `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl-doc-check-20261014-01/validation.json`.

## LIFE-03: pending-bucket failure and collective-order mismatch

Updated 2026-10-14T07:55:27Z. Validation used the current checkout `48de31b4157fa128c5e1e19b052fefa3be7ed098`, frozen baseline `3369809003c15cfcb1d843b81c1ed7027c204c97`, and fetched but unmerged `origin/2.0-refactor@c1131847f0ed50b0b2cbe52c4f049b253f28d704`. Physical GPUs 1/2 each had about 81 GiB free at launch. The runs used `NCCL_P2P_DISABLE=1`; NCCL reported `SHM/direct/direct`. MPI and `compat.torch` were not used.

- **Parameter-order mismatch passed:** rank 0 registered `alpha, beta` and rank 1 registered `beta, alpha`, with identical names, shapes and dtypes. Both ranks raised `RuntimeError: DistributedDataParallel could not verify the reduced module parameter/buffer metadata signature` during construction; parameter broadcast did not begin. The launcher returned 1 before its 90 s job deadline and left no rendezvous file or rank process.
- **Failure with a pending bucket passed:** rank 1 entered the native DDP all-reduce for its one 16-element bucket; the collective call returned to the host while `optimizer.backward` remained pending. After observing that entry, rank 0 exited with code 23. The launcher returned 23 before its 45 s job deadline, stopped rank 1, removed rendezvous state and left no rank process. This is direct launcher fail-stop evidence for a peer waiting in the bucket path.
- **Launcher regression passed:** the current `tests/distributed/test_launch.py` suite reports 17 passed, 0 skipped in 7.57 s. Its captured output is `/home/cgq/jittor-lab/_state/jittor-native-multicard/life03-launch-unit-20261014-01/pytest.log`.

Raw rank logs, marker, process-cleanup checks, commands, source hashes and the run manifest are retained under `/home/cgq/jittor-lab/_state/jittor-native-multicard/life03-failure-20261014-01/`. No other user's process was stopped. This closes LIFE-03 for the tested fallback topology; it does not change COMM-04's default P2P/CUMEM failure or OVL-04's frozen-tolerance failure. The 72-case matrix is now 62 PASS, 7 WAITING_RESOURCE, 0 NOT_RUN, 1 UNSUPPORTED and 2 FAIL.


Post-update board contract check: `tests/structure/test_refactor_board_contract.py` passed 3/3 in 0.06 s. The dedicated cache build completed 228/228 core translation units (269.856 s); the successful run log is `/home/cgq/jittor-lab/_state/jittor-native-multicard/life03-board-contract-20261014-01/pytest.log`. The first import attempt only requested the documented rerun after rebuilding `jit_utils`; its log is retained alongside the successful result.


## OVL-04 callback and repeatability diagnosis

Updated 2026-10-14. The checkout remains `48de31b4157fa128c5e1e19b052fefa3be7ed098`; a fresh fetch records `origin/2.0-refactor@46e5f7b0e2e2898e1a69cac0b6dc3d59b3481580`, still unmerged. The frozen FP32 comparison remains `atol=rtol=2e-6`.

A fixed-weight, fixed-input single-GPU ablation on physical GPU 1 (81,038 MiB free at launch) used the same 12-layer, hidden-768, 85,842,688-parameter Transformer and rank-0 local batch of four. It compared two ordinary `jt.grad` runs, two no-op `grad_with_callback` runs, callback capture, callback-side optimizer gradient-slot update, and callback bucket readiness with `jt.sync` plus gradient packing across 16 buckets. All outputs were finite and within the frozen tolerance. Ordinary `jt.grad` repeated-run max gradient difference was `4.10e-8`; the largest callback-path difference from that reference was `6.71e-8`, tolerance ratio `0.0311`. The bucket callback path differed by `3.73e-8` (ratio `0.0173`).

A second fixed-batch probe repeated ordinary autograd, no-op callback, and holder-update callback three times, saving all 2,304 `blocks.0.qkv.bias` values. At the later two-rank failure coordinate `[1469]`, the same first-batch gradient spans were `1.14e-11` for `jt.grad`, `3.18e-12` for no-op callback, and `1.05e-11` for holder update. This reproduces near-zero QKV-bias roundoff at the same scale through ordinary autograd; it did not reproduce a callback-specific gradient bug. Combined with each distributed path independently matching its AdamW recurrence, the current evidence attributes the small trajectory drift to AdamW `eps=1e-8` sensitivity after backward rounding at near-zero gradients. This does **not** satisfy OVL-04: the full two-rank parameter comparator still fails at `2.7503e-6` / `1.3752` frozen tolerance ratio. No reducer, optimizer, or tolerance changes were made, and OVL-04 remains FAIL pending a correctness-preserving way to meet the frozen trajectory contract or a separately approved matrix revision based on independent evidence.

Raw artifacts and exact environment manifests are under `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl04-single-callback-ablation-gpu1-20261014-01/` and `/home/cgq/jittor-lab/_state/jittor-native-multicard/ovl04-qkv-repeatability-gpu1-20261014-01/`. Both launchers returned 0. The fresh JIT cache was isolated to the first run; the second serial diagnostic reused that now-warm cache. No other process or experiment output was modified.

## INT-07 P2P transport diagnosis and single-rank fix

Updated 2026-10-14. The large Transformer INT-07 run used physical GPUs
4/7/8/9, `NCCL_P2P_LEVEL=SYS`, an 8,388,608-element oversized parameter and a
600-second launcher deadline. The launcher returned 124 and cleaned every rank;
rank 0 reached DDP construction and `step_start`, while ranks 1/2/3 remained in
construction. A reduced one-layer, 84,160-parameter four-rank probe on GPUs
0/1/4/5 reproduced the constructor stall with default P2P/CUMEM. Stage tracing
showed rank 0 advancing through the 16-element signature collective and 18
parameter broadcasts while ranks 1/2/3 did not return from that signature
collective.

An independent 16-element native ProcessGroup all-reduce probe on 0/1/4/5 also
failed to complete under P2P/CUMEM. The same probe with
`NCCL_P2P_DISABLE=1` selected SHM/direct/direct, returned the analytic mean
2.5 on all ranks, and completed a barrier. The full reduced DDP probe under the
same fallback completed construction, forward/backward, optimizer step and
final gradient checks on all four ranks with launcher `rc=0`. This isolates the
observed INT-07 failure to the current host's P2P transport path; it does not
clear default P2P acceptance or establish a Python reducer defect.

The raw single-card benchmark also exposed a separate lifecycle bug: `jtrun`
sets `jt.in_mpi=True` for its MPI-free environment even when `WORLD_SIZE=1`,
so an unwrapped raw optimizer called the unavailable legacy `Var.mpi_all_reduce`.
The optimizer now enters that legacy branch only when the distributed world size
is greater than one. The focused regression
`test_one_rank_launcher_does_not_use_legacy_mpi_collectives` passes, and a
direct isolated raw CUDA run completes. This change is limited to the current
working tree and remains uncommitted with the broader DDP work.

The matrix now has 72 cases: 62 PASS, 6 WAITING_RESOURCE, 1 UNSUPPORTED and 3
FAIL (`COMM-04`, `OVL-04` and `INT-07`). The new `INT-07` failure is a default
P2P/CUMEM transport failure; the fallback DDP result is evidence for the
workaround only. The fetched, unmerged `origin/2.0-refactor` tip recorded for
the next controlled integration is `46e5f7b0e2e2898e1a69cac0b6dc3d59b3481580`.

## INT-07 after constructor synchronization: topology and fallback follow-up

Updated 2026-10-14. The checkout remained `48de31b4157fa128c5e1e19b052fefa3be7ed098`; the `jt.sync_all(True)` constructor change remains uncommitted. Physical GPUs 2/3/4/5 were free at launch and no other process was stopped.

- **Default P2P/CUMEM remains a failure.** The exact INT-07 workload (12 layers, hidden 768, sequence 512, global batch 8, 85,842,688 base parameters plus an 8,388,608-element oversized parameter, four ranks, `NCCL_P2P_LEVEL=SYS`) was launched on GPUs 2/3/4/5 with the constructor synchronization fix and a 600-second deadline. All four ranks reached `sync_all_complete`, entered the metadata signature all-reduce, and submitted the 16-element collective, but none returned from that collective before `jtrun` returned `rc=124` and cleaned all ranks. An independent four-rank 16-element native ProcessGroup probe on the same devices and `NCCL_P2P_LEVEL=SYS` reproduced the same submitted-without-host-readback behavior and returned `rc=124`. This keeps INT-07 FAIL and adds a second topology sample to COMM-04; it does not identify whether the host fault is NCCL, driver, or PCIe peer transport.
- **SHM/direct fallback completes the large workload.** With the identical source/model/configuration and `NCCL_P2P_DISABLE=1`, the first run reached every rank's first backward but was stopped at a 300-second deadline while four ranks serialized their first CUDA/cutt compilation. Reusing that isolated cache with a 900-second deadline completed with launcher `rc=0`. All four rank JSONs report 94,231,296 parameters including the oversized probe, 17 buckets with sizes `[1,12,10,8,10,8,10,8,10,8,10,8,10,8,10,8,12]`, oversized singleton bucket `[0]` (33,554,432 bytes), 127 collective events, finite nonzero gradients, and the same parameter SHA-256 `42157e5b54e0d1e61f12b7f4ecfaa9158e55e5028ed9fe3d51579bd97263a6d7` on every rank. Measured throughput was 59.43-59.54 samples/s, or about 30.4k tokens/s, after warmup. This is fallback correctness and performance evidence only; it does not clear default P2P/CUMEM.

Raw artifacts are under `/home/cgq/jittor-lab/_state/jittor-native-multicard/int07-large-transformer-4gpu-sys-after-fix-20261014-01/`, `/home/cgq/jittor-lab/_state/jittor-native-multicard/int07-p2p-sys-probe-2345-20261014-01/`, and the completed fallback results under `int07-large-transformer-4gpu-shm-after-fix-20261014-01/results-retry-900s/`. The machine-readable ledger keeps INT-07 as FAIL with the fallback evidence recorded separately.

## LIFE-01 cold and warm cache lifecycle

Updated 2026-10-14T12:34:47Z. The current checkout remained
`48de31b4157fa128c5e1e19b052fefa3be7ed098`; the fetched
`origin/2.0-refactor@cded3102037edc3663b8f2054a6ebc06da6ebfd2` remains
unmerged. The lifecycle harness was corrected so its collective accounting
ignores the separate 16-element metadata signature reduction and counts only
gradient bucket reductions. That change is in the external experiment script
`/home/cgq/jittor-lab/jittor-native-multicard/optimizer_lifecycle_ddp_smoke.py`,
not product source.

The world-size-1 cold-cache prewarm on physical GPU 2 returned launcher
`rc=0`. A separate world-size-2 cold-cache run on physical GPUs 2/3 used
`NCCL_P2P_DISABLE=1`, `JT_BUILD_USE_MPI=0`, `JT_BUILD_USE_NCCL=1`, and isolated
`HOME`, `JITTOR_HOME`, `TMPDIR`, and `XDG_CACHE_HOME`. It compiled the isolated
Jittor cache and returned `rc=0`. The three lifecycle cases (optimizer before
DDP, optimizer after DDP, and dynamic parameter group) each completed four
updates; each had one gradient bucket and four bucket reductions, plus one
16-element signature reduction per backward. Optimizer state stayed finite,
the maximum parameter update was `3.92283499e-3`, and the maximum cross-rank
replica difference was `0`.

A warm retry reused the same world-size-2 Jittor cache and independently
returned `rc=0` with the same three cases and checks. Cold and warm runs are
stored separately under
`/home/cgq/jittor-lab/_state/jittor-native-multicard/life01-cold-cache-20261014-01/`;
their rank logs, JSON results, launcher logs, and isolated cache paths are the
evidence. No rank worker, launcher, or rendezvous file remained afterward.
This validates the cache/lifecycle behavior under the explicit SHM fallback;
it is not a startup performance claim and does not change the independent
default P2P/CUMEM failure. The matrix is now 63 PASS, 5 WAITING_RESOURCE, 1
UNSUPPORTED, and 3 FAIL (`COMM-04`, `OVL-04`, `INT-07`). `LIFE-04` and `INT-08`
remain open for failed-initialization retry coverage.

## LIFE-04 failed-initialization retry

Updated 2026-10-14T12:46:00Z. On physical GPUs 2/3, a 64-element native
ProcessGroup oracle used one isolated `JITTOR_HOME` and first launched with
default P2P/CUMEM (`NCCL_P2P_LEVEL=SYS`). Both ranks initialized and entered the
all-reduce, but host readback did not complete; `jtrun --job-timeout=45`
returned `rc=124`, stopped both rank process groups, and removed rendezvous
state. Rank logs and the bounded launcher log are under
`/home/cgq/jittor-lab/_state/jittor-native-multicard/life04-failed-init-retry-20261014-01/logs-p2p/`.

The immediate retry reused that same cache and changed only
`NCCL_P2P_DISABLE=1`. Both ranks completed the same analytic mean with
`max_abs=0`, destroyed their process groups, and the launcher returned `rc=0`.
No worker, launcher, root-info, heartbeat, or process-group file remained after
either launch. This closes the cleanup/retry lifecycle case and leaves the
default P2P/CUMEM transport failures in COMM-04 and INT-07 unchanged.

## INT-08 cold rank failure and warm retry

Updated 2026-10-15. The checkout remained
`48de31b4157fa128c5e1e19b052fefa3be7ed098` against validated baseline
`3369809003c15cfcb1d843b81c1ed7027c204c97`; the freshly fetched, unmerged
`origin/2.0-refactor` tip is `1e98e127e406f1924f3c3729f7f58254637b4455`.
The focused harness is `/home/cgq/jittor-lab/jittor-native-multicard/
int08_cold_rank_failure_warm_retry_20261015.py`.

On physical GPUs 4/7, `NCCL_P2P_DISABLE=1`, and a fresh isolated
`JITTOR_HOME`, the two ranks compiled and launched the actual native DDP
workload. Instrumentation excluded the DDP metadata signature collective
(shape 16) and observed rank 1's single gradient bucket (`shape=(4,)`,
`op=sum`) enter and return. Rank 1 then paused before `optimizer.backward`
could finish; rank 0 observed the marker and exited with code 23. `jtrun`
returned `rc=23`, stopped rank 1, removed rendezvous state, and left no rank
or launcher process. This exercises cleanup after a gradient collective has
returned but before backward finishes; it does not claim the peer remained
blocked inside NCCL.

The immediate warm retry reused the same isolated cache and completed two
updates on both ranks with `rc=0`. Each rank reported two 4-element bucket
reductions, analytic gradients `[2.5, 3.5]` and `[1, 1]` on both steps, finite
parameters, final parameters `[0.95000005, 1.93000007, 0.48000002,
-0.51999998]`, and cross-rank replica `max_abs=0`. Launcher/rank processes
and rendezvous files were absent after completion. There is no performance
claim. The measured transport was the explicit SHM fallback and does not
change COMM-04 or INT-07's default P2P/CUMEM failures.

Evidence, including rank logs, launcher logs, preflight, marker, and exact
rank JSON results, is under
`/home/cgq/jittor-lab/_state/jittor-native-multicard/
int08-cold-rank-failure-warm-retry-20261015-11/`. Earlier attempts under
`20261015-01` through `20261015-10` are retained separately; they exposed
harness path/marker mistakes, Jittor cold-import restart behavior, and
signature-collective stalls on other shared-device runs, and are not reported
as product failures. INT-08 is PASS for the bounded cleanup-and-retry scope
above. At that 2026-10-15 snapshot, the 72-case matrix had 65 PASS,
3 WAITING_RESOURCE, 1 UNSUPPORTED, and 3 FAIL (`COMM-04`, `OVL-04`, `INT-07`).

## COMM-04 default transport pair follow-up

Updated 2026-10-16. The checkout remains
`48de31b4157fa128c5e1e19b052fefa3be7ed098` against validated baseline
`3369809003c15cfcb1d843b81c1ed7027c204c97`. I fetched and recorded
`origin/2.0-refactor@4cefbacdca771f7d25512beccc541d902836b894`; it remains
unmerged. Both probes used the existing 16-element ProcessGroup oracle,
`JT_BUILD_USE_MPI=0`, `JT_BUILD_USE_NCCL=1`, isolated `HOME`/`JITTOR_HOME`/
`TMPDIR`, and explicitly unset `NCCL_P2P_DISABLE` and `NCCL_P2P_LEVEL`.
The helper SHA-256 is
`ba66a9c497c1ad239cf9922c0b42f0f263439afc45907bdfd29b20f0954c82cf`.

On physical GPUs 2/5, the default environment selected
`SHM/direct/direct`. Both ranks read the analytic mean `1.5` for all 16
float32 elements, completed the barrier and process-group destruction, and
the launcher returned `rc=0`. This is a default-routing SHM result for this
pair; it is not P2P/CUMEM evidence. On physical GPUs 2/3, the same default
environment selected `P2P/CUMEM` on all four channels. Both ranks submitted
the all-reduce but neither completed host readback. `--job-timeout=45`
returned `rc=124`; the launcher reported that both rank process groups were
stopped. A post-run process/GPU check found no probe processes or allocations,
and no rank result JSON was written. The frozen matrix therefore remains
`COMM-04=FAIL`; the result corroborates the existing transport failure and
cleanup behavior, without identifying whether NCCL, the driver, or PCIe peer
transport is the root cause.

The GPU 2/5 and 2/3 run directories, including preflight snapshots, rank logs,
launcher logs, exit codes and isolated caches, are retained under
`/home/cgq/jittor-lab/_state/jittor-native-multicard/
comm04-p2p-minimal-20261016-01/` and
`/home/cgq/jittor-lab/_state/jittor-native-multicard/
comm04-p2p-minimal-gpu23-20261016-01/`. The matrix remains 68 PASS, 3 FAIL,
and 1 UNSUPPORTED, with no `WAITING_RESOURCE` cases. OVL-04 and INT-07 remain
open; no tolerance or training semantics changed.

The post-update board-contract validation used
`JITTOR_TORCH_SHIM=1 PYTHONPATH=/home/cgq/jittor/python` and a fresh isolated
`HOME`/`JITTOR_HOME`/`TMPDIR`. The cold cache compiled 228/228 core translation
units, and `tests/structure/test_refactor_board_contract.py` passed 3/3 in
0.06 seconds. Its full log is retained at
`/home/cgq/jittor-lab/_state/jittor-native-multicard/
comm04-doc-contract-20261016-01/pytest.log`.

## OVL-04 local atomic-order diagnosis and production repair candidate

Updated 2026-10-16. Checkout remains `48de31b4157fa128c5e1e19b052fefa3be7ed098`
with the preserved working tree. The refreshed, unmerged remote is
`origin/2.0-refactor@fcc5515d29abecde1fff8275c9141235e7fbf514`.
FP32 `atol=rtol=2e-6`, AdamW epsilon `1e-8`, model/data/global batch and update
counts were retained. All runs below use native CUDA/NCCL with explicit SHM
fallback, without MPI or the Torch compatibility frontend.

Fresh-process two-rank repeats on physical GPUs 1/9 establish that the drift
also exists within each path: three-step sync/sync final parameter max abs is
`2.15758717558856e-6`, readiness/readiness is `2.6213569697119965e-6`.
Each individual run has identical final replica hashes. A separate fixed
`[2,64,384]` cancellation test excludes DDP, GEMM and softmax and tests sum,
bias broadcast backward and three ordinary AdamW updates. Default
`para_opt_level=3` gives 30 distinct sum hashes in 30 trials, sum repeat max
abs `1.311946107307449e-10`, and parameter repeat max abs
`4.813529244529491e-6`. Existing level 4 gives one hash and zero repeat
differences. Generated CUDA confirms multiple blocks atomically add to the
same bias output in the default kernel; its warp fold cannot trigger when
lanes hold different output channels.

Level 4 alone reduces the error but does not pass the complete 12-layer,
hidden-768, 85,842,688-parameter Transformer: with seq512, global batch 8,
three warmup plus four measured updates and fused AdamW, final parameter
max abs remains `2.3371571842289995e-6` (tolerance ratio
`1.168577378421826`). The remaining first-step local difference is in the
embedding weight gradient, whose repeated token indices use atomic scatter.

Combining level 4 with an experiment-only embedding backward that adds each
token's occurrences in input order completes the same large workload with
launcher `rc=0`. Final parameters, initial local gradients, final check
gradients, and each of the four recorded parameter/gradient/first-moment/
second-moment comparisons have max abs **zero**. The small two-layer model
also gives zero differences. This isolates both local atomic accumulation
sources for the observed failure. It does not establish default production
acceptance: neither the global level-4 flag nor the diagnostic vocabulary
scan is the proposed release implementation.

The production candidate instead adds CUDA-only FP32 NN backend paths for
linear bias (fixed partial tiles followed by one writer per channel) and
ordinary dense embedding (CUB stable radix sort followed by one writer per
token segment). The latter normalises validated original indices to int64
before sorting, so negative/positive aliases share a segment. Both custom
backwards define a differentiable transpose for higher derivatives. General
reductions, scatter, other dtypes and unsupported embedding options keep
their existing paths. Real-device oracle/boundary tests and default-model
acceptance are pending; **OVL-04 remains FAIL**.

Unversioned artifacts are under
`$JITTOR_LAB_ROOT/_state/jittor-native-multicard/ovl04-v2-20261016-01/`:
`bias-reduce-minimal.json`, `complete-repeatability-comparison.json`,
`results/large-lvl4/`, and `results/stable-embedding-large/`, including
per-rank snapshots and four-step comparisons. The large diagnostic run's
manifest is `stable-embedding-large-manifest.json`, and its launcher log
explicitly reports all ranks done with `rc=0`. The no-fuse experiment in
`ovl04-nofuse-20261016-01/` exhausted its 240-second first-JIT window and
returned `124`; it supplies no no-fuse numerical conclusion.

## COMM-04 independent legacy P2P/IPC control

On the same physical GPU 2/3 pair, independent PyTorch 2.6.0+cu124 and NCCL
2.21.5 with `NCCL_CUMEM_ENABLE=0` and unset P2P disable/level variables
selected `P2P/IPC`. The 64-element oracle stalled at host readback and the
75-second bound returned `rc=124`; the owned processes were cleaned.
The explicit `NCCL_P2P_DISABLE=1` control selected `SHM/direct/direct`,
returned `rc=0`, and both ranks reported analytic max abs zero. Turning off
CUMEM alone therefore does not resolve the host's sampled P2P transport
failure. It remains outside the Jittor reducer and does not identify the
underlying NCCL, driver or interconnect fault. COMM-04 and INT-07 remain FAIL.

The retained unversioned run directories are
`$JITTOR_LAB_ROOT/_state/jittor-native-multicard/comm04-torch-cumem0-gpu23-20261016-02/`
and `comm04-torch-shm-gpu23-20261016-03/`. No other user's process was stopped.

## Final production r2 acceptance (2026-10-16)

Owner: native autograd/distributed maintainers. Checkout is the preserved
`48de31b4157fa128c5e1e19b052fefa3be7ed098` working tree; the original validated
baseline remains `3369809003c15cfcb1d843b81c1ed7027c204c97`. A fresh fetch again
records `origin/2.0-refactor@fcc5515d29abecde1fff8275c9141235e7fbf514`;
integration is deferred under the user's acceptance-first order. Source changes
remain uncommitted. Review when the kernels, autograd/reducer, precision policy,
model, transport or upstream baseline changes.

**OVL-04 now passes on the tested SHM fallback.** Production FP32 linear-bias
backward uses fixed tiles and a unique final writer per channel. Dense FP32
embedding backward validates and canonicalises original indices, stably sorts
them with CUB, and sums segments in fixed order. The final r2 embedding path
uses 64-position partial chunks for counts of at least 256 and a second kernel
to finish each real token segment. The finish kernel reads only segment-start
and chunk-boundary partials, each with a unique writer; its explicit tensor
input retains the scratch and orders execution. This avoids the first
candidate's long serial sum for repeated tokens. Both custom reductions retain
their differentiable transpose. Optional embedding semantics and other dtypes
use their existing implementations. No global reduction flag, epsilon change,
near-zero clipping or tolerance relaxation is involved.

Default `para_opt_level=3`, FP32 `atol=rtol=2e-6`, fused AdamW `eps=1e-8`,
model/data/global batch and update counts remain fixed. The full model is still
12 layers, hidden 768, 85,842,688 parameters, sequence 512 and global batch 8,
with three warmup and four recorded updates. Physical GPUs 5/9 were used for
the two-rank full model and timeline; GPUs 5/7/8/9 for the four-rank full model.
Every launcher returned `rc=0`. Resource checks, actual channels, rank logs,
runtime switches and isolated cache settings are retained with each manifest.
All compute used real CUDA and native NCCL with explicit
`NCCL_P2P_DISABLE=1`, `JT_BUILD_USE_MPI=0`; MPI and the Torch frontend are absent.

| Final r2 validation | Result |
| --- | --- |
| Full model, 2 and 4 ranks | Final parameters, local/check gradients and all four per-update parameter/gradient/AdamW first/second-moment comparisons have max abs 0. Replicas are identical. |
| Small model, 2 ranks, 3 updates | Cross-path final and recorded parameter/gradient/moment differences are 0. |
| Fresh processes, 2 ranks, 3 updates | sync/sync, readiness/readiness and sync/readiness parameter snapshots all differ by 0. |
| World size 1, 3 updates | Ordinary autograd fallback passes; final parameter max abs `1.3214263390182168e-6`, frozen tolerance ratio `0.6607128268043846`. Gradient/moment comparisons also pass. This does not claim distributed overlap at world size 1. |
| Embedding/linear edge model, 2 and 4 ranks | Two microbatches, first `no_sync`, clip 0.1, rank-only/unused/empty parameters: restore exact; continuation parameter/optimizer-state max abs 0; replicas exact; embedding gradients finite/nonzero. |
| CUDA NN tests | 9 passed: 8 stable-gradient oracle/boundary/derivative tests and 1 existing linear test, 0 skipped. |
| Native core and focused DDP | 154 passed, 10 skipped, 2 xfailed; focused DDP 26 passed. Skips are not CUDA acceptance. |
| Affected structure | 24 passed, including inventory entries for all four new backend gradient classes. Repository layout and diff whitespace checks pass. |

The full rank-wise collective sequence matches within each path. Across paths,
readiness adds one 16-element metadata signature per backward: eight such
collectives accompany the 128 gradient-bucket collectives in the full workload.
Filtering only these known signatures gives exactly the sync bucket sequence.
Buffer ownership is retained through communication join, with optimizer gradient
holders published afterward. Independent inspection found no missing partial
writer, uninitialised scratch read or missing graph edge in the final kernels.

Two harness issues are recorded separately from product behavior. The original
world-size-1 capture assumed a readiness callback even though that path correctly
uses ordinary autograd; its `IndexError` and failed manifest remain, and the v7
capture retry passes. An existing bucket mock identified gradients by optimizer
holder identity before those holders were published. Its reproduced failure was
fixed by testing the actual collective inputs and their pre-group readiness.
The focused 26-test DDP regression then passes.

### Final candidate device timeline and memory

The final candidate was profiled in one process per rank with the same initial
state and generated inputs for sync/readiness. Per-update CPU tensor snapshots
and local-gradient capture wrappers are disabled. The analyzer selects each
trace's unique process, matches the last four of seven backward markers to
updates 0–3, and preserves nanosecond intervals. Summing the intersections of
the GEMM and NCCL interval unions within each rank/update window gives:

| Window scope, summed over 8 rank/update windows | NCCL union | GEMM/NCCL intersection | Intersection / NCCL |
| --- | ---: | ---: | ---: |
| Sync, backward marker | 0 ms | 0 ms | Undefined: no NCCL in these windows |
| Sync, complete update | 1959.032718 ms | 0 ms | 0% |
| Readiness, backward marker | 106.697928 ms | 47.547299 ms | 44.5625% |
| Readiness, complete update | 1940.726922 ms | 1176.396668 ms | 60.6163% |

All eight readiness backward windows contain real device overlap before the
backward call returns. The host marker gives a bounded pre-return measurement;
lazy work outside that marker belongs to the complete-update accounting.
The synchronous negative control has zero device intersection even over whole
updates. Final parameters and check gradients still agree exactly in this
profiled run. This closes the final-candidate timeline requirement.

Maximum rank time is 0.486969645 s/update for sync and 0.358687391 s/update for
readiness, corresponding to 8411.2 and 11419.4 tokens/s. These are one profiled
shared-load observation; they do not establish a universal speedup or a new
Jittor/PyTorch throughput comparison. Prior independent PyTorch baseline evidence
remains under OVL-02 and is not mixed into this timing ratio.

Measured allocator used peaks are **3,797,251,072 B (3.54 GiB) for sync and
8,402,551,808 B (7.83 GiB) for readiness**, about 2.21 times. Reserved peaks
are 3,880,779,776 B and 8,733,589,504 B. Both paths finish near 1.37 GB used.
Owned-process NVML sampling reports 8988 MiB per rank over the full profiled run;
it includes profiler/context overhead and both variants, and is not a
per-variant allocator figure. Other users' GPU allocations are excluded.

Static inspection identifies extended intermediate-gradient liveness during
callback execution: `grad_impl` holds all intermediate `VarPtr`s while callbacks
already execute backward, whereas the sync path returns and releases these
extra roots before execution. Communication inputs/outputs are also held until
join. This is an implementation explanation to test, not a measured allocation
breakdown. The run does not establish a leak. The explicit follow-up is
`KI-NATIVE-DDP-MEM-001`: use a clean delayed-callback control and current-used
sampling to isolate the holds before changing core graph ownership. Reducing
this peak must preserve higher derivatives, frozen multi-step tolerances,
accumulation/resume, safe communication storage and actual overlap.

### Stable-kernel performance and evidence index

The first production embedding candidate took 4.88–7.08 ms for the long
all-same-token segment. Final r2 reduces that path to 248.431 microseconds.
It still pays a cost for fixed summation order. Three-trial median CUDA device
kernel plus memset times (20 calls per trial, compilation excluded) are:

| Backward shape | Existing path | Final production r2 |
| --- | ---: | ---: |
| Embedding, 2048 positions × 768 channels, sparse token repeats | 64.484 us | 161.801 us |
| Embedding, 4096 × 768, all same token | 116.342 us | 248.431 us |
| Linear bias, 2048 × 768 | 11.490 us | 24.911 us |
| Linear bias, 2048 × 2304 | 25.594 us | 32.184 us |
| Linear bias, 2048 × 3072 | 29.051 us | 36.138 us |
| Linear bias, 4 × 768 | 5.285 us | 2.758 us |

Each scenario also passes its FP64 oracle. These figures sum CUDA kernel and
memset durations, rather than timing Python graph construction. Chunk scratch
adds 12 MiB for 4096 × 768 FP32 positions. The fixed-order kernels improve the
observed numerical reproducibility; they are not universally faster operators.

All paths below are unversioned, under
`$JITTOR_LAB_ROOT/_state/jittor-native-multicard/`:

- `ovl04-v2-20261016-01/production-r2-numerical-acceptance.json`: assertion-backed
  aggregate of current source hashes, per-rank comparisons, collective order,
  fresh-process repeats, recovery, timeline and memory; the collector is
  `$JITTOR_LAB_ROOT/jittor-native-multicard/collect_production_r2_acceptance_20261016.py`.
- The same directory's `production-r2-*-manifest.json`, `results/`, and
  `production-r2-fresh-repeatability-comparison.json` retain actual commands,
  launcher return codes, rank results and parameter snapshots. Old r1
  `production-numerical-acceptance.json` remains a historical artifact.
- `ovl04-v2-20261016-01/production-r2-timeline/`: rank `.nsys-rep`/SQLite files,
  `backward-overlap-summary.json`, `manifest.json`, measured allocator JSON
  and `nvml-owned-pid-samples.json`. Reproduce with the retained
  `run_production_r2_timeline_20261016.py` and
  `analyze_production_overlap_20261016.py` lab scripts using a new destination.
- `ovl04-production-unit-20261016-02/`: final candidate SHA256, nine-test CUDA
  log/result and 24-test structure log/result. `ovl04-production-unit-20261016-01/`
  retains `native-core-r2*`, `native-ddp-r2.log`, and
  `embedding-edge-r2-manifest.json` with the two/four-rank recovery artifacts.
- `ovl04-production-perf-20261016-01/production-r2*`: kernel/memset raw timeline,
  extracted device times, FP64 oracles and manifest.
- `ovl04-v2-20261016-01/matrix-before-r2-acceptance.json` preserves the previous
  matrix; numeric tolerances, comparison protocol, case IDs and the two P2P
  failures were unchanged when publishing the new statuses.

The current matrix is **69 PASS, 2 FAIL, 1 UNSUPPORTED**. OVL-04 is closed for
the stated configuration; COMM-04/INT-07 and the measured readiness memory cost
remain visible. Full 8.27 and unconditional production release are open.

## Readiness memory follow-up (2026-10-16)

Owner: native distributed maintainers. Checkout remains
`48de31b4157fa128c5e1e19b052fefa3be7ed098`, with the preserved uncommitted
native DDP work. `origin/2.0-refactor@fcc5515d29abecde1fff8275c9141235e7fbf514`
was fetched again and not merged, following the user's acceptance-first order.
The optimizer implementation SHA256 is
`9b5e412f0dc24e7e53fc54ecbf8c12dfbd8d08f160c0be0d9fa7e47b8228cd61`.
P2P investigation is explicitly deferred; all new runs use native NCCL SHM
fallback, with MPI and the Torch frontend disabled.

The old readiness callback evaluated backward kernels while `grad_impl` still
owned intermediate-gradient roots. Those temporary roots kept already-consumed
buffers alive until graph construction finished. A clean delayed-callback
control reduced the used peak from 8,402,551,808 B to 3,809,867,264 B, locating
the avoidable lifetime cost without changing C++ autograd ownership.

The production optimizer now records `(target_index, gradient)` in the original
C++ readiness order, lets backward graph construction return, then replays
that order into the existing bucket reducer. A local `auto_flush_ops=0` scope
covers both return conversion and replay. Without this scope, conversion of
the returned gradients to Python holders can automatically submit all pending
backward roots before the first bucket, eliminating device overlap. That
unsuccessful prototype is retained as a negative control. The general
`grad_with_callback` API, higher derivatives, collective ordering, frozen dtype
tolerances and AdamW epsilon are unchanged. Communication buffers remain held
through join, and optimizer gradient holders are published afterward; the local
record list is cleared on success and exceptions.

The production memory capture uses the same 12-layer, hidden-768, 85,842,688
parameter Transformer, sequence 512, global batch 8, three warmup and four
measured updates, default `para_opt_level=3` and fused AdamW `eps=1e-8`.
Per-update CPU snapshots and local-gradient capture are disabled, and allocator
peaks are reset after warmup. Both ranks report the same values:

| Allocator measurement | Synchronous | Current readiness |
| --- | ---: | ---: |
| Peak used | 3,797,251,072 B | 3,809,867,264 B |
| Peak reserved | 3,880,779,776 B | 3,931,111,424 B |
| Used at end | 1,374,663,680 B | 1,376,761,344 B |

Readiness now adds **12,616,192 B (12.03 MiB, 0.332%)** to the synchronous
used peak, and is **54.658% below the former readiness peak**. This is a
measured `retain_graph=False` bound for this workload; retaining graphs and
other model shapes still need their own memory sizing. Reserved allocator
memory is reported separately from live tensor memory.

The final production Nsight capture retains actual CUDA kernel intervals and
launch correlations. Its original full-update accounting finds **1171.191249 ms
of GEMM/NCCL intersection / 2026.419778 ms of NCCL = 57.7961%**, with positive
intersection in all eight rank/update windows; the synchronous negative
control has zero intersection. The eight metadata signature collectives have
137.718504 ms duration and zero GEMM intersection. All intersection comes from
the 128 real gradient buckets; their separate payload-only denominator is
1888.701274 ms, giving 62.0104% as a supplementary metric.

All eight final host backward windows contain a payload intersection before
the Python call returns, totaling 12.303988 ms. The first production capture,
also preserved, hit seven of eight host windows and all eight device updates;
its rank 1/update 3 payload intersection occurred after host return.
Each update still has 139 GEMMs after its first real gradient bucket starts;
all 139 launches correlate to that update's backward host range, on compute
stream 31 while NCCL runs on stream 33. For this decoder without activation
recomputation and with a GEMM-free fused optimizer, this establishes actual
backward/communication overlap even when device work outlives the host range.
Host NVTX return is therefore not used as the device backward end. Both captures'
host-window figures, including the earlier missed window, remain in the evidence.

Maximum rank time in the final profiled shared-load run is 0.703344042 s/update
for sync and 0.390110499 s/update for readiness (5823.6 versus 10499.6 tokens/s).
These paired observations do not establish a universal speedup or a new
Jittor/PyTorch comparison, and cannot be compared directly with old r2 wall
times recorded under different concurrent load.

All new unversioned artifacts are under
`$JITTOR_LAB_ROOT/_state/jittor-native-multicard/readiness-memory-20261016-01/`.
`production-memory-final/` retains the no-wrapper final production manifest,
source/harness snapshots, rank `.nsys-rep`/SQLite files, result JSON,
`overlap-summary.json` and the independent `payload-overlap-audit.json`.
`production-memory-timeline/` preserves the first production capture.
`baseline-delayed-control/`, `ordered-delay-prototype-timeline/` and
`ordered-delay-no-autoflush-prototype/` retain the diagnostic controls and are
not substituted for production results. The lab drivers and analyzers use
guarded destinations and preserve all previous artifacts.

### Final memory-repair numerical and recovery acceptance

Status: accepted for the measured native CUDA/NCCL SHM path. The final
`production-acceptance-final/` manifest contains four completed launchers, all
with `rc=0`: full Transformer and accumulation/recovery edge cases at both
two and four ranks. These runs use the production optimizer without callback
substitution. Source hashes agree with the final memory/profile capture and
the four low-precision recovery runs; the assertion-backed aggregate checks
them against the current working tree. FP32 `atol=rtol=2e-6`, AdamW `eps=1e-8`
and default `para_opt_level=3` remain unchanged.

| Final-source check | Result |
| --- | --- |
| 85.84M Transformer, 2 and 4 ranks | Four recorded updates after three warmups. Cross-path parameters, gradients and AdamW first/second moments have max absolute difference 0 at every recorded update. Final local/check gradients also differ by 0; replicas are exact. |
| Full-model collective order | Rank sequences agree. Sync has 128 payload collectives; readiness has the same 128 payloads plus eight known 16-element signatures. Removing only those signatures gives the identical sequence. |
| Accumulation/recovery, 2 and 4 ranks | Two microbatches, first `no_sync`, clip 0.1, AdamW, embedding/linear and rank-only/unused/empty parameters. Model/optimizer restoration is exact; the next update's parameters and optimizer state, and rank replicas, differ by 0. All eight bucket callbacks occur in the expected order. |
| Native CPU/autograd/DDP regressions | 221 passed, 11 skipped, 2 xfailed; includes 28 focused DDP tests, callback timing, local flag restoration, `retain_graph=True` with subsequent differentiation, and communication join after an injected later-bucket failure. Skips/xfails are not device-support evidence. |
| Affected structure and repository checks | 23 structure tests passed; layout and diff whitespace checks returned 0. |
| Native BF16 serialization | Four focused CPU tests and two real-CUDA tests passed, with no skips. |

The final two-rank precision suite performs three updates, persists a
checkpoint, restores it and compares the next update with the uninterrupted
trajectory in that precision. Each launcher returns `rc=0`. Gradient errors
below use the independent global-batch reference and the original dtype
tolerances; cross-precision trajectories are not compared for equality.

| Precision | Maximum global-batch gradient error | Resumed parameter / replica difference | Resumed optimizer state |
| --- | ---: | ---: | --- |
| FP16 | `9.5367431640625e-7` | 0 / 0 | Exact |
| BF16 | `3.814697265625e-6` | 0 / 0 | Exact |
| AMP FP16 | `4.76837158203125e-7` | 0 / 0 | Exact; scaler scale 64 |
| AMP BF16 | `3.814697265625e-6` | 0 / 0 | Exact |

Pure FP16/BF16 have three gradient buckets plus one signature per backward;
AMP FP16/BF16 have six plus one. All three pre-checkpoint updates have the
expected collective count. Checkpoint model and optimizer state restore
exactly in every mode.

### BF16 checkpoint defect found during recovery validation

The first pure-BF16 recovery run failed because an ordinary BF16 `Var` was
serialized through `.data`, which exposes raw `uint16` storage to NumPy.
Loading reconstructed a `uint16` Var and lost its BF16 dtype. Model Parameters
retain their own dtype and adaptive moments are FP32, so prior model/moment
checks did not expose the optimizer gradient holder. Subsequent bucket
unpacking interpreted the restored bits incorrectly, with errors around
47,750. The same failure reproduces with the previous callback scheduling and
with the memory repair, establishing that this defect predates the scheduling
change. The original minimal CPU regression failed before the repair.

`python/jittor/_core/var.py` now serializes BF16 values with their explicit
dtype and restores that dtype independently of the ambient AMP policy. Other
dtypes retain their existing reducer. Bucket unpacking also rejects gradient
holders whose dtype differs from the reduced bucket, preventing silent
reinterpretation. Old checkpoints containing raw `uint16` without BF16 dtype
metadata remain `uint16`; the loader does not guess a conversion. A bad legacy
holder is rejected at unpacking rather than silently used. Correctly typed
new checkpoints pass the native two-rank BF16 recovery test above.

The native BF16 tests cover pickle protocols, AMP scopes, nested nonzero
positive/negative optimizer gradients, file save/load, other dtypes and legacy
uint16 compatibility. Real-CUDA verification checks nontrivial arithmetic on
the restored tensors. One intermediate test incorrectly assumed that a second
alias remained on CUDA after `.numpy()` moved their shared storage to CPU;
the residency probe, failed assertion and corrected test are all retained.
No production change was made to accommodate that assertion.

### Final evidence and scope

The canonical unversioned acceptance record is
`$JITTOR_LAB_ROOT/_state/jittor-native-multicard/readiness-memory-20261016-01/production-memory-acceptance.json`.
It binds source hashes, all four final numerical/recovery launchers, precision
results, both device-overlap accounting methods, allocator measurements and
regression commands/return codes. Recreate the checks with
`$JITTOR_LAB_ROOT/jittor-native-multicard/collect_readiness_memory_acceptance_20261016.py`
using a fresh destination if collecting another revision. Additional evidence
in the same run directory is:

- `production-acceptance-final/`: full-model and edge-case manifests, rank
  logs, per-update comparisons, snapshots and checkpoint continuation results.
- `production-precision-final/`: all four final precision modes and their
  source manifest. `production-precision/`, `bf16-legacy-diagnostic/` and
  `bf16-current-diagnostic/` retain the initial failure and both scheduling
  controls.
- `bf16-pickle-repair/final-summary.json`: before/after native serialization
  reproductions, focused CPU/CUDA checks, source hashes and the test correction.
- `verification.json`, `native-final.log` and `structure-final.log`: commands,
  source/test hashes, results and explicit skip/xfail counts.
- `matrix-before-memory-acceptance.json`: the matrix before publishing the
  memory follow-up. All 72 IDs, numeric tolerances and comparison protocol
  remain frozen.

This closes the measured readiness peak-memory issue and preserves OVL-03
through OVL-06 acceptance on explicit SHM fallback. The matrix remains
**69 PASS, 2 FAIL and 1 UNSUPPORTED**. COMM-04/INT-07 retain their historical
transport failures with investigation deferred by the user; they are not
reclassified as passes or attributed to the Jittor reducer. Other shapes and
`retain_graph=True` do not inherit the measured 0.332% memory bound. Latest
remote compatibility and unconditional production release remain outside
this completed memory-fix acceptance; no merge, commit, push or PR was made.
