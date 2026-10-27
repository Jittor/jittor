# Native sharded checkpoint and training-resume verification

- Status: native checkpoint I/O and CPU / single-host two-CUDA-rank resume pass within the scope of this report; training FSDP / TP / ZeRO and real multi-host remain unaccepted.
- Date: 2026-10-15.
- Baseline: initial `a1a3f44afdcec9d0ce3670ff77e78c09f16b80a2`; after stopping tests, preserving all work and fetch / fast-forward, final baseline `8e388d108eb83c1c438936709f29855f47ba2a69`.
- Owner: native serialization, distributed and optimizer maintainers.
- Review when: the native Module / Optimizer / RNG protocol, Store, tensor placement, BF16 storage, NCCL or the rank/world layout changes.

Added the `jittor.distributed.checkpoint` public entry and a native serialization leaf codec,
without importing Torch/compat and without copying FSDP or optimizer mathematics. Each real rank writes its own
payload; the Store exchanges only metadata and state, and the manifest is published atomically after complete verification.
`save_training_state` / `load_training_state` consume the existing Module, Optimizer and native RNG
owners, saving FQNs, trainability, shape/dtype, optimizer class and ordered bindings, hyperparameters,
moment / step, Python / NumPy / native CPU / CUDA RNG, and additional state.

Model reading goes through the existing native Module.update owner, which inherits the source's gradient flag;
therefore the detached tensor returned by generic I/O cannot be used directly for training resume. The helper saves separately
and restores the flags on the load source; the save process does not change the original model's flags and does not modify the Module owner.
Before restoring it checks all payloads and the FQN/shape/dtype/flags/optimizer bindings and state slots,
and an invalid RNG also rolls back the current random stream. Both the original and the new model, and both the frozen bias and the trainable weight, enter the actual resume training.

Native BF16 `numpy()` exports FP32, so the export dtype cannot be used to infer parameter or momentum storage.
The codec saves and restores the original BF16 bits through a UInt16 reinterpret view, including signed zero,
non-finite values, NaN payloads and subnormals. The training fixture also asserts that parameter / native SGD
momentum are actually BF16; it uses exactly representable dyadic inputs, and an independent NumPy FP32 SGD checks each step
that the low 16 bits are zero, then compares the complete BF16 raw bits, momentum, actual gradients, loss and counter.

This round also fixed a native optimizer MPI-free world=1 NCCL legacy break: the distributed
state was true but the historical Var.mpi_all_reduce alias was not installed. MPI-free synchronization now thinly forwards to the existing collective
owner; real MPI keeps the original method call, DDP-managed parameters are still excluded, and the singleton owner does not self-assign when it returns
the original tensor. Averaging, parameter broadcast and update formulas are unchanged.

## Final verification

Python 3.11.16, NumPy 1.26.4, pytest 7.4.4, with an independent environment and source path; the shared
Python environment was not modified and no new downstream library was installed. Two NVIDIA RTX 4090 cards, nvcc 12.2.140 / GCC 11.5,
NCCL `21803` actually loaded, MPI disabled. The CPU, each CUDA rank, the launcher and the two core modes have independent
caches; the first core / NCCL / BF16 compiles are warmed in order. The final cumulative fallback for both ranks was 0,
the whole process asserted that Torch was not imported, and the math reference is an independent NumPy implementation.

| Verification | Actual result |
| --- | --- |
| native CPU checkpoint / exact resume / frozen parameters / BF16 / rejection paths | 21 passed, 0 skipped |
| Real host TCPStore two-client metadata exchange | included in the 21 items above, public wait(timeout) + get(key) contract |
| NCCL world=1 unmanaged / DDP optimizer | 2 passed, 5-step independent NumPy Adam |
| Final single-host two-rank native optimizer + checkpoint | 3 passed per rank, 0 skipped, launcher rc 0 |
| Two updates after existing / fresh model restore | all models, m/v, gradients, loss and step exactly equal |
| Next draw of Python / NumPy / native CPU / CUDA | all exactly equal per rank |
| native BF16 parameter / BF16 momentum SGD 3-step save + 2-step resume | CPU and real CUDA pass, raw bits equal to the NumPy dyadic trajectory |
| CPU serialization / optimizer / Store neighbor regression | 35 passed / 2 Torch-mode skips / 1 xfailed |
| CPU core in both modes | 322 passed / 42 skipped / 1 xfailed, both modes exit 0 |
| Full structure | 1451 passed / 6 skipped / 2 central-document-registration failures |
| Layout / whitespace | pass |

The two full-structure failures are only that this round's new note / report is not yet in the corresponding toctree and that the generated
MANIFEST still awaits central registration; no claim is made that the whole structure gate passes. The actual violations of the test-helper dependency and optimizer
communication ownership have been fixed, the targeted run is 8 passed, and the full re-verification above was completed.

The independent FP64 NumPy reference for CUDA FP32 Adam uses the native epsilon-before-bias-scaling
convention; it differs from `torch.optim.Adam` in the epsilon position, and the mathematics was not aligned arbitrarily.
Maximum absolute difference over five steps: parameters `1.3598102172052151e-8`, moments
`5.952429563138573e-9`, actual mean gradient `2.5982546839747878e-8`, and loss per rank
`4.183892484202478e-9` / `4.018982271980498e-9` respectively.
The threshold remains `2e-6 + 2e-5 * abs(reference)`; the resume's own trajectory uses value-by-value equality, without relaxation.

Before NumPy readback, the CUDA-computed prediction, loss, gradients and updated parameters/moments are
synchronized and asserted to be device-resident; CUDA driver pointer attribute 9 verifies that the real allocation
belongs to that rank's logical device 0. The explicit host snapshot of `numpy()` turns residency
into CPU while keeping the logical placement; after load / I/O the CUDA placement is materialized through the existing device pointer accessor
and the pointer re-checked, so the host snapshot cannot be mistaken for a compute fallback.

The two ranks also actually verified flat-interval reassembly and local restore for different local inputs, replicated
metadata inconsistency, and failure of only one rank writer; the latter two raise on all participating ranks and leave no complete marker.
CPU checks cover missing/corrupt payload, FQN/group order, flags, shape/dtype, moment,
RNG payload/backend rejection with the target parameters/flags/LR/random stream unchanged; a failed overwrite
keeps the old complete manifest, and overlap/gap are rejected.

## Boundaries and reproduction

`TensorShard` describes the element interval of the checkpoint and is not responsible for training sharding. Ordinary inputs are
replicated: each rank takes a complete host snapshot, checks the full digest, then writes its own interval;
this makes no bounded host-memory promise. Only an explicit TensorShard consumes the existing local input,
and local restore allocates only the requested output range. The source/tests do not measure GPU memory peaks or performance, and draw no L4 conclusion.

The training helper supports only the completed optimizer step / zero_grad boundary and explicitly rejects pending
accumulation; it strictly checks the same rank set / world, the execution backend and the number of visible CUDA RNG devices.
World-size RNG migration, full transactionality of custom load hooks, multi-node, native FSDP / TP /
ZeRO, ROCm/NPU/Windows, actual MPI regression and power-loss/worker-death tolerance are outside this pass.
NPU storage is explicitly rejected, and the bridged HCCL branch grants no hardware acceptance.

The public API and the shared-filesystem / Store / generation cleanup rules are in the
[mechanism note](../notes/native-sharded-checkpoint.en.md). Entry points:

```bash
JITTOR_TORCH_SHIM=0 JT_BACKEND=cpu JT_USE_CUDA=0 JT_BUILD_NVCC_PATH= \
  PYTHONPATH=python <jittor-python> -m pytest -q tests/serialization/test_sharded_checkpoint.py
# world=1 使用 JT_NCCL_WORLD_SIZE=1 的真实 CUDA 环境，运行 optimizer_sync 文件。
# 双 rank 使用现有 launcher，每个 child 在导入前按 RANK 指定独立 cache/home。
<jittor-python> python/jittor/distributed/launch.py --backend nccl -n 2 \
  --device-ids=<two-gpus> -- <jittor-python> <isolated-rank-wrapper>
```

Set `JITTOR_CHECKPOINT_TEST_DIR` to a new shared path; the rank wrapper in the current worktree
invokes `pytest -q -s tests/backends/comm/nccl/test_native_optimizer_sync.py
 tests/backends/comm/nccl/test_native_sharded_checkpoint.py`. The shim is off, CUDA kernel math is
strict, real NCCL include/lib, fallback error scope. The NCCL timeout is constrained jointly by the Store and this API; the signature of Store.get does not accept a timeout, so the public contract of wait should be used.

The raw records, rank wrapper, manifest/payload, failed partial generations and independent caches are in
`$JITTOR_LAB_ROOT/_state/native-sharded-checkpoint/`, unversioned. The final key logs are
`logs/cpu-neighbors-8e-owner.log` (containing all 21 new tests), `logs/cuda-world1-optimizer-8e-owner.log`,
`logs/two-rank-8e-owner-launch.log`, `logs/two-rank-8e-owner/rank{0,1}.log`,
`logs/core-cpu-8e-owner.log` and `logs/structure-8e-owner.log`. The initial a1 baseline success log
`logs/two-rank-tcp/rank{0,1}.log` is kept separately; the final baseline re-run did not reuse the old pass markers.

## Main-branch integration re-verification

On `b930461562e73c67332581fe2384eaf8cfd7d838`, the subtask commit
`82fb93150fc51e0608db7f95e08f8d48bb7f78ec` was integrated, the toctrees of the two documents were completed and the
MANIFEST regenerated. The eight Python source/test files are byte-for-byte identical to the subtask version that completed the dual-card verification; the main branch's
other changes relative to its original measurement baseline are only the Transformers report and the skill documents.

After integration the CPU core is 322 passed / 42 skipped / 1 existing xfailed, the new checkpoint
CPU tests are 21 passed / 0 skipped, and the full structure is 1453 passed / 6 skipped / 0 failed.
Layout and whitespace checks pass. The central-registration failures above are fixed as of this point; the initial results are kept to distinguish the two verification rounds.
The raw integration logs and the source SHA256 comparison are in
`$JITTOR_LAB_ROOT/_state/native-checkpoint-integration/`, unversioned.
