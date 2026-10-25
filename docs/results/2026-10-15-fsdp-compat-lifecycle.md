# FSDP optimizer view lifetime and native checkpoint dispatch

- Status: bounded CPU and two-rank CUDA/NCCL correctness verified.
- Measured source: `e5260eb1aa48a95d8106ba9901ec54d850f31dac`, clean throughout
  the final CPU, single-CUDA and dual-rank CUDA runs.
- Date: 2026-10-15. Owner: Jittor compatibility maintainers.
- Review when: Adam parameter-cache alignment, FSDP view refresh, checkpoint
  dispatch or the installed Module loading contract changes.
- Scope: the two compatibility failures recorded in the
  [native FSDP report](2026-10-15-native-fsdp.md). Its original source, raw
  tensors, numerical thresholds and BF16 limitation remain separate evidence.

Parent integration: the delivered code and dependencies were applied on
`71c5be74ef6991b75fa79d73fe52eb04c0319d59` alongside the shared BF16 Linear
and TP changes. All delivered source files matched their final references.
The lifecycle manifest's three documents, 38 logs and seven runners were
rehashed successfully. Combined CPU compatibility owners passed **18** with
one CUDA-only class skip; actual CUDA compatibility owners passed **24** with
zero skips. CPU core passed **322**, and full structure passed **1470** with
six skips after central document registration. Raw parent logs and audits are
unversioned in `$JITTOR_LAB_ROOT/_state/distributed-bf16-integration/`.
Earlier dual-rank results retain their measured source and do not certify
native FSDP's separate unresolved BF16 numerical difference.

## Root causes and changes

The reversed-mesh custom-Adam regression passed its first numerical update but
retained both superseded shard objects. Actual Python referrer inspection on
each CUDA rank traced them to the parameter group's `_torch_adam_params` tuple.
The same stale identity cache could reset moments and per-parameter steps on
the next update.

The existing Adam cache synchronizer now accepts explicit old/new views of the
same logical parameter. FSDP refresh supplies only its actual entry replacements;
the synchronizer still applies its identity-based reordering and fresh-slot
initialization. Local tests check immediate weak-marker release without cyclic
GC, moment/variance object identity, step continuity, user reordering, fresh
ordinary parameters, state selection and two actual Adam/AdamW updates. The
existing CUDA regression independently checks three full-parameter Adam updates
against NumPy across a reversed two-rank mesh.

The checkpoint dispatcher previously passed `strict=` directly to the native
`Module.load_state_dict`, which lacks that argument. When that exact native
method is still bound, dispatch now uses the installation's Torch Module loader.
Custom overrides retain authority. Strict-key checks, shape checks, dtype and
allocation-preserving copying stay with that existing loader; tests check
pre-mutation failures, copying aliases and pointers, trainability and gradients.

The nested-FSDP regression also supplied a full plain-tensor dictionary with the
default `full_state_dict=False`, while expecting rank-zero broadcast. An
independent binary Torch 2.11.0+cu128 two-rank run rejects that input on both ranks
with a mixed Tensor/DTensor loading error. The same real model restores its bias
to seven and executes its forward on both ranks with explicit
`StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)`. The
repository node now requests those options and reuses the existing FSDP full
state loader. Dispatch does not infer full state from tensor shapes.

## Source and environment

The branch fetched remote `2c0d2681a7db089aa2e996f0ad8e15057fb8e35e`, integrated
as `361e49578cbc986d8a6a467cf66e1f09838f5d06` before edits. It then preserved local
changes outside the checkout while integrating these complete dependencies:

- `a6b943d8ca6d6ae31936b152ee53905ebbd61dba`, tensor-list clearing owner;
- `7b8230c19ab88f2d182233a24da231b622d77b6f`, storage-preserving collective and
  Module loading owner.

Their local equivalents are `db8ba7bec` and `c33b72784`. They are prerequisite
integration for this report; this report does not grant acceptance to their
separate downstream-engine work. The fix does not copy their loading mathematics.

Candidate execution used Python 3.11.16, NumPy 1.26.4, CUDA 12.2.140 and NCCL
2.18.3. CPU runs explicitly disabled CUDA visibility, CUDA execution and nvcc.
CUDA runs used real assigned devices, strict kernel math and separate warmed
per-rank caches. Final single- and dual-rank runs set `backend_fallback=error`
and verified the global fallback counters stayed zero. First compilation was
serial. Existing unrelated device occupancy excludes performance conclusions.
The independent reference records its actual Torch `_C` binary origin.

## Results and limits

| Check | Result | Source boundary |
| --- | --- | --- |
| Final CPU loading/optimizer/collective neighbors | 49 passed, zero skipped | Clean measured source |
| Final single-CUDA equivalent | 49 passed, zero skipped; global F0 | Clean measured source |
| Existing two-rank FSDP CUDA nodes | 8 passed per rank, zero skipped; global F0 | Clean measured source, real NCCL |
| Independent binary Torch full-state option contract | Default rejected on both ranks; explicit full/broadcast passed on both | Actual binary Torch, isolated interpreter |
| CPU core in native/Torch modes | 322 passed, 42 skipped, 1 expected failure | Same runtime bytes before source commit |
| Native FSDP + TP + ZeRO CPU combination | 51 passed, zero skipped | Same runtime bytes before source commit |
| Native process-group CPU / compatibility FSDP neighbors | 12 / 20 passed | Integrated dependency bytes |
| Collection contracts plus seven new local nodes | 41 passed, zero skipped | After relocating the Torch test |
| Full structure before that relocation | 1469 passed, 4 skipped, 3 failed | See exact resolution below |

The optional Accelerate node is explicitly deselected in the final 49-node
sessions. A preceding CPU run recorded its missing-dependency skip. The two-rank
selector explicitly excludes `test_two_dimensional_mesh_hybrid_gradient`, which
requires at least four ranks. The initial full nine-node run recorded eight
passes and that skip on each rank, and correctly exited nonzero under the strict
execution policy. Neither exclusion is accepted as coverage.

The structure run found a new Torch test placed under the native collection
path; moving it to `compat/tests/torch/test_fsdp_optimizer_rebinding.py` fixed
that contract, verified by the subsequent 41-node run. The two remaining
failures concern central toctree and generated MANIFEST registration for the
already delivered native-FSDP documents. Those files remain under the parent
integration owner's control. A full structure pass is not claimed here.

Ruff passes for the other five changed files. `optimizer_api.py` retains eight
existing F401/E701 diagnostics, reproduced from the unmodified prerequisite
source; this patch adds none. Whitespace and repository layout checks pass.

The new focused nodes are
`compat/tests/torch/test_fsdp_optimizer_rebinding.py` and
`compat/tests/torch/test_checkpoint_native_module.py`. Run them in separate
configured CPU and CUDA sessions with `JITTOR_TORCH_SHIM=1`, `PYTHONPATH=python`
and `JITTOR_TEST_DEVICES` set to the actual backend. CPU must explicitly disable
CUDA as above. The existing multi-rank entry is
`compat/tests/fsdp2/test_fsdp2_nccl.py`; use the
[distributed verification procedure](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md)
with independent rank caches and the recorded two-rank selector. The exact
external runners, environment files and selectors are bound by the artifact
manifest below.

## Raw evidence

Artifacts are outside the repository under
`$JITTOR_LAB_ROOT/_state/native-fsdp/20261015/artifacts/compat-lifecycle/`:

- `SOURCE.json` binds the clean source SHA and eleven relevant source hashes.
- `AUDIT.json` verifies final result protocols, global F0, source hashes,
  reference option behavior and the original native-FSDP seal.
- `SHA256SUMS.json` binds three metadata files, 38 raw logs and seven runners or
  environment files. SHA-256:
  `38a93c8ce1e8854d9f22034dc6e8a45dd9d52dd9f7bf5c464d8faef1b72500ac`.

The manifest retains repair-before/after logs, the failing strict nine-node
invocation, the original reference inspection and the final clean measurements.
The prior native-FSDP seal remains
`9901d1ed7dd834dae4d11b4afdb501a37d2cdc3bc692019daad2d6a0c27c9a27`;
all 236 original artifacts, 13 logs and five runners were independently rehashed
and still match. This follow-up does not reassign their baseline or change the
pending shared-Linear/BF16 numerical audit.
