# API ownership, Runtime state and installation integration

- Status: implementation integrated; validation scope recorded below
- Reviewed: 2026-09-08
- Owner: compatibility and repository maintainers
- Baseline: `7e83d6da4`
- Recheck when: an installer gains an API, a runtime owner changes, or packaging mappings change

This batch completes the named runtime-state contracts (5.24), installation
transactions (7.05), third-party adapter separation (7.07), and repository-tool /
resource-declaration layout (9.19). It also migrates complete API families toward
7.03. The entire 7.03 and independent Torch architecture (7.12) remain open.
The native Var/Op graph and seven meta-operators are retained; no second graph
or new backend optimization is introduced.

## Implementation

Tensor, NN, initializers, data loading, CUDA, utility/compiler, core misc,
optimizer/scheduler and distributed installer families publish stable function
and class implementations. Native delegates live in context-owned read-only
mappings. Required frontend Module subclasses reuse the existing class adapter;
their mathematical methods have physical module owners. Fidelity records retain
explicit limitations and can be rendered by `fidelity_table()`.

Activation and tensor-state owners use Runtime services. Leaf/retained/optimizer
root aliases are removed in both activation modes. Getitem lowering depth is
context-local. Native I/O/indexing/seed delegates and seed/precision policy no
longer use cross-module private root attributes. First lookup and owner binding
share the installation lock; weak owner tables do not retain unloaded frontends.

Outer activation can undo successful child installation, including modules,
classes, completed steps, cached tensor state and fidelity records. RuntimeHook
owns delayed vLLM and extension effects. Source imports restore owned entries
without deleting concurrent writes. Explicit and child-hook mutations are
deduplicated against later snapshots. Interrupts release locks, and rollback
does not turn an implementation NameError into an invented ownership conflict.

Transformers/TorchMetrics adapters live in a separate `adapters/` distribution.
Their entry points register exact module owners, validate supported versions,
and use the normal module finder. `builtins.__import__` remains unchanged.
An absent optional adapter is reported as unavailable, not applied.

Repository scripts now live in `tools/`. Per-distribution pyprojects own resource
declarations; generated MANIFEST files have drift checks. See the separate
[packaging evidence](2026-09-08-packaging-ownership.md) for artifact byte checks.

## Validation scope

Validation used isolated implementation worktrees and one integrated CPU/CUDA
cache per configuration. There was no full numerical suite or new core rebuild
for each small edit.

- Integrated legacy CPU batch: 179 passed and three failures, followed by a
  targeted 41-passed run fixing all three and covering the state/rollback
  regressions. The failures were wrong rollback exception classification,
  an obsolete scheduler import-failure probe, and missing legacy
  `Tensor.bincount` class binding. Three subprocess-heavy bootstrap cases were
  deliberately excluded from this batch, not counted as passed.
- Real CUDA, independent frontend: tensor residency explicitly checked on a
  visible CUDA device; tensor arithmetic, gradients, views, initializers, data
  loading and Linear training plus owner identity: 2 passed, zero skips.
- Real CUDA optimizer trajectories / checkpoint state restoration / scheduler
  behavior and core misc values: 7 passed, zero skips. SGD, Adam and AdamW match
  their numerical references; native update mathematics is retained.
- Family-local CPU checks covered independent and legacy API objects,
  dtype/shape/value behavior, Transformer templates and retry; the optimizer
  batch passed in both modes. Distributed ownership and collective parameter
  routing: 6 simulated nodes passed, without creating real communicators.
- Adapter finder/import tests: 4 passed. Its independent wheel was built and
  entry-point loading checked without importing Jittor, Torch or a third-party
  framework. Supported-version checks use controlled fake packages; they are
  not a new end-to-end third-party workload claim.
- Read-only audit found duplicate undo and concurrent first-owner creation;
  both were fixed. Five pure Python counterexample checks then passed, including
  nested hook/adopt rollback and preservation of foreign writes.
- The required full structure run, combined with the core-misc and serialization
  checks, concluded 1,270 passed / 51 failed / 6 skipped. This is not a green
  whole-repository gate. Owner/path, resource-manifest, dtype, re-export and
  exception-cleanup contracts affected by the migration were corrected;
  the selected 33 affected failure nodes then all passed (zero skips).
  Packaging/import structure checks separately passed 21 nodes, and duplicate
  implementation, collective ownership and optimizer signature checks passed.
- Remaining findings from that full run were not hidden or converted into
  passing tests. They include old C++ assertion counts, child-process and pytest
  helper contracts, a frozen FSDP reflection digest, and native-mode assumptions
  in a Torch-mode process. A native-mode control passed repeat and dtype-policy
  checks. The non-sharing allocator control still rejects scalar broadcast
  views with `Allocator cannot represent shared strided storage`; this C++
  path was not changed in this batch and remains a concrete limitation.

Raw logs remain outside the checkout under the task-specific `_state/` runs.
No NPU/CANN/HCCL, ROCm/Corex, or multi-machine execution is claimed. FSDP peak
memory optimization and the remaining performance tasks are still open.

## Remaining architecture work

7.03 still includes serialization/safetensors, distributions, autograd/library,
some numerical/FSDP/factory paths and frontend type factories. Legacy native API
modification and the full independent namespace boundary remain in 7.12.
Module-level identity alone does not establish exact Torch semantics.

Use the [current handoff](../architecture/refactor-handoff.md), whose historical
body is explicitly folded. Its old counts and old worktree-cleanliness reports
are not current state. Task counts come only from the board's task rows and
include derived identifiers; parent and derived tasks overlap.
