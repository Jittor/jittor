# Agent-Operable Framework Optimization

- Status: Research proposal; not implemented
- Last reviewed: 2026-08-12
- Baseline: `582fc51d`
- Owner: compiler and performance maintainers
- Review after: the first read-only prototype or any proposal to enable automatic
  source/configuration changes

## Problem

Jittor already compiles and tunes operations at runtime, but optimization work is
difficult for an automated agent to perform safely. Relevant decisions are
spread across logs, generated source, tuner state, cache directories, profiler
output, and backend-specific flags. The information is mostly human-readable,
not stable enough to compare across revisions, and often mixed with mutable
runtime state.

The goal is to expose a narrow, versioned control and observation surface so an
agent can form a hypothesis, run a bounded experiment, compare evidence, and
produce a reviewable patch or tuning recommendation. The agent must not become
an unrestricted runtime autotuner or silently change user execution.

## Scope

The proposal covers framework-internal optimization of operator fusion, generated
kernel choices, launch configuration, memory planning, and reusable backend
paths. It does not cover model architecture search, training-data decisions,
unreviewed package installation, or production changes made without tests and a
human-reviewed diff.

## Design principles

1. **Read-only first.** Observation and replay land before any mutation API.
2. **Every claim is replayable.** An experiment identifies the source commit,
   environment fingerprint, workload, cache state, commands, and metrics.
3. **Bounded control.** A mutation declares the exact parameters or source region
   it may change, with a timeout and resource budget.
4. **Correctness dominates speed.** Candidate output, gradients, errors, and
   backend placement pass before performance is compared.
5. **No hidden fallback.** Reports state which implementation and device actually
   executed.
6. **Reviewable artifacts.** Source patches remain ordinary diffs; tuning data is
   structured, small, and tied to a schema version.
7. **Isolation by construction.** Baseline and candidate runs use separate state
   and JIT caches.

## Observation surface

A read-only command should emit one JSON document per workload. The first schema
should include:

```json
{
  "schema_version": 1,
  "revision": "<git commit>",
  "workload": {"id": "<stable id>", "inputs": "<manifest hash>"},
  "environment": {
    "python": "<version>",
    "compiler": "<id and version>",
    "backend": "cpu|cuda|rocm|npu",
    "device": "<model and capability>"
  },
  "graph": {
    "operators": [],
    "fusion_groups": [],
    "dtype_shape_signatures": []
  },
  "compilation": {
    "cache": "cold|warm",
    "sources": [],
    "flags": [],
    "durations_ms": []
  },
  "execution": {
    "implementation_ids": [],
    "latency_ms": {},
    "memory_bytes": {}
  },
  "correctness": {"checks": [], "status": "pass|fail|not-run"}
}
```

Paths stored in the document are repository-relative or artifact-relative.
Secrets, user home directories, host addresses, and raw environment dumps are
excluded. Generated source may be stored as a content-addressed external
artifact; the repository records only its hash and a small reviewed excerpt when
needed.

## Control surface

After the observation schema is stable, a local experiment manifest may permit a
small allowlisted set of controls:

- compiler/tuner flags with declared type and valid range;
- fusion on/off or one named fusion-policy candidate;
- launch dimensions and tile sizes within backend limits;
- selection among registered implementation variants;
- one repository patch supplied as a normal diff.

The runner rejects unknown keys, absolute output paths, commands outside an
allowlist, shared baseline/candidate caches, network access, and workloads with
no correctness oracle. It records the resolved controls rather than only the
requested values.

The control API must not expose arbitrary Python evaluation, shell fragments,
or unrestricted compiler flags. Source-editing agents already produce reviewable
Git patches; embedding another code execution channel in the optimizer would add
risk without useful capability.

## Experiment loop

```text
capture baseline -> propose one hypothesis -> validate candidate correctness
       -> warm up -> measure repeated samples -> compare -> retain or reject
```

Each experiment changes one coherent factor. Baseline and candidate use the same
inputs, precision policy, synchronization, compiler toolchain, device, and
measurement method. Results include dispersion and sample count, not only the
best run. A candidate is rejected when correctness fails, the requested path did
not execute, compilation cost grows outside budget, or the improvement is below
the predeclared noise threshold.

## Guardrails

- Run in a disposable worktree and isolated `$JITTOR_LAB_ROOT/_state/` subtree.
- Require explicit CPU/accelerator parity and gradient checks for affected
  differentiable operations.
- Cap wall time, compilation count, disk use, and generated-source size.
- Never reuse benchmark caches for unit tests.
- Treat crashes, timeouts, NaNs, and missing metrics as failures, not zero values.
- Require a human-reviewed patch before any source or default policy reaches the
  main branch.
- Retain a reliable fallback and a runtime identifier for the selected fast path.

## Staged implementation

### Phase 0: schema and capture

Define the JSON schema, stable workload identifiers, environment redaction, and
a read-only capture command. Validate it on one CPU elementwise operation and one
CUDA matrix operation without changing compilation decisions.

### Phase 1: deterministic replay

Replay a captured workload in isolated cold and warm caches. Prove that operator
signatures, implementation identifiers, and correctness results are stable
enough to compare across two worktrees.

### Phase 2: allowlisted parameter search

Expose one existing tuner parameter family through a typed manifest. Compare a
small fixed candidate set and produce a report without editing source or changing
global defaults.

### Phase 3: patch evaluation

Accept an ordinary Git diff generated outside the runner, build it in a fresh
worktree, and apply the same correctness/performance gate. The output is a review
bundle, never an automatic merge.

### Phase 4: policy integration

Only after multiple reviewed studies may a winning choice enter a registered
runtime policy. The policy includes device guards, a fallback, telemetry for path
selection, and regression benchmarks.

## Initial acceptance criteria

The first prototype is successful when it can, without modifying the source
checkout:

- capture a redacted, schema-valid CPU and CUDA workload;
- replay each workload in isolated caches;
- prove which implementation ran;
- fail a deliberately wrong candidate through independent output or gradient
  checks;
- compare repeated latency and memory samples with a declared noise threshold;
- emit a compact report that another maintainer can reproduce from the recorded
  commit and manifest.

No autonomous mutation or merge capability is in scope until these criteria are
met and the control surface receives a separate architecture review.
