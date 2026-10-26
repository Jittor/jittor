# Jittor Project Context

- Status: Current-state index, not a history log
- Last reviewed: 2026-10-26
- Baseline reviewed: `810654fc4`
- Owner: Jittor core maintainers
- Freshness expires: 2027-01-06
- Review when: the release line, a top-level goal or an indexed location changes

Find where things are here, then read only the document your task needs.

## What the project is

Jittor is a JIT deep-learning framework built on meta-operators: Python builds a
graph, the C++ core in `src/` fuses and compiles kernels, and backends live under
`backends/<name>/` (CPU, CUDA, ROCm, ACL/Ascend, Corex, and `comm/{mpi,nccl,hccl}`).
The Torch compatibility layer is the separate `jittor-torch` project in `compat/`
(published as `jittor.compat`, including the `import torch` shim); downstream
library adapters live in `adapters/`. The goal is a maintainable, Torch-grade
framework that keeps the JIT/meta-operator design, under the gates in
[Torch compatibility principles](../../docs/compatibility/principles.md).

## Current state

The 2.0 restructuring has landed and the tree is in pre-release cleanup: the
refactor-period board, plans and handoff documents are gone, and task status now
lives in GitHub issues and pull requests. The accepted layout is in
[repository layout](../../docs/development/repository-layout.md); packaging targets
Python 3.7-3.13; [`noxfile.py`](../../noxfile.py) and
[`tools/run_test_suite.py`](../../tools/run_test_suite.py) are the maintained gate
surface (tiers `core` / `smoke` / full, see `AGENTS.md`). User-facing breaking
changes: [2.0 migration notes](../../docs/releases/2.0.md). The Ascend 310P
support from PR #18043 is integrated (SoC capability table, reduced-precision
opt-in, ACL inference helpers and mapped composite ops), and user documentation
is now paired: Chinese-default `X.md` plus English `X.en.md`, with
`README.md` / `README.en.md` at the root. The duplicate-capability audit
consolidated optimizer state-name translation, vLLM accelerator detection,
slot mapping, async host copies and sampling-snapshot caches, and the AMP
half-preference test; forward-placement and jagged-axis have contract
tripwires.

## Where information lives

| Kind | Location |
| --- | --- |
| Rules for agents and maintainers | [`AGENTS.md`](../../AGENTS.md), [collaboration](collaboration.md) |
| Entry index of manuals and skills | [agent-index.md](agent-index.md) |
| Environment, isolation, backend prerequisites | [environment.md](environment.md) |
| Open defects and limitations | [known-issues.md](known-issues.md) |
| Work waiting for hardware | [deferred-hardware.md](deferred-hardware.md) |
| Source architecture and module boundaries | [source architecture](../../docs/development/source-architecture.md), [development index](../../docs/development/index.md) |
| Test system and gate tiers | [test system](../../docs/development/test-system.md) |
| Mechanism notes (numerics, placement, profiling) | [notes index](../../docs/notes/index.md) |
| Torch compatibility | [compatibility index](../../docs/compatibility/index.md) |
| Backend guides (Ascend, Corex, jtrun; legacy MPI) | [guides index](../../docs/guides/index.md) |
| Performance method | [benchmarking](../../docs/performance/benchmarking.md) |
| Reproducible verification results | [results index](../../docs/results/index.md) |
| Reusable verification tools | `agent/skills/` (listed in [agent-index.md](agent-index.md)) |
| Task status, handoff, evidence | GitHub issues and pull requests |

## Active focus areas

- **Accelerator coverage without CPU fallback.** NPU reduction, FFT and atan2 gaps
  (KI-BACKEND-001..003), ACL launcher/descriptor work that has never run on a
  device (KI-BACKEND-012), untested backend gradients (KI-BACKEND-013), and
  ROCm/NPU verification of shared semantics (KI-OPS-002, KI-SEMANTICS-003).
  Commands for hardware day are in [deferred-hardware.md](deferred-hardware.md).
- **Silent numerics.** cuDNN autotuning coupled to scheduling (KI-EXEC-003).
- **Torch frontend gaps.** torchmetrics on CUDA (KI-COMPAT-010), RNN off cuDNN
  (KI-COMPAT-011), in-process shim metadata (KI-COMPAT-012).
- **Executor and liveness.** Threaded writers to one parameter (KI-EXEC-005),
  liveness underflows (KI-EXEC-008), the op-level
  parallel compiler (KI-COMPILER-001).
- **Performance gaps.** Broadcast operands in fused kernels (KI-CODEGEN-001), dead
  matmul/conv relays (KI-TUNER-001), UNet reductions and elementwise kernels
  (KI-CODEGEN-003, KI-CODEGEN-004, KI-COMPAT-009), pipelined memory and backward
  submission (KI-EXEC-004, KI-EXEC-010).
- **Gate trust.** Gate cost (KI-TEST-012, KI-TEST-013).
- **Research.** [Agent-operable optimization](../../docs/research/agentic-optimization.md)
  is a proposal only; no autonomous mutation path exists.

## Before running work

Sync with the target remote branch as `AGENTS.md` requires and record the SHA;
isolate the run per [environment](environment.md); search
[known-issues.md](known-issues.md) and the [results index](../../docs/results/index.md)
for existing evidence; reproduce minimally before editing. Update this page only
when its current-state summary or a location changes.
