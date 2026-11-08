# Contributor guide to the implementation

[Documentation](../index.en.md) · [中文](index.md)

Read by the change you are making. Start with [contributing](contributing.en.md) to prepare
the environment; use the first three pages below to locate code and choose validation.
The topic pages define implementation contracts, not a checklist that every contributor must read.

## Start a change

- [Repository, packaging and resource boundaries](repository-layout.en.md)
- [Find the source owner](source-architecture.en.md)
- [Choose and run tests](testing/test-system.en.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

repository-layout.en
source-architecture.en
testing/test-system.en
```
-->

## Change core or operators

- [Runtime state and graph submission](runtime/native-runtime.en.md)
- [Storage, strides and aliases](runtime/storage-layout.en.md)
- [Operator registration, dispatch and fallback](runtime/operator-dispatch.en.md)
- [Write JIT operator source](runtime/jit-operator-source.en.md)
- [Backend builds and resource packaging](backends/backend-build.en.md)
- [ACL launchers, attributes and descriptors](backends/acl-backend-contracts.en.md)
- [Graph replay and training-step capture](runtime/static-graph-replay.en.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

runtime/native-runtime.en
runtime/storage-layout.en
runtime/operator-dispatch.en
runtime/jit-operator-source.en
backends/backend-build.en
backends/acl-backend-contracts.en
runtime/static-graph-replay.en
```
-->

## Change Torch compatibility or downstream integrations

- [Frontend object ownership and dtypes](compatibility/frontend-boundaries.en.md)
- [Install transactions and rollback](compatibility/install-transactions.en.md)
- [Downstream scope and acceptance evidence](testing/downstream-scorecards.en.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

compatibility/frontend-boundaries.en
compatibility/install-transactions.en
testing/downstream-scorecards.en
```
-->

## Validate and diagnose

- [Numerics, gradients and coverage](testing/test-evidence.en.md)
- [Process isolation, skips and completion](testing/test-isolation.en.md)
- [JIT cache and duration](testing/jit-cache.en.md)
- [User errors and internal errors](runtime/error-categories.en.md)
- [Locate asynchronous errors](runtime/async-error-diagnostics.en.md)
- [Record NPU reproduction and performance evidence](testing/npu-validation-templates.en.md)
- [Parallel compiler failure investigation](known-issues/parallel-compiler-segfault.en.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

testing/test-evidence.en
testing/test-isolation.en
testing/jit-cache.en
runtime/error-categories.en
runtime/async-error-diagnostics.en
testing/npu-validation-templates.en
known-issues/parallel-compiler-segfault.en
```
-->

## Report a result or defect

Track tasks, defects, pending hardware acceptance and execution results in GitHub issues/PRs.
Repository rules live in [AGENTS.md](https://github.com/Jittor/jittor/blob/2.0-refactor/AGENTS.md);
select a practical method from the [skill catalogue](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/README.md).
The rules page links immutable historical issue and hardware records. Unverified items remain unresolved.

## Compatibility architecture and contribution workflow

Use these pages when changing the compatibility frontend or documentation, rather than when running a model.

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

contributing.en
compatibility/torch-architecture.en
compatibility/torch-lowering.en
compatibility/principles.en
```
-->
- [Contribution workflow](contributing.en.md)
- [Shared graph and frontend layers](compatibility/torch-architecture.en.md)
- [Lowering calls to the core](compatibility/torch-lowering.en.md)
- [Compatibility acceptance rules](compatibility/principles.en.md)
