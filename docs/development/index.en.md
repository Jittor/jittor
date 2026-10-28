# Development documentation

For Jittor contributors: how the code is organised, how tests are run, and
which defects are currently known.

The **ledger** of known defects is
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md):
every entry carries severity, evidence, workaround, and exit criteria.
`known-issues/` holds only entries that need a long investigation record; there
is currently one.

Read the repository's [`CONTRIBUTING.md`](https://github.com/Jittor/jittor/blob/master/CONTRIBUTING.md)
and the [contributing guide](../contributing.en.md) first, then the documents
below as needed.

```{toctree}
:maxdepth: 1

repository-layout.en
source-architecture.en
acl-backend-contracts.en
torch-install-transactions.en
storage-layout.en
jit-operator-source.en
error-categories.en
async-error-diagnostics.en
test-system.en
static-graph-replay.en
npu-validation-templates.en
downstream-scorecards.en
known-issues/parallel-compiler-segfault.en
```

| Document | Contents |
| --- | --- |
| [Repository layout](repository-layout.en.md) | Responsibilities and packaging boundaries of the top-level directories |
| [Source architecture](source-architecture.en.md) | Python module ownership and dependency direction; backend build configuration, dynamic shapes, and commit boundaries |
| [ACL backend contracts](acl-backend-contracts.en.md) | Shared launcher, attribute data channel, descriptor-cache shell, and Ascend acceptance criteria |
| [Torch install transactions](torch-install-transactions.en.md) | Which process state install and lazy hooks write, what is rollbackable, and what tests may claim |
| [Static graph replay](static-graph-replay.en.md) | Repeated steps stop rebuilding the graph in Python: `jt.graph_replay`, `jt.capture_step`, the automatic policy, and its guards |
| [Storage layout contract](storage-layout.en.md) | Strides and storage belong to the native side; the frontend keeps no second copy |
| [JIT operator source contract](jit-operator-source.en.md) | Operator sources are compiled twice; `jit_run` must satisfy both C++ and KernelIR |
| [Error categories](error-categories.en.md) | Which entry point user errors and internal invariants take |
| [Async error diagnostics contract](async-error-diagnostics.en.md) | The bounded launch-record ring and how it differs from graph metadata |
| [Test system](test-system.en.md) | Test tiers, gates, and acceptance rules; optional fixtures, manual scripts, and maintainer probes |
| [Downstream coverage and scorecards](downstream-scorecards.en.md) | Registry denominators, device and numerics evidence, and L2-L5 acceptance fields |
| [NPU defect and performance record templates](npu-validation-templates.en.md) | Which fields an Ascend conclusion must record under `docs/results/` |
| [Parallel compiler: file-level deadlock and operator-level segfault](known-issues/parallel-compiler-segfault.en.md) | The `KI-COMPILER-001` investigation: two same-name failures with different causes, reproduction protocol, and acceptance bar |
