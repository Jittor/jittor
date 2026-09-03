# Torch Compatibility Principles

- Status: Accepted
- Last reviewed: 2026-08-12
- Baseline: `582fc51d`
- Owner: compatibility and core maintainers
- Review when: a new compatibility layer, fallback policy, or downstream patch
  mechanism is proposed

Torch compatibility is a means to run useful workloads with Jittor semantics and
Jittor devices. It is not permission to copy every Torch namespace into one
module or to report success while computation has escaped to another runtime.

## Goals

Compatibility work must satisfy five gates:

- **G1, semantic correctness:** outputs, dtypes, shapes, errors, gradients, and
  state transitions match the documented contract for supported inputs.
- **G2, device execution:** advertised CUDA, ROCm, or NPU operations execute on
  that backend; host conversion is not an accelerator implementation.
- **G3, composition:** the API works through the public import path used by real
  applications, including registration and serialization boundaries.
- **G4, maintainability:** capability has one owner, failures are observable, and
  version-specific behavior is isolated.
- **G5, evidence:** regression tests and, where performance is material,
  reproducible benchmarks support the claim.

## Ownership decision

Use the lowest reusable layer that correctly owns the behavior:

| Behavior | Owner |
| --- | --- |
| Tensor semantics, gradients, dtype/device rules | Jittor core or native domain |
| General neural-network operation | `jittor.nn` |
| Torch spelling or argument adaptation | `jittor.compat.torch` |
| Reusable import/build/patch mechanism | `jittor.compat` |
| Torch shim runtime/deploy and top-level `torch` surface | `jittor.compat.shim` |
| Triton API bridge | `jittor.compat.triton` |
| One project's model/version glue | an optional integration package |

If several downstream projects need the same operation, promote the operation to
the native domain. If they only need the same patching mechanism, promote the
mechanism while leaving each patch outside core.

## Activation boundary

Plain `import jittor as jt` retains native Jittor contracts and does not publish
Jittor as the top-level `torch` module. Torch compatibility is activated by an
explicit `jittor.compat.shim.activate()` call, `JITTOR_TORCH_SHIM=1`, the
deployed `torch` entry point, or the historical `jittor.torch_compat` import.
Merely spelling a local alias as `import jittor as torch` does not activate the
shim. The installer is idempotent, but activation is process-wide because the
two modes necessarily differ on class-level APIs such as `Var.data`.

Native and compatibility tests therefore run in separate processes. A native
test must not rely on a Torch wrapper installed by another collected test, and a
Torch compatibility test must prove that its explicit entry point completed the
required installer graph.

## Correctness rules

- Compare against an independent oracle or written contract, not another alias
  of the implementation under test.
- Cover positional and keyword forms, dtype promotion, broadcasting, empty and
  non-contiguous inputs, device selection, gradients, and stateful behavior when
  relevant.
- An unsupported contract raises a focused error. Returning a plausible but
  wrong value is never an acceptable fallback.
- A compatibility wrapper must not globally change unrelated native Jittor
  semantics unless the native behavior is itself corrected and tested.
- Import and installer steps are idempotent and report each failed component
  independently.

### Storage sharing

Torch's aliases share storage; Jittor's Vars do not, and the difference is
silent in the direction that matters -- a write that lands in a temporary
instead of the tensor reports nothing, and surfaces much later as wrong numbers.

- A basic-index slice is a view. `param[:rows] = w`, `param[:rows].copy_(w)`
  and `param[:rows].data.copy_(w)` all have to reach `param`; a sharded weight
  loader writes exactly that way, and losing the write leaves the parameter at
  its initial value with nothing to report.
- `Tensor.numpy()` shares storage in Torch and copies here. Do not emulate that
  by handing out Jittor's own buffer (`Var.data`): the view detaches the moment
  the graph re-materialises the Var, so the first write lands and every later
  one vanishes -- worse than copying, because it looks like it works. The
  faithful direction is a write-through array that mirrors each host write back
  with a setitem the lazy graph can see.

## Device and fallback rules

Host-side NumPy may be used as a test oracle or an explicitly documented
CPU-only implementation. It must not sit behind an API advertised as CUDA, ROCm,
or NPU capable. A backend fallback is acceptable only when all of these hold:

1. the public contract permits the fallback;
2. device transfer is explicit and observable;
3. gradients and dtype behavior remain correct;
4. the performance impact is documented;
5. a test distinguishes fallback from native execution.

Backend availability is probed by a small real operation. Environment failures
and framework defects are recorded separately; neither becomes a blanket skip.

## Performance rules

Correctness is required before optimization. Performance claims use warmup,
synchronization, fixed shapes/dtypes, isolated caches, and an exact revision.
Compare equivalent execution modes and report both latency and memory when the
change affects either. See [performance benchmarking](../performance/benchmarking.md).

An optimization must retain a reliable fallback and a way to prove which path
executed. It may not weaken error reporting or test coverage to improve a
headline number.

## Compatibility aliases

Aliases preserve imports; they do not preserve duplicate source trees. A legacy
name must resolve to the canonical module or callable object whenever observable
identity matters. Every new alias documents:

- the canonical path;
- the compatibility promise it serves;
- tests for identity and import order;
- the condition under which it can be retired.

In particular, `jittor.torch_shim` is a same-object legacy alias of the canonical
`jittor.compat.shim` package. The deployed top-level `torch` surface delegates to
that canonical runtime rather than owning a second implementation.

## Minimum evidence

A compatibility change should include the smallest applicable set:

- focused unit and error-contract tests;
- CPU reference comparison;
- gradient and second-order gradient checks when supported;
- CPU-to-accelerator parity on each advertised backend;
- import-order, identity, and repeated-install tests;
- a real downstream smoke test for integration-facing work;
- benchmark evidence for a claimed fast path.

Record persistent limitations in the
[known-issues ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md),
with an owner and a
test or reproduction that signals when the limitation is gone.
