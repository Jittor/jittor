# torch.vmap Owner Plan

## Current owner

`python/jittor/compat/torch/installers/numerical.py:1557-1647` currently owns
the compatibility implementation inside `install(ctx)`. The implementation has
two coupled pieces:

- `_vectorized_getitem_vmap` handles nested pointwise maps used by transformer
  mask construction. It depends on `jt.Var` shape/dtype checks and the runtime
  flag `g._transform_getitem_to_index_depth`.
- `_vmap` builds nested mapping specifications, applies the vectorized fast path,
  or loops over mapped axes and stacks the results. Its wrapped callable stores
  `_jittor_vmap_base` and `_jittor_vmap_specs` metadata.

There is no native `jt.vmap` owner. The public binding is currently the
`_alias("vmap", _vmap)` call in the same installer.

## Migration boundary

Do not extract only `_vmap`: the vectorized helper and runtime flag access are
part of the same contract. The eventual module-level owner should receive an
explicit context object (or a narrow callback for the transform-depth probe)
instead of reading a module-global `g`. `install(ctx)` should only bind that
stable callable to `ctx.jittor_module.vmap`.

Keep the existing mapping semantics unchanged: `in_dims`/`out_dims`, nested
spec metadata, scalar-to-singleton shape normalization, and the bool broadcast
fast path. Do not introduce a no-op fallback or a process-global runtime state.

## Focused acceptance

Before broad tests, add CPU-only nodes covering:

1. module identity and fidelity registration (`torch.vmap` is the module-level
   owner and is marked approximate with the context limitation documented);
2. a simple mapped unary function with `in_dims=0` and an explicit `out_dims`;
3. nested mapped boolean mask construction exercising the vectorized helper.

Run these nodes with an isolated Jittor cache. GPU/NPU behavior remains deferred
until a backend with the required transform integration is available.

## Verifiable contract

The module-level owner should satisfy these static/runtime checks before any
behavioral optimization:

- `torch.vmap is numerical.vmap`, with `__module__` and `__name__` identifying
  the stable owner; fidelity is registered as `approximate` and documents the
  injected context dependency.
- `install(ctx)` performs binding only. It must not create a new closure per
  install or retain the context in a process-global variable. Repeated installs
  must leave the same callable identity in place.
- The context callback exposes only the transform-depth probe. The callable
  must remain usable when that probe is false or absent, taking the ordinary
  loop/stack path.
- For every mapped argument, `in_dims=None` leaves the argument unchanged;
  mapped dimensions preserve their extent, and `out_dims` is applied exactly
  once. Nested calls append one mapping spec and retain the existing metadata
  attributes.
- Unsupported combinations (mismatched mapped extents, non-zero nested helper
  dimensions, or non-bool vectorized results) return to the ordinary path or
  raise a documented `RuntimeError`; they must never silently become a no-op.

Static checks should assert that no `g`/`ctx` object is captured by a module
global and that install contains only the stable binding. These checks are
independent of backend availability and can run without JIT compilation.

## Acceptance node sketch

The eventual test file should keep each contract independently runnable:

- `test_vmap_module_identity_and_fidelity`: import the numerical module, assert
  callable identity and the `approximate` report detail.
- `test_vmap_install_is_idempotent_and_context_free`: install two fresh contexts
  and assert the published function object is identical; inspect its closure and
  module globals to ensure no `InstallContext`/`jittor_module` instance is held.
- `test_vmap_simple_in_dims_out_dims_cpu`: map a pure elementwise function over
  axis 0, then over a non-zero output axis, comparing shape and values to a
  NumPy loop without invoking CUDA or NPU paths.
- `test_vmap_nested_boolean_fast_path_cpu`: exercise nested maps with boolean
  output and assert the broadcasted batch shape; this is the regression guard
  for transformer mask construction.
- `test_vmap_unsupported_mapping_falls_back_or_raises`: feed mismatched mapped
  extents and non-zero nested helper dimensions, asserting a documented
  fallback/`RuntimeError` rather than a silent direct call.

The first two nodes are static/metadata-only and should stay in the fast gate;
the remaining three may be marked CPU numerical and run with an isolated Jittor
cache when the owner extraction lands.

## Owner extraction checklist

Implement the migration in one commit so the binding and its dependencies stay
consistent:

1. Introduce a module-level `vmap` owner plus a private helper for the nested
   boolean fast path. Pass a `transform_depth()` callback into the owner rather
   than importing `InstallContext` or retaining `g`.
2. Move the wrapper metadata (`_jittor_vmap_base`, `_jittor_vmap_specs`) into the
   module-level implementation and preserve their tuple/list shapes exactly.
3. Replace `_alias("vmap", _vmap)` with a single binding to the stable owner;
   remove the install-local definitions in the same change.
4. Register `torch.vmap` fidelity before installation so metadata queries work
   even in static-only environments.

The extraction is complete only when the AST gate sees one public `vmap`
definition, no nested `def _vmap` under `install`, and no module-global closure
cell containing `ctx`, `g`, or `Var` instances.

## Context protocol sketch

Use a minimal callable protocol instead of a concrete context type:

```text
VmapContext:
    transform_depth() -> int
```

`transform_depth()` may return zero when the transform hook is unavailable. The
owner must then use the ordinary loop/stack path. This keeps CPU-only tests and
future ACL/NPU backends independent of the current torch installer object while
preserving the specialized transformer mask path when the hook is present.

## Test fixture and failure triage

The focused CPU nodes should run in a child process with isolated
`HOME`/`JITTOR_HOME`/`TMPDIR` and `JITTOR_TORCH_SHIM=1`. They must pin
`use_cuda=0` and avoid importing optional backend modules. A static-only run is
valid when no compatible JIT cache is available.

Record failures in one of three buckets:

- **owner/identity**: module function is not the object published at
  `torch.vmap`, or fidelity metadata is missing/stale;
- **context leak**: the function closure or module globals retain an installer
  context, `g`, or `Var` instance after installation;
- **mapping semantics**: shape/value mismatch, nested metadata loss, or an
  unsupported combination silently taking the direct-call path.

Do not classify a cold-cache compile timeout or an unavailable accelerator as a
semantic failure. Keep those as environment evidence beside the node result.

## Documentation handoff

When the owner extraction lands, update the 7.03 board entry with the code
commit, static identity result, and each CPU node id. Keep the vmap row marked
`待领` until the nested boolean path and unsupported-combination node pass; do
not infer completion from the identity-only gate. The handoff should link this
plan so a future backend-specific run can reuse the same contract.

## Static unsupported gate

Add a no-JIT AST node that parses `numerical.py` and checks the vmap boundary:

```text
assert one module-level `def vmap`
assert no nested `def _vmap` below `def install`
assert install has exactly one vmap binding
assert no module-level assignment references a name `g`, `ctx`, or `Var`
```

The same node should inspect the documented unsupported cases and require an
explicit branch or `RuntimeError` for each: mismatched mapped extents, a nested
helper with a non-zero mapped dimension, and a vectorized result whose dtype is
not boolean. This keeps unsupported behavior observable without constructing a
Jittor graph or requiring a backend.

## Context fixture

The static/runtime boundary can be exercised without a real installer by using
two sentinel contexts:

```text
class DepthZero:
    transform_depth() -> 0

class DepthTwo:
    transform_depth() -> 2
```

Construct the module-level owner with each callback and assert that both produce
the same ordinary loop result for a simple mapped function. Then delete the
sentinel objects and force collection; the owner must remain callable, proving
that no context instance was retained in a closure or module global. A separate
AST check should reject any default argument or closure cell whose value is an
`InstallContext`, `jittor_module`, or `Var` object.

## Concrete extraction order

Use this order when implementation work is scheduled:

1. Copy the two nested helpers to module scope without changing their loop,
   stack, or broadcast expressions. Add a private `_VmapRuntime` protocol that
   exposes only `transform_depth()` and pass it explicitly to the helpers.
2. Add the public `vmap` wrapper and fidelity registration beside the other
   numerical owners. Its defaults and metadata attributes must match the
   current install-local callable byte-for-byte at the API boundary.
3. In `install(ctx)`, construct one `_VmapRuntime` adapter around
   `ctx.jittor_module` and bind `g.vmap` to the module-level owner. Do not store
   that adapter in a module global; repeated installs should replace the binding
   with the same function object.
4. Delete the old nested helper definitions and `_alias("vmap", _vmap)` in the
   same commit. Run the AST owner/unsupported gate before any numerical test.

Rollback is mechanical: restore the single install-local binding and remove the
module-level adapter while preserving the new focused tests. No backend-specific
code should be needed for this owner extraction.

## Review checklist and evidence

The implementation review should attach one compact evidence block containing:

- the AST node count for module-level `vmap`, nested `_vmap`, and install
  bindings;
- the closure/global scan result (zero captured `InstallContext`, `g`, `ctx`,
  or `Var` instances);
- the fidelity record (`api`, `level`, and context limitation detail);
- the five focused node ids and their pass/skip reason, with cold-cache or
  unavailable-backend skips kept separate from semantic failures.

This block is sufficient for the board/handoff update; do not paste full JIT
logs. A missing field is a contract gap and should keep 7.03 marked `待领`.

## Unsupported behavior matrix

| Condition | Required behavior | Evidence |
| --- | --- | --- |
| mapped arguments have different extents | fall back to ordinary loop or raise a typed `RuntimeError` | node id plus exception/fallback marker |
| nested helper uses a non-zero mapped dimension | skip vectorized helper; preserve `in_dims`/`out_dims` metadata | AST branch and focused node |
| vectorized result is not boolean | skip vectorized helper; ordinary path remains authoritative | dtype guard assertion |
| transform-depth callback absent | treat depth as zero; no context access error | sentinel `DepthZero` node |
| `out_dims` is a tuple/list | normalize once and stack at the requested axis | shape/value assertion |

The matrix is normative for both CPU and future backend implementations. Any
backend-specific deviation must be documented as a separate fidelity limitation,
not hidden in the generic owner.

## Backend portability boundary

This task is an organization refactor, not a vectorization optimization. The
owner extraction must keep the existing loop/stack implementation and avoid new
backend kernels, device transfers, or scheduling flags. CPU is the reference
backend; CUDA, ACL, and other backends consume the same callable and may only
provide their own `transform_depth()` probe when available.

Record backend evidence independently:

- CPU: run the five focused nodes from the acceptance sketch;
- CUDA/ACL: first run the identity/context-free AST nodes, then opt into the
  numerical nodes only on hardware with a working transform hook;
- unavailable hardware: mark the numerical nodes as environment-skipped and
  retain the static contract results.

No backend skip may downgrade an owner/identity or unsupported-behavior failure
to a pass.

## Stable API and fidelity template

The extracted owner should expose this minimal signature at module scope:

```text
vmap(func, in_dims=0, out_dims=0, *, runtime=None, **kwargs)
```

`runtime` is an internal-only keyword used by `install(ctx)` to provide the
`transform_depth()` callback; it must not appear in the public Torch shim
signature. If the callback is omitted, behavior is identical to
`transform_depth() == 0`. Preserve any additional Torch kwargs by rejecting
unsupported values explicitly rather than silently dropping them.

Register fidelity with a detail string that names the supported mapping paths,
the context callback limitation, and backend scope. This keeps reports stable
across releases and lets static tests assert that context-dependent behavior is
documented instead of inferred from implementation details.

## AST gate output contract

Implement the static gate as a small AST walker so it remains backend-agnostic:

```text
module_defs = functions named `vmap` at module scope
install_defs = functions named `_vmap` nested under `install`
bindings = assignments to `g.vmap` or `_alias("vmap", ...)`
captures = names `g`, `ctx`, `Var`, `InstallContext` in module-level defaults/closures
```

Emit one JSON-like record with integer counts and boolean results:

```text
{"module_vmap": 1, "nested_vmap": 0, "install_bindings": 1,
 "forbidden_captures": false, "unsupported_guards": true}
```

The node should fail closed when a field cannot be determined (for example,
dynamic `setattr` binding) and report the source line rather than guessing. This
record is the canonical static evidence attached to the 7.03 handoff.

## Namespace and install order

The owner mapping must remain explicit across compatibility namespaces:

| Namespace | Owner after extraction | Install action |
| --- | --- | --- |
| `torch.vmap` | `installers.numerical.vmap` | bind the stable callable once |
| `torch.compat.torch` re-export | same object | publish by identity, no wrapper |
| `Var` methods | no new method owner | keep existing call sites routed through root `torch.vmap` |

Install order is `tensor.base` -> `tensor.methods` -> `numerical`; the numerical
step may read the transform-depth probe but must not rebind tensor methods. Any
future `torch.linalg` or backend namespace re-export should point at the same
object and be recorded as an identity assertion, not a second implementation.

## Version compatibility and exit criteria

Keep the extracted owner compatible with the currently supported Torch shim
signature (`in_dims`, `out_dims`, and mapped callable metadata). New Torch
keywords should be rejected with a typed error until a corresponding contract
is documented; do not silently absorb them in `**kwargs`.

The 7.03 `vmap` item may leave `待领` only when all of the following evidence is
present in one handoff update:

- AST gate record shows one module-level owner, one install binding, and no
  forbidden captures;
- fidelity report is queryable and names the context/backend limitation;
- simple and nested CPU nodes pass, including explicit `out_dims`;
- unsupported mapping node observes the matrix in this plan;
- any CUDA/ACL run is either passing or separately marked environment-skipped.

Until then, keep the board status unchanged and link this plan from the partial
handoff rather than inferring completion from static checks alone.

## State and resource non-regression

The owner extraction must not introduce process-wide mutable state or retain
per-install resources. The static gate should additionally assert:

- no module-level list/dict/cache is mutated by `vmap` calls;
- repeated `install(ctx)` calls do not increase the number of registered
  callables or closures;
- a failed install leaves the previous `torch.vmap` binding and fidelity record
  intact (transactional patcher or explicit hard failure);
- temporary mapped outputs are released after each wrapped call and do not add
  persistent entries to Jittor's hold-var registry.

These checks are metadata/resource checks only and may run without constructing
large tensors. Any resource leak is a blocker for owner completion even when all
numerical nodes pass.

## Deterministic test data

Keep the CPU numerical nodes deterministic and small: seed NumPy and Jittor with
a fixed value, use integer/boolean inputs for the nested mask case, and limit
mapped extents to at most four elements. Compare against a pure Python or NumPy
loop with an absolute tolerance appropriate to the input dtype. Avoid randomized
property tests in the first gate; they make cold-cache failures and backend
differences harder to classify. A later stress suite may widen shapes only after
the owner/identity and unsupported gates are green.
