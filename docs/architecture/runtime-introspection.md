# Runtime introspection

`jt.introspection` is the supported read-only observation surface for tests and
diagnostics. Its three layers read existing native/runtime services; they do
not own a second configuration, backend registry, or counter table. Write
operations remain in `jt.runtime.scope(...)`. Importing the implementation
module itself requires only the standard library and the existing state and
capability definitions, not native bootstrap.

## Capabilities

```python
inventory = jt.introspection.capabilities.devices("cuda")
if inventory.capability.failed:
    raise RuntimeError(inventory.capability.reason)
if inventory.capability.enabled:
    print(inventory.count, inventory.devices)
```

`backends()` returns the native core's known backend names; `registered_backends()`
returns this runtime's registered backend names. `backend(name)` returns an
immutable `Capability`; `devices(name)` returns an immutable `DeviceInventory`
containing that record and a tuple of `Device(backend, index)` values. Indices
are logical indices in that backend's current visible-device namespace, not
physical GPU identifiers. CPU is represented as `Device("cpu", 0)`.

Registered backend queries use `core.backend_device_count(name)`. They do not
change the current device or use `use_cuda` as an availability check. Thus a
CPU-default runtime may still expose two visible CUDA devices. A query exception
or negative native count yields FAILED, with reason and error evidence;
`inventory.count` is `None`, never a fabricated zero. An unregistered accelerator
reuses `jt.capability`'s physical-presence/build evidence and cannot report
enabled. A registered backend with zero visible devices reports DISABLED.
Unknown names raise `ValueError` rather than silently assuming CPU.

`libraries()` and `library(name)` forward to the existing capability service.
Library observation always uses `load=False`: UNPROBED stays UNPROBED and no
optional extension is compiled or loaded. This namespace has no `load=True`
option. If a test intentionally initializes an optional library, that explicit
operation is separate from observation. Existing `jt.capability` APIs remain
available, including their explicit loading contract.

Capability records and device inventories reject implicit boolean conversion.
Callers must distinguish `.enabled`, `.failed`, `.disabled`, `.absent`, and
`.unprobed`; a broken build must not become a skip. `present` describes the
evidence behind a capability, not a promise that a numerical operation succeeds.
Device inventory is a read-only driver query, not a numerical hardware test.

## Effective policy

```python
view = jt.introspection.policy
before = view.snapshot()
with jt.runtime.scope(no_grad=not before.runtime["no_grad"]):
    assert view.runtime.no_grad != before.runtime["no_grad"]
```

`policy.startup` reads `jt.config`; `policy.runtime` reads
`jt.runtime.context`. Both offer attribute and mapping access. Canonical runtime
flags are in the runtime view; startup settings such as `cache_path`, compiler
paths and architecture selections are in the startup view. Counters are not
policy. Unknown keys raise `KeyError` and unknown attributes raise `AttributeError`.

Views remain live across native scope changes. Returned dictionaries and lists
are recursively frozen as mapping proxies and tuples, so a nested
`compile_options` value cannot mutate native policy. `policy.snapshot()` returns
an immutable `PolicySnapshot(startup, runtime)` with detached frozen values;
later policy changes do not rewrite previous snapshots. Assignment and deletion
through every observation namespace raise `AttributeError`.

`runtime.use_cuda` is a default execution/placement policy, not proof of device
availability, and does not describe the placement of every tensor in a mixed
CPU/CUDA graph. Use the capability layer for availability and tensor placement
APIs for an individual tensor.

## Counters

`counters` has live read-only `exec_calls`, `allocator`, `held_vars`, `live_vars`,
and `live_ops` properties. `snapshot()` returns a detached immutable
`CounterSnapshot` with those fields. Native liveness queries remain the single
owner of counts; querying them does not keep additional tensor references.

`allocator` is an `AllocatorCounters(enabled, alloc_calls, allocated_bytes,
free_calls, freed_bytes)` record. Values are the native stat allocator's
cumulative allocation/free traffic, not RSS or current allocated memory.
`enabled=False` makes missing instrumentation explicit. Native changes to its
instrumentation mode can reset these counters: subtracting two observations
across a mode change is not a valid traffic measurement. There is no reset
operation in this API.

Snapshots neither submit pending graphs nor synchronize devices nor run GC.
They are sequential observations, not an atomic cross-thread/device checkpoint.
Callers needing a completed-work boundary must explicitly synchronize outside
the observation API. Executor-call counts measure the native executor counter,
not kernel launches or elapsed time.

## Remaining test-consumer migration

This batch delivers the API foundation. Bulk consumer migration follows the
test-layout batch; it is not counted as complete here. These mappings preserve
the question each test is actually asking:

| Existing read | Supported replacement | Boundary |
| --- | --- | --- |
| `jt.compiler.has_cuda`, accelerator skip based on `jt.flags.use_cuda` | `jt.introspection.capabilities.backend("cuda")` | Check failure before skipping disabled/absent. |
| `core.get_device_count()` for a named backend | `jt.introspection.capabilities.devices(name)` | Preserve FAILED/UNPROBED versus zero. |
| `compile_extern.cudnn_ops is not None` and other library checks | `jt.introspection.capabilities.library("cudnn")` | UNPROBED does not mean unavailable. |
| Current `jt.flags.use_cuda`, `amp_reg`, `no_grad`, precision reads | `jt.introspection.policy.runtime.<name>` | Preserve effective-policy semantics. |
| `jt.flags.cache_path`, compiler configuration reads | `jt.introspection.policy.startup.<name>` | Immutable startup configuration. |
| `jt.flags.exec_called` | `jt.introspection.counters.exec_calls` | No implicit graph execution. |
| `stat_allocator_total_alloc_byte` and matching calls/free values | `jt.introspection.counters.allocator.*` | Instrumentation mode and reset semantics matter. |
| `number_of_lived_vars/ops`, `number_of_hold_vars`, `liveness_info()` | `jt.introspection.counters.live_vars/live_ops/held_vars` | No implicit GC. |
| Flag assignment | `jt.runtime.scope(...)` where reversible | Writes are not part of introspection. |

At the pre-layout `0df` tree, direct AST observations of `jt`/`jittor` flag
accesses found 330 reads and 296 assignments; these are not the old 283 count.
Examples for migration are `test_cpu_parallel_pass.py` (CUDA policy versus
capability), `test_unary_op.py` (AMP save/restore),
`test_fused_identifier_rename.py` (cache path), `test_allocator2.py` (traffic),
and `test_node.py`/`test_core_invariant_properties.py` (liveness).
These counts exclude aliases, references inside embedded scripts, and writes
through a nested container. They are a bounded inventory, not a claim that all
private testing dependencies can or should be mechanically replaced. Tests
whose purpose is validating native setters, registration internals or binding
generation retain access to the implementation they are explicitly testing.
