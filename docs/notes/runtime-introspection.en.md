# Runtime introspection

- Status: implemented
- Last reviewed: 2026-10-05 (baseline `e3c369acb`)
- Owner: native Runtime and test-infrastructure maintainers
- Review trigger: adding a capability query, policy field or counter, or changing the
  write semantics of `jt.runtime.scope`

`jt.introspection` is a **supported read-only observation surface** for tests and
diagnostics. Its layers read existing native/runtime services and **do not own a
second copy of configuration, backend registries or counter tables**; writes still
go only through `jt.runtime.scope(...)`. The implementation module
(`python/jittor/_runtime/introspection.py`) depends only on the standard library and
existing state and capability definitions, so importing it needs no native
bootstrap.

## Capabilities: which backends, how many devices, which libraries

```python
inventory = jt.introspection.capabilities.devices("cuda")
if inventory.capability.failed:
    raise RuntimeError(inventory.capability.reason)
if inventory.capability.enabled:
    print(inventory.count, inventory.devices)
```

- `backends()` returns the backend names the native core **knows about**;
  `registered_backends()` returns the ones **registered** in this runtime.
- `backend(name)` returns an immutable `Capability`; `devices(name)` returns an
  immutable `DeviceInventory` containing that record and a set of
  `Device(backend, index)`. Indices are **logical** within the backend's currently
  visible device namespace, not physical GPU identifiers; CPU is represented as
  `Device("cpu", 0)`.
- Queries on a registered backend go through `core.backend_device_count(name)`:
  they **do not switch the current device and do not use `use_cuda` as an
  availability criterion**. So a runtime that runs on CPU by default can still
  report two visible CUDA devices.
- A query that throws or a native call returning a negative number yields FAILED,
  with a reason and the error evidence, and `inventory.count` is then `None` —
  **a 0 is never fabricated**. A registered backend with 0 visible devices is
  DISABLED; an unregistered accelerator follows `jt.capability`'s physical
  presence/build evidence and can never be reported as available. An unknown name
  raises `ValueError` rather than defaulting to CPU.
- `libraries()` and `library(name)` forward to the existing capability service but
  **always with `load=False`**: UNPROBED stays UNPROBED, and observing never
  compiles or loads an optional extension; this namespace has no `load=True`. Tests
  that need to initialize an optional library should keep that explicit operation
  separate from observation. `jt.capability`'s existing explicit loading interface
  remains available.

Capability records and device inventories **reject implicit boolean conversion**.
Callers must distinguish `.enabled`, `.failed`, `.disabled`, `.absent` and
`.unprobed`: a broken build must not become a skip. `present` describes the evidence
behind a capability and does not promise any particular numerical operation
succeeds; the device inventory is a read-only driver query, not a numerical
hardware test.

## Effective policy

```python
view = jt.introspection.policy
before = view.snapshot()
with jt.runtime.scope(no_grad=not before.runtime["no_grad"]):
    assert view.runtime.no_grad != before.runtime["no_grad"]
```

- `policy.startup` reads `jt.config` and `policy.runtime` reads
  `jt.runtime.context`; both support attribute and mapping access. Canonical runtime
  flags are in the runtime view; startup settings such as `cache_path`, compiler
  paths and architecture selection are in the startup view. Counters are not policy.
  An unknown key raises `KeyError` and an unknown attribute raises
  `AttributeError`.
- Views stay "live" across native scope changes. Returned dictionaries and lists are
  recursively frozen into mapping proxies and tuples, and nested `compile_options`
  cannot mutate native policy either. `policy.snapshot()` returns an immutable
  `PolicySnapshot(startup, runtime)` whose values are already separated from the
  source, so later policy changes do not rewrite an old snapshot. Assigning to or
  deleting through any observation namespace raises `AttributeError`.
- `runtime.use_cuda` is the default execution/placement policy and **does not prove
  devices are available**, nor does it describe where each tensor in a mixed
  CPU/CUDA graph lives. For availability see the capability layer; for a single
  tensor's location see the tensor placement API (see
  [devices and placement](device-placement.en.md)).

## Counters

- `counters` provides live read-only `exec_calls`, `allocator`, `held_vars`,
  `live_vars`, `live_ops`; `snapshot()` returns a detached immutable
  `CounterSnapshot`. The sole owner of the counts remains the native liveness query,
  and the query itself holds no extra tensor references.
- `allocator` is `AllocatorCounters(enabled, alloc_calls, allocated_bytes,
  free_calls, freed_bytes)`: the native instrumented allocator's **cumulative**
  allocation/free traffic, not RSS and not current usage. `enabled=False` states
  explicitly that there is no instrumentation. Native instrumentation mode changes
  reset these counts, so **subtracting across a mode change is not a valid traffic
  measurement**; this API offers no reset.
- `live_vars` is process-global and carries the "floor" of Vars already cached in
  the process. When asserting on leaks, call `jt.clean()` first, take the baseline,
  and compare a return to the baseline rather than to an absolute 0.
- Snapshots **do not submit pending graphs, do not synchronize devices and do not run
  GC**. They are successive observations, not atomic checkpoints across
  threads/devices; callers that need a "work is done" boundary synchronize
  explicitly outside the observation. `exec_calls` counts native executor
  invocations, not kernel launches, and is not a duration.

## Launch diagnostics

`jt.introspection.diagnostics.launch_history(backend="cuda", device=0, stream=None)`
returns detached text describing recent launch candidates. `stream=None` selects all
streams recorded for that device, and an integer selects that native stream handle
(not an ordinal). The query does not submit work, synchronize, or initialize a
backend; negative device indices and negative explicit stream handles are rejected.
The native history owns bounded, copied source locations and records that survive
graph release and hold no tensors or Python frames; the text states omissions and
truncation. **A candidate does not prove which operation triggered an asynchronous
fault**; CUDA error handling attaches this information automatically, so callers do
not need to query first. The contract is in
[asynchronous error diagnostics](../development/async-error-diagnostics.en.md).

For the mapping from the old idioms that test code migrated away from to these APIs,
see the "runtime introspection" section of
[the test system](../development/test-system.en.md).
