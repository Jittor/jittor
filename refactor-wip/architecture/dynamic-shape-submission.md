# Dynamic shapes and submission boundaries

Status: implementation contract for refactor task 3.05. CPU construction,
composition, scalar readback and fetch tests cover the boundary. The integrated
CUDA where, CUB where, candidate and fetch readbacks passed real-device checks.

Operator constructors and `Op::init()` build edges, infer shape metadata and
capture autograd policy. They do not submit work or invoke callbacks. A negative
extent records an allocation upper bound for a data-dependent output; the
operator's execution must replace it with the actual nonnegative shape.

`schedule_pending_from_python()` runs after a complete Python holder has been
created. `Executor::submit_pending()` resolves a dynamic result's required
subgraph there, preserving the existing concrete Python shape contract. Ordinary
static results retain the configured lazy/eager submission policy.

Native C++ composition does not require an intermediate Python holder.
`Executor::run_sync()` resolves internal dynamic inputs in dependency order
before constructing its execution plan, then refreshes downstream `infer_shape()`
metadata. This is an execution-stage operation, never constructor recursion.
An explicitly submitted dynamic output that remains unresolved fails rather
than publishing a negative concrete shape.

`VarHolder::item()` uses `sync(false, false)`: it submits the requested dependency
graph without weak-sync expansion into sibling branches. CUDA device-to-host
copies use a producer-stream event. Native where, candidate and CUB where count
readbacks use the same `backend_copy()` gateway; allocator and GIL policies
remain owned by that gateway and the executor entry scope.

Fetch construction only creates a fetch node. The public Python `fetch()` calls
`core.submit_pending_fetches()` after construction to execute ready inputs and
enforce bounded pending-fetch backpressure. Native callers can use that same
explicit boundary or submit their fetch graph normally. Shutdown device waits
remain cleanup operations and are not removed by this contract.

For concurrent development, keep these ownership boundaries:

- New dynamic operators implement upper-bound output metadata, pure
  `infer_shape()`, and execution-time final shape publication. Do not restore a
  constructor-side `run_sync()` workaround.
- Frontends submit only after their language objects and graph edges exist.
  They do not duplicate the executor's shape dependency traversal.
- Backend providers implement ordered copies and event waits behind runtime
  callbacks. Operator families do not introduce whole-device readback waits.
- Changes to construction, `submit_pending`, shape resolution and fetch
  backpressure must be coordinated as one core interface change; ordinary
  operator or frontend additions consume those interfaces independently.

Focused regressions: `tests/compiler/test_dynamic_shape_submission.py`,
`tests/ops/test_where_op.py`, `tests/data/test_fetcher.py`. These exercise native
construction without execution, native where/binary/reduce composition,
Python-visible shapes, sibling exclusion and fetch callback/backpressure.
