# ACL backend contracts

- Status: Accepted. Host-only contracts are implemented and gated; device
  acceptance on Ascend 910B3/CANN is a separate, still-required gate.
- Owner: ACL backend maintainers
- Recheck when: a runner stops using `BaseOpRunner::launch`, an attribute schema
  changes, a descriptor cache is attached to a runner, or `BackendOps` changes ABI

This page is the developer contract for the ACL (Ascend/CANN) backend: how
runners launch, how operator attributes travel from Python to C++, what the
descriptor cache shell guarantees, and what counts as device evidence. Build
layout, registration and the `jt.code(..., backend="acl")` source rule are in
[Source architecture](source-architecture.md) ("ACL kernel 注册" and
"后端回退策略"); installation and day-to-day use are in the
[Ascend 910B guide](../guides/ascend-910b.md).

## What host-only evidence proves

Code organization may proceed without the hardware. Host-only evidence (stub
SDK syntax checks, CPU-only decoder probes, static source contracts) can
establish a code boundary, but must not be reported as NPU validation. Real
Ascend 910B3/CANN acceptance remains a separate device gate, and every such run
must prove no CPU fallback.

## Shared launcher migration is closed for the standard owners

The standard ACL workspace/query/execute tail is centralized in
`BaseOpRunner::launch` (`backends/acl/kernels/native/base_op_acl.cc`). The last
standard owners that drove the aclnn execute call themselves --
SWhere, Sigmoid backward, BatchNorm forward and BatchNorm backward -- go through
it, each keeping its synchronous execution policy; BatchNorm backward still frees its output
mask after the launch has synchronised. A failed `aclnn` execute therefore
raises with the operator name and the decoded ACL status instead of printing a
line and leaving the output undefined while the graph continues.

The closing census was "all 71 `executeOp` owners are tail-free"; owners added
since then are written against `launch` directly, and the survey in
`tests/_helpers/acl_launch_tails.py` still finds no hand-rolled tail (it now
counts 80 owners). Reduce prod's three paths -- whole tensor, one axis, and a
staged multi-axis reduction over intermediate tensors -- each call `launch`, the
staged one asynchronously so its unconditional barrier still lands before the
intermediates are freed. Two owners keep a `syncRun()` of their own for a reason
the shared tail cannot express: the AdamW loop synchronises once after its last
step rather than once per tensor, and the staged product path synchronises after
freeing its intermediates. KVCacheMemcpy never had a tail; it is a per-token
`aclrtMemcpyAsync` path with no aclnn workspace executor, and it is outside the
attribute and descriptor contracts below.

The same conversion closed silent failures: a failed product used to return the
untouched output buffer as the reduction, and a failed `aclnnMaxDim`/
`aclnnMinDim` workspace query left both values and indices uninitialised. Both
raise now. The invariant (no family issues its own execute call, allocates its
own workspace, or handles its own query failure) is asserted by
`tests/structure/backends/acl/test_acl_launcher_contract.py` and
`test_acl_runner_failure_contract.py`, not by a site count.
`tools/build/acl_launch_program.py <old-tree> <new-tree>` reduces every owner
to its (workspace query, execute entry, sync policy) token stream, so a
launcher refactor can be shown to change no device-visible sequence.

Validate on an Ascend 910B3 after sourcing CANN and confirming the device:

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/acl/test_acl.py tests/backends/acl/test_aclop.py \
  2>&1 | tee "$TMPDIR/acl-launcher.log"
```

The run counts only if the owners actually executed on the NPU with
independent result/gradient checks. Each owner must synchronize inside
`forbid_backend_fallbacks()` and record a zero `backend_fallback_count()` delta.
Fallback attempts are NOT NPU validation, even if the request was rejected and
caught by the test. Keep the `execute launcher failed` and
`aclrtSynchronizeStream failed` diagnostics for failure attribution; log matching
is not the fallback acceptance gate. Repeat with `sync_run=0` to confirm the
asynchronous path still launches on ACL under the same checks. Hosts without
CANN and an Ascend device must not report hardware validation for launcher
changes.

## Syntax-checking ACL sources without CANN

A host without CANN can still parse the ACL translation units against a
generated stub SDK; see `agent/skills/acl-host-syntax-check`. That check catches
parse errors, unknown identifiers, and a workspace query passed where a launcher
belongs. It cannot check the arguments of an `aclnnXxxGetWorkspaceSize` call,
because those signatures are not knowable without the SDK.
It is not hardware validation.

## Structure boundaries

- `AclOpFunctions` type erasure is implemented: one uniform query callback and
  one checked execute pointer replace the signature-specific slots
  (`backends/acl/include/acl_op_registry.h`). The grouped unary/cast/binary/add
  query families keep their argument adaptation; other runners keep their typed
  direct queries. One immutable registry in `backends/acl/src/acl_jittor.cc` is
  shared across translation units, and all consumers and preflight signature
  checks use it.
- Attribute data plumbing is connected for the attribute owners: Softmax and
  its backward, Triu, Flip, Cumsum, Gather and Scatter as well as the
  convolution, normalization, pooling, indexing, attention and matmul families
  declare a schema in `acl_code_attributes.h`, and their Python builders send
  values through `_code.py` instead of interpolating them into generated
  source. The data-channel contract is below.
- Descriptor caching remains a host-only shell. No runner caches `aclTensor`
  descriptors yet; ownership and invalidation must be wired before any
  shape-keyed cache is attached, and descriptors must not be cached by shape
  while addresses remain mutable.

An attribute owner migrates together with its Python caller, forward/backward
attribute construction, wire schema and key contract. Changing only a generated
assignment leaves an incomplete boundary.

## Atomic attribute migration gate

| Field | Required invariant | Static evidence |
| --- | --- | --- |
| `schema_version`/`op` | version and registered owner are validated before decode | decoder rejects an unknown version or owner |
| scalar/vector value | type tag, required/default rule and canonical vector order are preserved | schema contract covers valid and malformed records |
| generated `OpAttr` | C++ receives decoded values without parsing generated source text | no attribute string interpolation for the owner |
| JIT/cache key | compiled code identifies owner/schema and tensor signature; attribute identity uses sorted typed values and the schema version | different attributes share source but keep distinct data; canonical keys exclude pointer/object identity |
| failure path | malformed data raises `UserError`; internal schema mismatch raises `InternalInvariantError` | negative cases are asserted before any ACL call |

The JIT source contains a fixed decoder call, not interpolated attribute values.
CodeOp owns the data map for each invocation. The decoded canonical key is not
the compiled kernel key: changing a runtime dimension does not recompile the
decoder. `apply_acl_code_attributes` memoizes decoding per distinct payload in a
bounded thread-local table, so a payload that changes every execution (a seeded
dropout) cannot grow it without limit. Descriptor caching and registry changes
stay separate atomic changes from an attribute slice so each can be reviewed and
rolled back on its own.

## Data-channel schema and decoder

The wire schema is a versioned, operator-scoped map:

- `schema_version`: integer, currently `1`, required;
- `op`: immutable string matching the registered ACL owner, required;
- scalar fields: typed `int64`, `float64`, or `bool`; absent fields use the
  operator's documented default, never an implicit zero;
- vector fields: typed homogeneous `int64[]`/`float64[]`/`bool[]`; absent vectors
  use an explicit empty/default value;
- `cache_key`: sorted `(field_name, type_tag, value)` tuples plus schema version;
  pointer addresses and Python object ids are forbidden.

Non-finite floating-point values are rejected. The C++ key serializer always
uses `std::locale::classic()`, so a host with a comma decimal separator
serializes the same float as one with a period, and keys stay stable when a
graph is prepared on one host and executed on another.

The shared C++ decoder boundary is `backends/acl/include/aclops/acl_data_channel.h`.
The production adapter `acl_code_attributes.h` is a consumer; the decoder
validates the
operator name, schema version, type tag, and required fields before an owner
constructs an `OpAttr`. The header has no
ACL/CANN include and can be compiled on a CPU-only host. Its CodeOp wire bridge
is `acl_code_data.h`; attribute classes and runner assignment stay in the
CANN-aware adapter.

```cpp
AclDecodedData decode_acl_data(
    const AclDataRecord& record, const std::string& expected_op,
    const AclAttrSchema& schema, std::string& canonical_cache_key);
```

The Python half is `backends/acl/kernels/ops/acl_data.py`, with no CANN
dependency. `validate_acl_data()` applies schema defaults, rejects unknown or
wrongly typed fields and emits an address-independent `canonical_cache_key`.
`encode_code_data()` sends the record through the existing string-to-double
DataMap: signed int64 values use two exact uint32 lanes, so values beyond 2^53
are not rounded. Unknown keys within the reserved prefix are rejected; other
CodeOp data is untouched.

`AclDataOwner` holds an immutable operator name and schema copy and exposes
`op()`, `schema()`, `decode(record, canonical_cache_key)` and
`consume(record, canonical_cache_key, consumer)`. Constructing an owner
validates its schema (including an invalid type tag on a field without a
default) and rejects an empty operator as `InternalInvariantError`, so a bad
registry declaration is never misclassified as caller data. Malformed user data
-- an unknown field, wrong type, missing required value, a
non-canonical vector representation, or an unsupported schema version -- raises
`UserError` before any ACL call.

`consume()` passes the callback an `AclDataView`: a borrowed, non-copyable and
non-movable view with typed accessors (`int64`, `float64`, `boolean` and the
three vector forms), `has(name)`, and the validated metadata. A consumer asking
for a wrong type or an absent field gets `InternalInvariantError`. The view is
valid only during the callback: a launcher copies values into its own attribute
object and must not retain the view. `AclAttrRunnerContract` adds a fixed
`AclAttrBinding` list on top; construction rejects duplicate, undeclared or
type-incompatible bindings, and its view rejects fields outside the bindings.

## Descriptor identity and cache shell

An attribute key alone is not a descriptor identity. The descriptor key also
contains the complete shape (an empty shape is the valid zero-dimensional case),
dtype, layout and device; strings are length-prefixed under a fixed key version,
and negative dimensions, empty metadata, pointer values and Python object ids
are rejected before lookup. A descriptor prepared for another device, layout or
shape cannot be reused because operator and attributes match.

`AclDescriptorKey`, `canonical_descriptor_key()` and `make_descriptor_key()`
implement this in `acl_data_channel.h`; `descriptor_cache_key()` and
`DescriptorCache` mirror it in `acl_data.py`, validating the full key before
every mutating or lookup entry point. `AclDescriptorCache<T>` is a lifecycle
shell: the caller supplies the value and a builder, and the shell never creates,
aliases or retains a raw `aclTensor`. Both halves provide:

- build-once `get_or_create`, and `acquire(key)` leases on existing entries;
- `is_current(handle)`/`get(handle)`, which fail closed with an internal error
  on a stale or malformed lease, and `release(handle)`, a no-op (`false`) for a
  stale lease, so a delayed callback cannot erase a rebuilt equivalent entry;
- `erase(key)` before an owner releases or replaces a device allocation (a
  per-key tombstone epoch keeps older leases stale even if the same key is
  rebuilt), `erase_device(device)` for allocator/context teardown (advances that
  device's generation even when it has no entries), and `clear()` for global
  teardown;
- `device_size(device)`, a read-only teardown diagnostic counting live entries
  of one device.

Generations are lifecycle bookkeeping and are absent from the canonical key, so
an equivalent descriptor rebuilt after teardown has the same identity while old
handles stay detectably stale. A future CANN runner may replace the value with
an RAII `aclTensor` owner; it must release that value before invalidating the
key and keep the schema checks, key fields, generation invalidation and the
user/internal error split.

## Migration order

1. **Implemented:** data-channel schema, CodeOp wire encoding and C++ decoder.
2. **Implemented:** attribute owners consume runtime attributes through the
   channel, forward and backward together.
3. **Host-only prerequisite defined:** descriptor identity and the cache shell
   above. Next: device-side address rebinding and invalidation, then attaching
   the shell to a real ACL runner. A shape cache must never reuse a descriptor
   with a stale address.
4. **Implemented:** `AclOpFunctions` erased query plus checked execute ABI, one
   immutable registry, typed runner arguments and the synchronous/asynchronous
   launch policies preserved.

## Host-only gates

```bash
python -m pytest -q \
  tests/structure/backends/acl/test_acl_data_channel_contract.py \
  tests/structure/backends/acl/test_acl_data_schema_normalizer.py \
  tests/structure/backends/acl/test_acl_production_attributes.py \
  tests/structure/backends/acl/test_acl_launcher_contract.py
```

`test_acl_data_channel_contract.py` compiles and executes a consumer probe on a
CPU-only host (defaults, vector order, cache-key identity, wrong-type failure,
binding whitelist and registration failures). Passing these is not evidence
that an Ascend device executed anything.

## Device acceptance

Run on an Ascend 910B3 after sourcing CANN:

```bash
source "$CANN_SET_ENV"
npu-smi info
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
  python -m pytest -q -s tests/backends/acl/test_acl.py
```

A run is accepted only when the intended ACL operator executes, independent
values/gradients and residency pass, the native `backend_fallback_count()` delta
is zero, and `npu-smi info` confirms the target card. The `tests/backends/acl`
autouse fixture applies `forbid_backend_fallbacks()` when Jittor is already
loaded; each test must synchronize or fetch inside that scope. Standalone probes
and tests outside that directory import the scope explicitly:

```python
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    result = run_owner()
    jt.sync_all(True)
    actual = result.numpy()
assert jt.core.backend_fallback_count() == before
```

`run_owner()` is the operation under validation; construct its device inputs
inside the scope too. The scope performs no implicit synchronization.
`jt.runtime.backend_fallback` accepts `error`, `warn` and `allow`; only `error`
is an acceptance policy. The counter includes rejected attempts, so catching an
unsupported-operation exception does not make a test pass. Only
preflight unsupported decisions may request fallback; execution exceptions propagate after
cleanup and must not retry on CPU. `warn` and `allow` are explicit
debugging policies. SDK/launcher logs remain useful for failure attribution, but the
absence of a fallback log line is not evidence of no CPU fallback.

For an attribute owner, also run its exact node, for example:

```bash
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
  python -m pytest -q -s tests/backends/acl/test_aclop.py -k 'softmax or triu'
```

Record the card model, CANN version, selected node, device residency and the
zero fallback-attempt delta with the result (see
[NPU validation templates](npu-validation-templates.md)). A host-only or static
pass never closes device acceptance.
