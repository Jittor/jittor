# ACL Structure Migration Boundary

The standard ACL workspace/query/execute tail is already centralized in
`BaseOpRunner::launch`. Reviewed 2026-09-08: the following three boundaries have
different implementation status. The board's earlier claim that only type
erasure remained was too narrow; host schema/cache shells do not establish
that the production attribute or descriptor paths use them.

- `AclOpFunctions` type erasure is implemented: one uniform query callback and
  one checked execute pointer replace the signature-specific slots. Four live
  query families retain their argument adaptation; other runners keep their
  already typed direct queries. One immutable registry in `acl_jittor.cc` is
  shared across translation units. All consumers and preflight signature checks
  use that registry; no header diagnostic is filtered in host syntax checks.
- Attribute data plumbing remains: migrate `backends/acl/kernels/ops/_code.py` and its Python callers
  together, preserving generated operator arguments and cache keys.
- Descriptor caching remains: establish ownership and invalidation rules before adding
  shape-keyed caches; do not cache descriptors by shape while addresses remain
  mutable.

`KVCacheMemcpy` is outside this contract. It is a per-token
`aclrtMemcpyAsync` path without an ACL workspace executor.

The user permits code organization to proceed without the missing hardware.
Host-only evidence can establish that code boundary, but must not be reported
as NPU validation. Real Ascend 910B3/CANN acceptance remains a separate device
gate and every such run must prove no CPU fallback.

Candidate attribute owners reviewed for an isolated slice were `triu.diagonal`,
`softmax.dim`, and `flip.axes`. None is safe to move alone: each currently
serializes an `OpAttr` assignment in generated C++ while the corresponding
Python call also determines the JIT key. Move them only after the data-channel
schema and cache-key contract are defined for every owner.

## Atomic attribute migration gate

An attribute owner may enter the implementation queue only when all of the
following fields are migrated in the same change. A partial change is not a
valid intermediate state and must remain a design-only patch:

| Field | Required invariant | Static evidence before device work |
| --- | --- | --- |
| `schema_version`/`op` | version and registered owner are validated before decode | decoder contract rejects an unknown version or owner |
| scalar/vector value | type tag, required/default rule, and canonical vector order are preserved | schema contract covers valid and malformed records |
| generated `OpAttr` | C++ receives decoded values without parsing generated source text | source check finds no attribute string interpolation for the owner |
| JIT/cache key | key is made from sorted typed values and schema version | key contract excludes pointer/object identity and address values |
| failure path | malformed data raises `UserError`; internal schema mismatch raises `InternalInvariantError` | negative cases are asserted before any ACL call |

The first implementation slice must complete this table for one owner (and its
Python caller, C++ decoder, generated call, and cache key) in one commit. Do
not migrate only `softmax.dim`, only `triu.diagonal`, or only `_code.py`.
Descriptor caching and `AclOpFunctions` type erasure remain separate atomic
changes; combining them with an attribute slice makes rollback and review
ambiguous.

## Schema and host-only decoder boundary

The proposed wire schema is a versioned, operator-scoped map:

- `schema_version`: integer, currently `1`, required;
- `op`: immutable string matching the registered ACL owner, required;
- scalar fields: typed `int64`, `float64`, or `bool`; absent fields use the
  operator's documented default, never an implicit zero;
- vector fields: typed homogeneous `int64[]`/`float64[]`/`bool[]`; absent vectors
  use an explicit empty/default value;
- `cache_key`: sorted `(field_name, type_tag, value)` tuples plus schema version;
  pointer addresses and Python object ids are forbidden.

The C++ cache-key serializer always uses `std::locale::classic()`.  It must not
inherit the process locale: a host configured with a comma decimal separator
must still serialize the same float value as a host configured with a period.
This keeps generated/cache keys stable when a graph is prepared on one host and
executed or restored on another.

The host-only C++ decoder boundary is now defined in
`backends/acl/include/aclops/acl_data_channel.h`. It is one shared decoder
boundary. The `BaseOpRunner` helper is the future consumer; the decoder
validates the
operator name, schema version, type tag, and required fields before an owner
constructs an `OpAttr`. The header has no
ACL/CANN include and can be compiled on a CPU-only host; it is deliberately not
wired into an ACL runner until the first owner migrates its schema, generated
attribute construction, and JIT key atomically.

The Python host-side half of this contract lives in
`backends/acl/kernels/ops/acl_data.py`. `validate_acl_data()` applies
schema defaults, rejects unknown or wrongly typed fields, and emits an
address-independent `canonical_cache_key`. It has no CANN dependency and does
not change the existing generated `OpAttr` path; the module is therefore safe
to exercise on a CPU-only host. The negative contract is covered by
`tests/structure/backends/acl/test_acl_data_schema_normalizer.py`.

The C++ interface is:

```cpp
AclDecodedData decode_acl_data(
    const AclDataRecord& record, const std::string& expected_op,
    const AclAttrSchema& schema, std::string& canonical_cache_key);
```

The registry-facing owner wrapper is `AclDataOwner`. It owns an immutable
operator name and schema copy and exposes `op()`, `schema()`, and
`decode(record, canonical_cache_key)`. A future ACL registry entry should hold
one owner rather than pass an operator string and a temporary schema through
each launcher call. Constructing an owner validates its schema and rejects an
empty operator as `InternalInvariantError`; decoding still classifies caller
data as `UserError` before any ACL call. This owner boundary is host-only and
does not claim that an ACL launcher consumes the channel yet.

The owner also exposes `consume(record, canonical_cache_key, consumer)`. The
callback receives an `AclDataView`, a short-lived read-only view with typed
accessors (`int64`, `float64`, `boolean`, and the three vector forms),
`has(name)`, and the validated operator/schema/cache-key metadata. The view
checks the declaration again before returning a value and classifies a
consumer asking for a wrong type or absent field as
`InternalInvariantError`. This is the C++ decoder-to-attribute-consumer
interface: an eventual `OpAttr` adapter can consume typed values without
parsing generated source text or touching the decoder's map. It deliberately
creates no ACL/CANN object, and the view cannot outlive the callback.

`tests/structure/backends/acl/test_acl_data_channel_contract.py` compiles and executes a
consumer probe on a CPU-only host. That probe checks defaults, vector order,
cache-key identity, and the wrong-type failure path. Passing this contract is
not evidence that an ACL operator has been migrated or that an Ascend device
executed anything.

`AclAttrRunnerContract` is the next host-only seam above that view. A registry
entry gives it an operator schema plus a fixed `AclAttrBinding` list; construction
rejects duplicate, undeclared, or type-incompatible bindings as an
`InternalInvariantError`. Its `consume()` decodes once, verifies every bound
field, and invokes the generated/static consumer with a bounded `AclDataView`
whose `has()`/typed accessors reject fields outside that binding list. This
freezes attribute names and types before a future `BaseOpRunner` adapter
constructs `AclOpAttr`; it does not include `base_op.h`, allocate an ACL object,
or make a CANN call. The CPU-only contract probe covers the valid path, the
binding whitelist, and all three registration failures.

Malformed integration schemas are rejected at owner construction, including
an invalid type tag on a field without a default. This keeps a bad registry
declaration from being misclassified later as caller data. Malformed user data
(unknown field, wrong type, missing required value, a
non-canonical vector representation, or an unsupported schema version) raises
`UserError`; a violated internal schema
raises `InternalInvariantError`. The helper produces the canonical cache key
from sorted typed values before the owner constructs its `OpAttr`. Vectors keep
their semantic order while the map and field encoding are deterministic. It
must not read pointer addresses, process-global state, or Python object identity.
The host-only compile and runtime contract is covered by
`tests/structure/backends/acl/test_acl_data_channel_contract.py`.

## Descriptor identity and cache shell

The descriptor cache key is a second boundary after attribute decoding.  It
must include the decoded attribute key, the complete shape (an empty shape is
the valid zero-dimensional case), dtype, layout, and device identity.  The
canonical form is length-prefixed for strings and uses a fixed key version;
negative dimensions, empty metadata, pointer values, and Python object ids are
rejected before a cache lookup.  Consequently a descriptor prepared for
`npu:0`, a different layout, or a different shape cannot be reused merely
because its operator and attributes match.

`AclDescriptorKey`, `canonical_descriptor_key()`, and
`make_descriptor_key()` in `acl_data_channel.h` implement this validation
without including CANN.  `AclDescriptorCache<T>` is only a lifecycle shell:
the caller supplies `T` (a test value on a host, or a future owned ACL
descriptor on 910B3) and a builder; the shell never creates, aliases, or
retains a raw `aclTensor` pointer.  A repeated key invokes the builder once,
while a shape/layout/device change creates an independent entry.  The owner
must call `erase(key)` before releasing or replacing a device allocation; this
invalidates only that identity and prevents a shape-equivalent descriptor from
retaining a stale address.  `erase_device(device)` invalidates every entry for
one device during allocator/context teardown, while `clear()` remains the full
teardown escape hatch.  Neither operation inspects or owns an ACL handle; the
eventual CANN runner must release its value before invalidating the key.  This
is a host-only prerequisite, not evidence that ACL descriptors are already
cached or that addresses are correctly rebound on device.

The Python mirror is `descriptor_cache_key()` plus `DescriptorCache` in
`acl_data.py`.  It returns an immutable tuple derived from the normalized
schema key and metadata, and has the same build-once semantics.  Both halves
expose single-key invalidation (`erase`) as well as full teardown.  Both halves
are intentionally separate from the generated `OpAttr` path until one
attribute owner can migrate decode, attribute construction, descriptor
address rebinding, and invalidation atomically.

The Python cache validates the complete canonical key before every mutating
or lookup entry point (`get_or_create`, `acquire`, `erase`, and membership).
Malformed tuples therefore fail before a value can enter the cache, matching
the C++ `canonical_descriptor_key()` boundary. This is important for a future
device owner: an invalid key must not be inserted and only rejected later
during lease acquisition or teardown.

## Migration order

1. **Done on the host:** define the data-channel schema and its cache-key
   representation, with a C++14 static/generated-code contract. This does not
   claim that an ACL operator consumes the channel yet.
2. Migrate `softmax.dim` or `triu.diagonal` as the first attribute owner; keep
   `pool_op.py` out of this step because pooled descriptors have a separate
   lifetime/cache contract.
3. **Host-only prerequisite now defined:** validate the descriptor identity
   key and cache ownership shell above. Define device-side descriptor address
   rebinding and invalidation next, then add the shell to a real ACL runner. A
   shape cache must never reuse a descriptor with a stale address.
4. **Implemented:** `AclOpFunctions` uses a uniform erased query plus a checked
   execute ABI. The immutable registry is defined once; typed runner arguments
   and the synchronous/asynchronous launch policies are preserved.

The real-device acceptance command is intentionally explicit and must run on
an Ascend 910B3 after sourcing CANN:

```bash
source "$ASCEND_HOME/set_env.sh"
npu-smi info
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
  python -m pytest -q -s tests/backends/acl/test_acl.py
```

The run is accepted only when the intended ACL operator executes, independent
values/gradients and residency pass, the native `backend_fallback_count()` delta
is zero, and `npu-smi info` confirms the target card. The NPU directory's autouse
fixture applies `forbid_backend_fallbacks()` when Jittor is already loaded;
each test must synchronize or fetch inside that scope. Standalone probes and
tests outside that directory must import the scope explicitly:

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

Here `run_owner()` is the owner-specific operation under validation; construct
its device inputs inside the scope as well. The scope performs no implicit
synchronization. `jt.runtime.backend_fallback` accepts `error`, `warn`, and
`allow`; only `error` is an acceptance policy. The counter includes rejected
attempts, so catching an unsupported-operation exception does not make a test
pass. Only preflight unsupported decisions may request fallback. Execution
exceptions propagate after cleanup; they must not retry on CPU. `warn` and
`allow` are explicit debugging policies. SDK/launcher logs remain useful for
failure attribution, but absence of CPU-compilation or fallback log messages
is not evidence of no CPU fallback.

For an attribute owner, the device gate must additionally use its exact test
node (replace the example with the owner-specific node after the slice lands):

```bash
source "$ASCEND_HOME/set_env.sh"
npu-smi info
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
  python -m pytest -q -s tests/backends/acl/test_acl_torch_compat.py -k 'softmax or triu'
```

Record the card model, CANN version, selected node, device residency, and zero
native fallback-attempt delta in the handoff. A host-only/static pass never
closes the ACL task.
