# ACL Structure Migration Boundary

The standard ACL workspace/query/execute tail is already centralized in
`BaseOpRunner::launch`. The remaining 8.06 work is intentionally split into
three dependent migrations:

- `AclOpFunctions` type erasure: migrate the fields, constructors, registry,
  and every query consumer as one change; do not replace one field in isolation.
- Attribute data plumbing: migrate `aclops/_code.py` and its Python callers
  together, preserving generated operator arguments and cache keys.
- Descriptor caching: establish ownership and invalidation rules before adding
  shape-keyed caches; do not cache descriptors by shape while addresses remain
  mutable.

`KVCacheMemcpy` is outside this contract. It is a per-token
`aclrtMemcpyAsync` path without an ACL workspace executor.

The ACL structure migrations require a real Ascend 910B3/CANN run before being
marked complete. The host-only contract is source/static validation and must
not be reported as NPU validation; every real run must prove no CPU fallback.

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

The host-only C++ decoder boundary is now defined in
`python/jittor/extern/acl/aclops/acl_data_channel.h`. It is one shared decoder
boundary. The `BaseOpRunner` helper is the future consumer; the decoder
validates the
operator name, schema version, type tag, and required fields before an owner
constructs an `OpAttr`. The header has no
ACL/CANN include and can be compiled on a CPU-only host; it is deliberately not
wired into an ACL runner until the first owner migrates its schema, generated
attribute construction, and JIT key atomically.

The Python host-side half of this contract lives in
`python/jittor/extern/acl/aclops/acl_data.py`. `validate_acl_data()` applies
schema defaults, rejects unknown or wrongly typed fields, and emits an
address-independent `canonical_cache_key`. It has no CANN dependency and does
not change the existing generated `OpAttr` path; the module is therefore safe
to exercise on a CPU-only host. The negative contract is covered by
`tests/structure/test_acl_data_schema_normalizer.py`.

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

`tests/structure/test_acl_data_channel_contract.py` compiles and executes a
consumer probe on a CPU-only host. That probe checks defaults, vector order,
cache-key identity, and the wrong-type failure path. Passing this contract is
not evidence that an ACL operator has been migrated or that an Ascend device
executed anything.

`AclAttrRunnerContract` is the next host-only seam above that view. A registry
entry gives it an operator schema plus a fixed `AclAttrBinding` list; construction
rejects duplicate, undeclared, or type-incompatible bindings as an
`InternalInvariantError`. Its `consume()` decodes once, verifies every bound
field, and then invokes the generated/static consumer with the same bounded
`AclDataView`. This freezes attribute names and types before a future
`BaseOpRunner` adapter constructs `AclOpAttr`; it does not include `base_op.h`,
allocate an ACL object, or make a CANN call. The CPU-only contract probe covers
the valid path and all three registration failures.

Malformed user data (unknown field, wrong type, missing required value, a
non-canonical vector representation, or an unsupported schema version) raises
`UserError`; a violated internal schema
raises `InternalInvariantError`. The helper produces the canonical cache key
from sorted typed values before the owner constructs its `OpAttr`. Vectors keep
their semantic order while the map and field encoding are deterministic. It
must not read pointer addresses, process-global state, or Python object identity.
The host-only compile and runtime contract is covered by
`tests/structure/test_acl_data_channel_contract.py`.

## Migration order

1. **Done on the host:** define the data-channel schema and its cache-key
   representation, with a C++14 static/generated-code contract. This does not
   claim that an ACL operator consumes the channel yet.
2. Migrate `softmax.dim` or `triu.diagonal` as the first attribute owner; keep
   `pool_op.py` out of this step because pooled descriptors have a separate
   lifetime/cache contract.
3. Define descriptor address rebinding and invalidation, then add shape-keyed
   caching. A shape cache must never reuse a descriptor with a stale address.
4. Migrate `AclOpFunctions` type erasure only after all query signatures and
   registry entries have a single launcher representation.

The real-device acceptance command is intentionally explicit and must run on
an Ascend 910B3 after sourcing CANN:

```bash
source "$ASCEND_HOME/set_env.sh"
npu-smi info
JITTOR_TEST_DEVICES=npu sync_run=1 \
  python -m pytest -q -s tests/backends/npu/test_acl.py
```

The run is accepted only when the intended ACL operator executes, the log has
no `fallback cpu`/`cpu fallback`, and `npu-smi info` confirms the target card.

For an attribute owner, the device gate must additionally use its exact test
node (replace the example with the owner-specific node after the slice lands):

```bash
source "$ASCEND_HOME/set_env.sh"
npu-smi info
JITTOR_TEST_DEVICES=npu sync_run=1 \
  python -m pytest -q -s tests/backends/npu/test_acl_torch_compat.py -k 'softmax or triu'
```

Record the card model, CANN version, selected node, and the absence of CPU
fallback in the handoff. A host-only/static pass never closes the ACL task.
