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

## Schema draft

The proposed wire schema is a versioned, operator-scoped map:

- `schema_version`: integer, currently `1`, required;
- `op`: immutable string matching the registered ACL owner, required;
- scalar fields: typed `int64`, `float64`, or `bool`; absent fields use the
  operator's documented default, never an implicit zero;
- vector fields: typed homogeneous `int64[]`/`float64[]`/`bool[]`; absent vectors
  use an explicit empty/default value;
- `cache_key`: sorted `(field_name, type_tag, value)` tuples plus schema version;
  pointer addresses and Python object ids are forbidden.

The C++ decode entry should be one `BaseOpRunner` helper that validates the
operator name, schema version, type tag, and required fields before constructing
an `OpAttr`. This is a design target only; no such shared decoder exists yet.

## Migration order

1. Define the data-channel schema and its cache-key representation for one
   non-pooled owner, with a static generated-code contract.
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
