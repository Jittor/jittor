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
