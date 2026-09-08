# ACL CodeOp attribute wire contract

- Status: Shared bridge implemented and cross-language host verified
- Reviewed: 2026-09-08
- Owner: ACL integration maintainers
- Recheck when: schema version/type codes or CodeOp's DataMap representation changes

Python `encode_code_data(record, *, expected_op=None, schema=None,
prefix="acl_attr.")` returns a fresh string-to-double dictionary. C++
`jittor::acl_data::decode_code_data(data, expected_op, schema, prefix)` returns
`AclDecodedData` through the existing `decode_acl_data` semantic validator.
Neither entry imports or calls CANN.

The reserved prefix contains `version`, `op.<hex UTF-8 name>` (value 1), and
`fields`. Each sorted field index has `field.<index>.name.<hex UTF-8 name>`
whose value is its type code: int64=0, float64=1, bool=2, int64[]=3,
float64[]=4, bool[]=5. A scalar uses `value`, except signed int64 uses exact
unsigned 32-bit `lo`/`hi` lanes representing its two's-complement bits.
Vectors add `length` and `item.<index>.` scalar slots. All keys are prefixed.
Metadata and integer lanes must be finite integral doubles in their stated
range. The C++ conversion covers INT64_MIN/INT64_MAX without an out-of-range
signed cast. Finite float64 values, including negative zero, retain their bits.

The decoder requires canonical field order, exact schema types, matching owner
and version, and rejects every unconsumed key inside the reserved prefix.
Unrelated CodeOp keys are ignored and the input map is const. Missing optional
defaults are resolved by the existing decoder; invalid defaults remain internal
schema errors. Python now also rejects int64 overflow and a record type tag
that differs from its declaration, including malformed schema defaults.

Validation command:

```sh
JITTOR_TORCH_SHIM=1 python -m pytest -q \
  tests/structure/backends/acl/test_acl_code_data_wire.py \
  tests/structure/backends/acl/test_acl_data_schema_normalizer.py
```

Result: **44 passed, 0 skipped, 1.38 seconds**. The test compiles the actual
header once, transfers Python-encoded double/key pairs to the C++ decoder,
and reads its typed result back. It covers both int64 endpoints and values
beyond 2^53, all scalar/vector types, empty vectors, signed zero, custom prefix,
Unicode operator identity, C++ defaults and malformed headers/type/order/lanes.
An unrelated NaN-valued CodeOp key is preserved without being validated as an
ACL attribute. Ruff checks also passed.

## Production integration

The same batch connects Softmax/SoftmaxBackward, Triu, Flip, Cumsum, Gather
and Scatter. Python owner methods call the shared `acl_code(attributes=...)`;
the generated source calls `apply_acl_code_attributes(op, data)`, which decodes
the payload and constructs the original C++ attribute class. Forward/backward
dimension and vector values travel through data instead of source literals.
Different values share generated code but retain independent per-invocation
payloads. Cumsum's backward Flip/Cumsum/Flip chain and separate Softmax contexts
retain their own attributes. Reserved-prefix collisions and mixed attr_code /
attributes requests are rejected before jt.code is called.

Root verification combined the wire suite with four production-boundary tests:
47 passed / 1 failed in 2.74 seconds. The failure was the recording harness not
accepting jt.code's three positional arguments; after correcting that harness,
the failed node passed in 0.06 seconds. No production fix was needed for that
failure. The C++ test feeds Python-produced payloads into the actual decoder,
schemas, attribute classes and assignment function; only the runner carrier
and SDK declarations are stand-ins. A separate TU instantiates the adapter
against the real BaseOpRunner and passed the ACL host syntax checker.

No CANN/NPU execution is claimed. These are code-organization and host-boundary
checks, not numerical certification of these owners' pre-existing mathematics.
On Ascend, compile with the real SDK first, then check actual NPU residency,
zero fallback, distinct positive/negative dimensions, axis order, repeated
invocations and forward/backward values against independent references. The
entry and strict fallback scope are in the
[Ascend guide](../guides/ascend-910b.md) and
[attribute migration contract](../guides/acl-structure-boundary.md).
Remaining owner families and production descriptor caching still keep 8.06 open.
