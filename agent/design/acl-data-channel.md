# ACL Data and Descriptor Boundary

This document describes the host-only boundary used while ACL/CANN is not
available. It is an implementation contract, not a claim that an ACL runner
is wired on the host.

## Attribute records

`acl_data.py` and `acl_data_channel.h` validate the same value classes:
`int64`, `float64`, `bool`, and their vector forms. A record carries an
operator name and schema version. Unknown fields, missing required fields,
schema/type mismatches, and non-finite floating-point values are rejected
before a backend call. Defaults are materialized into owned storage.

The canonical attribute key is deterministic: fields are ordered by name,
floating-point formatting uses the classic locale, and no pointer or object
identity is included. The C++ decoder and Python normalizer intentionally do
not import ACL/CANN.

## Descriptor identity

An attribute key alone is not a descriptor identity. The descriptor key also
contains shape, dtype, layout, and device. The key is an immutable value and
the cache stores caller-provided values; the host-only cache never creates or
owns an `aclTensor`.

Before CANN integration, a runner must use `erase(key)` when one descriptor
owner is released and `erase_device(device)` during device teardown. Teardown
increments a per-device generation even when the cache has no entry. A runner
holding a descriptor handle records the generation at creation and rejects it
when the current generation differs. `clear()` is global teardown and advances
all generations observed by the cache.

The generation is lifecycle bookkeeping only. It is deliberately absent from
the canonical descriptor key, so recreating an equivalent descriptor after
teardown yields the same identity while an external owner can still detect a
stale handle.

## CANN handoff

The next ACL implementation step may replace the cache value with an RAII
`aclTensor` owner and connect the existing decoder to an ACL runner. It must
preserve the schema/version checks, canonical key fields, per-device
generation invalidation, and fail-closed user/internal error split. These
contracts are covered by the host-only structure tests; they require no CANN
installation and do not count as 910B3/NPU hardware validation.
