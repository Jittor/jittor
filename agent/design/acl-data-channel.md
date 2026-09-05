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

The host-only cache exposes this lifecycle as a value-only handle contract:
`acquire(key)` may only lease an existing entry, `is_current(handle)` checks
both key presence and the device generation, and `get(handle)` fails closed
with an internal error for a stale or malformed lease. A single-key `erase`
invalidates the lease by removing its entry; device/global teardown also
advances generations so an external owner cannot accidentally reuse a handle
after recreating an equivalent descriptor. The handle contains no pointer,
allocator, or CANN type and is therefore safe to carry across the future
decoder/runner boundary.

The cache also keeps a host-only per-key tombstone epoch. This means a handle
from before `erase(key)` remains stale even if the same canonical key is built
again immediately; checking only the device generation would incorrectly make
that old handle appear live.

The generation is lifecycle bookkeeping only. It is deliberately absent from
the canonical descriptor key, so recreating an equivalent descriptor after
teardown yields the same identity while an external owner can still detect a
stale handle.

## Consumer view lifetime

`AclDataOwner::consume()` and `AclAttrRunnerContract::consume()` decode into
storage owned by the call, then pass an `AclDataView` to the consumer callback.
The view is intentionally borrowed and is non-copyable/non-movable at the C++
boundary. A future launcher must copy scalar/vector values into its own ACL
attribute object while the callback is active; it must not retain the view or a
reference to it. This keeps the host-only contract aligned with the eventual
RAII runner and makes an accidental use-after-consume a compile-time error for
the common copy/move paths.

## CANN handoff

The next ACL implementation step may replace the cache value with an RAII
`aclTensor` owner and connect the existing decoder to an ACL runner. It must
preserve the schema/version checks, canonical key fields, per-device
generation invalidation, and fail-closed user/internal error split. These
contracts are covered by the host-only structure tests; they require no CANN
installation and do not count as 910B3/NPU hardware validation.
