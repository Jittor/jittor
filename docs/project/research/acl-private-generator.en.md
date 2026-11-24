# Private ACL Generator: isolated state and operator integration

[Research proposals](index.en.md) · [中文](acl-private-generator.md)

Status: **proposal, not implemented**. Maintainer discussion is tracked in
[#18162](https://github.com/Jittor/jittor/issues/18162). This document does not
declare private ACL Generators, seeded sampling or offset/discard support.
Runtime and compat maintainers must review the state representation, algorithm
and consumption units before implementation.

## Problem and scope

The default device RNG and a private Generator have separate contracts. Passing
seed replay and global state restoration does not establish that
`tensor.exponential_(generator=g)` consumes the independent state of `g`.
Successful construction and `manual_seed` do not establish it either.

Legal inference paths without request seeds already exist. This gap must not be
described as all Engines or all ACL random operations being unusable. Affected
callers include explicitly seeded requests, interleaved request owners, and some
warmup/discard paths. Each requires its own acceptance; greedy output is not
private RNG acceptance.

### Reviewed source boundary

These source findings are pinned to
`f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb`, without claiming device revalidation
of that commit. Historical hardware results, commands, outputs and matching Core
identity belong in the linked PR.

1. The [compat Generator](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/compat/torch/generator.py#L19)
   exposes CPU/CUDA private-owner semantics and no `get_offset` / `set_offset`.
2. [Factory owner validation](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/compat/torch/installers/factories.py#L761)
   maps backend 2 to `unsupported`, rejecting the explicit draw before allocation
   and submission.
3. [RandomOp](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/src/ops/composite/random_op.cc#L39)
   retains a supplied Generator state, but the
   [ACL launcher](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/backends/acl/src/acl_op_exec.cc#L714)
   obtains seed/offset from `reserve_acl_random(device, numel)`, without consuming
   `op.gen`.

Removing the compat refusal or renaming ACL to CUDA cannot provide isolation.
The default launcher's element-based advancement does not establish the counter
units of every CANN distribution.

## Minimum implementation boundary

| Layer | Capability requiring review |
| --- | --- |
| Native runtime | Explicit backend/device owner; isolated retained state; validated atomic restoration; immutable draw reservation |
| ACL random launch | Explicit owner supplies its own seed/counter; only `generator=None` uses default state; unsupported distributions fail before consumption |
| compat | Official backend-2 device resolution; owner/output-ordinal validation; seed/state and explicitly defined offset APIs |
| Downstream Adapter | Capability/version admission, request binding and actual consumption acceptance; no private C++ ABI, fake Generator or global-stream swapping |

Existing CPU/CUDA algorithms, state formats and placement checks must remain.
Native state tags and launcher changes require a matching Core: upgrading Python
alone cannot authorize reuse of a binary with unknown provenance or ABI.

### State and lazy execution

- Interleaving private owners, default draws and requests must not perturb other
  owners' streams.
- Reseeding affects only the selected owner. State frames identify version,
  algorithm, backend, device and position. Invalid frames, cross-device restore
  and overflow fail before mutation.
- A draw retains immutable seed/base/span and required lifetimes. Wrapper
  deletion, subsequent reseeding, completion ordering and rematerialization may
  not change an issued draw or consume its state twice.
- Validation failure consumes no state. Failures after publication need explicit
  semantics, without silently reusing an issued range.

### Offset is not an arbitrary Python counter

Define the algorithm, offset unit and consumption rule for each supported
distribution/dtype/shape first. vLLM's use of
`set_offset(get_offset() - 4)` is a caller assumption, not evidence of identical
ACL units or consumption. Ordinary and SIMT exponential advancement cannot be
interchanged; adding two methods does not establish correct discard behavior.

If the chosen algorithm differs, separately review request-level restoration
through public full-state snapshots, bound to request, owner and sampling step.
Reject stale snapshots and outstanding later draws; preserve CUDA's existing
branch. Graph capture, asynchronous/speculative sampling and cross-process
sharing are not implicitly covered by the first implementation.

## Staged acceptance

N0–N5 and A0–A1 below are proposed acceptance cases, not passed results.

- **N0**: real ACL owner, backend/ordinal validation, fake-handle and cross-device refusal.
- **N1**: same-seed replay; isolation under interleaved owners and default draws.
- **N2**: seed/reset/initial_seed and high-bit seed round trips.
- **N3**: algorithm reference vectors, boundary lengths, split draws, empty draws,
  overflow and exact consumed spans.
- **N4**: state/offset restoration, atomic invalid-state refusal, lazy-op lifetime,
  and unchanged issued draws after reseeding.
- **N5**: actual in-place `exponential_(generator=g)`, preserving object/storage,
  consuming the selected owner on ACL with zero fallback.
- **A0**: an original warmup/request path that really needs a private Generator,
  including natural shutdown.
- **A1**: original discard restoration, request isolation and stale-snapshot refusal.

CPU checks, import success, default RNG or one random operator execution do not
replace these cases. Cross-backend floating-point bitwise identity is not required.

## Manual minimal reproducer

The [reproducer](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/probes/acl/private_generator_repro.py)
compares small default/explicit draws and records whether offset methods exist.
The operator must arrange device admission and isolated caches first. This tool
does not authorize resource use or build a Core.

```bash
# Complete resource admission and use a Core matching the selected framework.
JT_BACKEND=acl JITTOR_NO_BUILD=1 python tools/probes/acl/private_generator_repro.py \
  --run --ordinal "$ACL_USER_ORDINAL"
```

The input API spelling is logical `cuda:<user ordinal>`; Jittor-owned Torch and
actual backend 2 must establish ACL ownership. It does not mean CUDA hardware.
The report retains interface provenance, original errors and fallback counters.
Nonzero exit preserves failure; catching an exception or passing the default
stream does not count as private Generator PASS.
