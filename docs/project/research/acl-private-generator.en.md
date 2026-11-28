# Private ACL Generator: isolated state and operator integration

[Research proposals](index.en.md) · [中文](acl-private-generator.md)

Status: **initial implementation, limited to the contract below**. Review is
tracked in [#18162](https://github.com/Jittor/jittor/issues/18162). Actual hardware
results, matching Core identity and remaining acceptance gaps belong in the
implementation PR. This is not acceptance of downstream seeded/discard workflows.

## Supported stream

`jt.Generator("acl:N")` is a native private owner on ACL user ordinal `N`.
With the Jittor Torch frontend active, `torch.Generator("npu:N")` and the native
`acl:N` spelling select that owner. Logical `cuda:N` resolves through the existing
frontend backend resolver; it identifies an ACL owner only in an ACL runtime.
Owner and output backend/ordinal must match. A matching rebuilt Core is required;
a Python-only upgrade cannot add this native interface to an old binary.

The first native distribution is dense **float32 uniform**, generated on the NPU
by an AscendC Philox4x32-10 kernel. It uses neither host random values nor the
process-default CANN RNG. The existing compat inverse-CDF implementation consumes
this uniform through the supplied owner for `exponential_(generator=g)`.
Unsupported native distributions/dtypes fail before reservation; this is not a
claim that all Torch random factories or all exponential output dtypes work.
`generator=None` retains the existing CANN implementation and its state format.
CPU/CUDA algorithms and state frames remain unchanged.

## Exact seed, position and consumption contract

- The seed is a uint64 bit pattern, carried by the native signed-int64 API.
- A position is one **raw 32-bit word**, in `[0, 2**63 - 1]`, rather than a draw,
  a CANN/SIMT offset or a CUDA launch offset.
- For position `p`, the counter is `(low32(p//4), high32(p//4), 0, 0)`, the key
  is `(low32(seed), high32(seed))`, and the selected lane is `p % 4`.
- A float32 uniform is `(word >> 8) * 2**-24`, exactly in `[0, 1)`.
  Each output element reserves one word. Empty draws reserve zero; shape splitting
  does not change the concatenated sequence. This mapping is not promised to
  match PyTorch, torch-npu or CANN distribution output bit for bit.
- `manual_seed` resets the position to zero. `get_offset`/`set_offset` expose this
  raw32 position on ACL only. Invalid, negative and overflowing positions fail
  without mutation. CPU/CUDA offset methods remain explicitly unsupported here.

The algorithm is described by the original
[Random123 paper](https://thesalmons.org/john/random123/papers/random123sc11.pdf).
The tests include the published zero-key/zero-counter
[Philox known-answer vector](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors)
and an independent integer reference for nonzero seeds and high positions.

## State and lazy execution

The native owner holds a mutex-protected seed and position. State frames encode
version `JITTOR_ACL_RNG_1`, algorithm `philox4x32-10-raw32-v1`, backend, user ordinal,
seed and position. Restore validates all fields, including trailing bytes, before
changing either field. Foreign algorithm/backend/device frames are refused.
Compat exposes the frame as a CPU ByteTensor rather than interpreting its fields.

A RandomOp atomically reserves and captures immutable seed/base/span when it is
constructed. Later reseeding/restoration, reversed execution, Python-wrapper
deletion or rematerialization cannot change an issued draw or consume its span a
second time. Independent owners and default draws do not advance one another.
Validation failures consume no state. Once a draw is published, a compile/launch
failure does not silently rewind or reuse its reserved span; the operation fails.
Private draws cannot fall back to CPU or to the default device RNG.

The initial kernel uses one logical block and UB-to-GM copies with explicit
scalar/DMA synchronization, including partial tiles. It does not infer a physical
core count. It requires the installed CANN AscendC compiler and the actual SoC's
platform configuration. This is a correctness implementation with no throughput
or asynchronous/graph-capture support claim.

## Acceptance boundary

The native tests in `tests/backends/acl/test_acl_private_generator.py` cover
algorithm vectors, lane/tile boundaries, high positions, reservations, owner
lifetime and atomic refusal. The compat tests in
`compat/tests/torch/test_acl_private_generator_compat.py` cover real owner routing,
isolation, state/offset replay, in-place exponential consumption and invalid
inputs. Both require a real admitted ACL device and zero fallback. CPU checks,
import success and default RNG checks cannot replace those device tests.

The proposed N0–N5 categories concern these native/compat semantics. **A0**
(original warmup/request use and shutdown) and **A1** (original discard restoration,
request binding and stale-snapshot refusal) remain downstream acceptance tasks.
In particular, `set_offset(get_offset() - 4)` rewinds exactly four raw32 words;
it does not undo a draw containing an arbitrary number of vocabulary elements.
Adding methods must not be treated as validation of that vLLM assumption.
Graph capture, speculative sampling and cross-process state sharing are outside
this initial contract. Downstream adapters must use reviewed public state APIs
and preserve their existing CUDA branches.

## Manual reproducer

The [reproducer](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/probes/acl/private_generator_repro.py)
records small default/private draws, interface provenance, original failures and
fallback counts. Arrange resource admission and an isolated, matching Core first;
the tool neither grants device permission nor builds a Core.

```bash
JT_BACKEND=acl JITTOR_NO_BUILD=1 python tools/probes/acl/private_generator_repro.py \
  --run --ordinal "$ACL_USER_ORDINAL"
```

Its logical `cuda:<user ordinal>` input does not identify CUDA hardware: actual
backend 2, Tensor placement and SDK mapping establish the physical ACL device.
A caught exception, a passing default stream or exit zero alone is not private
Generator acceptance.
