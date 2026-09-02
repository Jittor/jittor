# Device Placement in One Process

Jittor knew one CUDA device per process. `Var` had no notion of where it
lived; the allocator, the cuDNN/cuBLAS handles and the executor all assumed
"the" device; and setting `jt.flags.device_id` re-executed the process with
`CUDA_VISIBLE_DEVICES` rewritten. Under the Torch facade this surfaced as
`torch.cuda.set_device` doing nothing, `.to("cuda:1")` dropping the index and
every tensor reporting `cuda:0`. Real multi-GPU use was limited to one device
per process behind NCCL.

This document describes the device model that replaces it. It keeps the
meta-operator graph and lazy execution untouched; it adds placement. Two
implementations of this model existed on the branches `device-select` and
`multi-device`; §5 records what was taken from each and why, and what the
merged version does *not* prove.

## 1. Model

- **Every `Var` carries `Var::device_id`**, the accelerator index it lives on
  or will be computed on, fixed when the Var is created. An op's outputs take
  the device of its inputs (`Op::propagate_device`); a source op (`array`,
  `random`, `zeros`, …) takes the *current device*. A Var migrated to host
  memory keeps its device and returns to it.
- **The current device** is `jt.flags.device_id` / `jt.current_device()`, set
  with `jt.set_device(i)` and scoped with `jt.flag_scope(device_id=i)`.
  Setting it calls `cudaSetDevice` and lets each library wrapper swap in that
  device's handle. Nothing restarts, and other devices stay visible and
  usable.
- **Ops run where their outputs live.** The executor makes each op's device
  current before allocating its outputs and launching it. A final
  `device_sync` waits on every device the run touched and restores the
  caller's current device.
- **Mixing devices in one op is an error at graph-construction time**, as in
  torch: `Expected all inputs to be on the same CUDA device`. This runs from
  `Op::init`, so `jt.grad`'s new operators are checked by the same rule — a
  forward that is refused cannot be followed by a silently mixed backward.
  One exception mirrors torch's CPU scalars: a Var that is **both** unfinished
  **and** flagged `_is_scalar` (the `2` in `x * 2`, a gradient's starting `1`)
  follows the operand it meets, together with the small pending subgraph
  behind it. See §5 for why both halves of that test are needed and where its
  edge is.
- **`Var.to_device(i)`** is the `device_copy` op and the only way data changes
  device. The copy runs on the destination's stream after the producer's event
  on the source, and the source's stream waits for the copy before it may
  reuse the memory. It is differentiable: the gradient is a copy back. Peer
  access is enabled once per device pair where the hardware allows it.
  `device_copy` is the one op whose output device is not its inputs' — it says
  so with `NodeFlags::_manual_device`, and it is also the one op whose input
  must be migrated to the *input's* device rather than the op's.

## 2. Per-device state

| Resource | Before | After |
| --- | --- | --- |
| device memory pool | one `cuda_device_allocator` | one pool per device (`get_allocator(device, temp)`); the global instance is device 0's |
| cuDNN / cuBLAS / cuSPARSE handles | one global | one per device; the global name always means the current device's handle, swapped by a device-switch hook |
| cuRAND generator | one | one per device, seeded together |
| cuFFT plans | one cache | one cache per device, swapped like the handles |
| synchronisation | `cudaDeviceSynchronize()` | every device the run launched on |
| NCCL | `cudaSetDevice(local_rank)` | goes through the same switch, so handles and pools agree |

`Allocator::device()` reports which device a block belongs to, forwarded
through the SFRL, stat, temp and NFEF wrappers, so host-to-device migration
and device-to-host fetches always run on the right device.

## 3. Torch facade

`torch.cuda.device_count()`, `current_device()`, `set_device()`, the
`torch.cuda.device(i)` and `device_of(tensor)` contexts,
`with torch.device("cuda:1")` as the default device for new tensors,
`Tensor.device` with its real index, `Tensor.get_device()`, `.to("cuda:N")`
and `.cuda(N)` (a copy when the index differs, identity when it is the
tensor's own device), `device="cuda:N"` in factory functions (created on N,
not copied there), and `Module.to("cuda:N")` in place with parameter identity
preserved. A bare `.to("cuda")` means the current device, as in torch.

## 4. Out of scope

- Streams and events stay the per-device default stream; `torch.cuda.Stream`
  remains a no-op object.
- Kernels are compiled for the compute capabilities `query_cuda_cc` found;
  devices of different architectures in one process are not handled.
- Memory swapping (`save_mem`) still assumes device 0.
- Non-CUDA backends. Device placement and backend selection are different
  axes; see `multi-backend-design.md`, which proposes making the backend a
  value on the device rather than a build-time property.

## 5. Merged: what was taken from each branch, and why

Task 4.02 merged the two implementations. The model above is what landed; this
section records the four choices and the one place where the model has an edge
that neither branch had noticed.

| | taken from | why |
| --- | --- | --- |
| `Var::device_id` | `multi-device` | The word "device id" is already the vocabulary of `jt.flags.device_id`, `CUDA_VISIBLE_DEVICES` and `torch.cuda.current_device`. `device-select`'s `Var::cuda_device` / `Var.device_index()` names the same thing twice more. |
| scalar exemption = `!is_finished() && _is_scalar` | both, as a conjunction | Neither half alone is sound; see below. |
| copy ordering = destination stream + events both ways | `multi-device` | `device-select` used `cudaMemcpyPeer`, which gets the ordering by being *synchronous* — every move drains both pipelines. Events express the dependency without the drain. |
| facade surface | union of both | `device-select`'s `Module.cuda(i)`, `multi-device`'s `get/set_default_device` with an index and `torch.accelerator.*`. |

### Why the scalar rule is a conjunction

`device-select` exempted by element count, `multi-device` by pendingness. Each
is wrong on a case the other catches, and the repository has a test for each:

* **Element count alone** exempts a real one-element tensor that already holds
  the user's data (`tests/backends/cuda/test_multi_device.py::
  test_a_one_element_tensor_is_not_a_scalar`). `device-select` chose it because
  a flag bit was said to be unavailable; that is no longer true —
  `node.h`'s `_is_scalar` has been its own bit (26) since the mixed-precision
  fix.
* **Pendingness alone** retargets a `jt.array(np.ones(1000))` that the user
  deliberately built on `cuda:0` and merely has not synced yet
  (`::test_a_placed_pending_tensor_is_not_retargeted`) — silently, where torch
  raises.

`_is_scalar` is set by `array_op.cc` on a shape-`[1]` source and carried
through `broadcast_to_op.cc` and `unary_op.cc`, so `x * 2` passes and a real
array does not.

### The edge the conjunction does not remove

`jt.zeros(n)` / `jt.ones(n)` are `unary(0).broadcast(n)`: the `_is_scalar` flag
comes through the broadcast, so an unsynced `jt.zeros(1000)` built on `cuda:0`
*does* follow an operand on `cuda:1`, where torch would raise. This was
expected to be excluded by the conjunction and is not; measuring it is what
`::test_a_pending_broadcast_constant_does_follow` exists for.

It is accepted rather than patched. The value is a compile-time constant with
no data anywhere, produced bit-identically on either card, so nothing the user
computed is moved — this is constant placement, not data movement. Every path
that does carry data (a `jt.array` of more than one element, or any value that
has already been computed) is still refused. Narrowing it further would need a
fourth condition, and the obvious candidate — "the Var has no `VarHolder`" —
breaks the legitimate torch-compatible `two = jt.array(2.0); x_on_cuda1 * two`.

### What the merged version does not prove

The copy ordering is **not** exercised as a regression guard on this hardware.
All eight GPUs here report `cudaDeviceCanAccessPeer == 0` for every pair
(consumer cards, P2P disabled), so the driver stages every cross-device copy
through host memory and serialises it against the source device itself:
deleting the `cudaEventRecord`/`cudaStreamWaitEvent` pair from
`DeviceCopyOp::run` leaves the whole file passing. The test is written and
reports which regime it ran in (`_peer_regime`); it becomes a guard on a
peer-capable pair. See `agent/skills/multi-device-verification/SKILL.md`.

Streams and events beyond the per-device default stream stay out of scope
(task 4.08), as does `save_mem`, which still assumes device 0.
