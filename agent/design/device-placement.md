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
implementations of this model exist on the branches `device-select` and
`multi-device`; §5 records how they differ and what remains to be decided
before one of them is merged.

## 1. Model

- **Every `Var` carries a device**, the accelerator index it lives on or will
  be computed on, fixed when the Var is created. An op's outputs take the
  device of its inputs; a source op (`array`, `random`, `zeros`, …) takes the
  *current device*. A Var migrated to host memory keeps its device and
  returns to it.
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
  torch: `Expected all inputs to be on the same CUDA device`. One exception
  mirrors torch's CPU scalars: a pending one-element source with no data yet
  (the `2` in `x * 2`, a gradient's starting `1`) follows the operand it meets.
- **`Var.to_device(i)`** is the `device_copy` op and the only way data changes
  device. The copy runs on the destination's stream after the producer's event
  on the source, and the source's stream waits for the copy before it may
  reuse the memory. It is differentiable: the gradient is a copy back. Peer
  access is enabled once per device pair where the hardware allows it.

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

## 5. Two implementations, one decision outstanding

Both branches implement §1 to §3 and pass their own multi-device, facade and
regression suites on two GPUs. They differ in detail:

| | `device-select` | `multi-device` |
| --- | --- | --- |
| Var field | `Var::cuda_device`, `Var.device_index()` | `Var::device_id` |
| scalar exemption | element count (a flag bit was unavailable) | pending source with no data yet |
| copy ordering | `cudaMemcpyPeer` | destination stream with events both ways |
| facade extras | `Module.cuda(i)` moves parameters | `get/set_default_device`, `torch.accelerator.*` |

Neither has run the full CUDA or native gates. Before either is merged:
choose one field name and one scalar rule, run both gates on the chosen
branch, and confirm the structure-test failures each branch reports are the
three that already fail on `2.0` (optim facade signature, runtime
composition) and not new ones.
