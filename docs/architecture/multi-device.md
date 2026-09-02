# Devices in One Process

Jittor used to know one CUDA device per process. `Var` had no notion of
where it lived, the allocator, the cuDNN/cuBLAS handles and the executor all
assumed "the" device, and setting `jt.flags.device_id` re-executed the
process with `CUDA_VISIBLE_DEVICES` rewritten. Under the Torch facade this
surfaced as `torch.cuda.set_device` doing nothing, `.to("cuda:1")` dropping
the index and every tensor reporting `cuda:0`. Real multi-GPU use was
limited to one device per process behind NCCL.

This document describes the device model that replaced it. The design keeps
the meta-operator graph and lazy execution untouched; it adds placement.

## Model

- **Every `Var` carries `device_id`**, the CUDA index it lives on or will be
  computed on. It is fixed when the Var is created: an op's outputs take the
  device of the op's inputs, a source op (`array`, `random`, `zeros`...)
  takes the *current device*. A Var migrated to host memory keeps its
  `device_id` and returns to that device.
- **The current device** is `jt.flags.device_id` / `jt.current_device()`,
  set with `jt.set_device(i)` or `jt.flags.device_id = i` and scoped with
  `jt.flag_scope(device_id=i)`. Setting it calls `cudaSetDevice` and lets
  each library wrapper swap in that device's handle; nothing restarts.
- **Ops run where their outputs live.** The executor switches the current
  device per op, allocates outputs from that device's pool and launches
  there. A final `device_sync` waits on every device the run touched and
  restores the caller's current device.
- **Mixed devices are an error at graph-construction time**, as in torch
  ("Expected all inputs to be on the same CUDA device"), with one
  exception that mirrors torch's CPU scalars: a pending source with no data
  yet (the `2` in `x * 2`) follows the operand it is combined with.
- **`Var.to_device(i)`** is the `device_copy` op: its output lives on `i`,
  the copy runs on the destination's stream after the producer's event on
  the source, and the source's stream waits for the copy before it may
  reuse the memory. Gradients flow back to the source device. Peer access is
  enabled once per device pair when the hardware allows it.

## Per-device state

| Resource | Before | Now |
| --- | --- | --- |
| device memory pool | one `cuda_device_allocator` | one pool per device (`get_allocator(device, temp)`); the global instance is device 0's |
| cuDNN / cuBLAS / cuSPARSE handles | one global | one per device; the global name always refers to the current device's handle, swapped by a device-switch hook |
| cuRAND generator | one | one per device, seeded together |
| cuFFT plans | one cache | one cache per device, swapped like the handles |
| synchronisation | `cudaDeviceSynchronize()` | every device the run launched on |
| NCCL | `cudaSetDevice(local_rank)` | goes through the same switch, so handles and pools agree |

`Allocator::device()` reports the device a memory block belongs to, forwarded
through the SFRL/stat/temp/NFEF wrappers, so host-to-device migration and
device-to-host fetches always run on the right device.

## Torch facade

`torch.cuda.device_count()`, `current_device()`, `set_device()`,
`with torch.cuda.device(i)`, `torch.cuda.device_of(t)`,
`with torch.device("cuda:1")` (default device for new tensors),
`Tensor.device` with the real index, `Tensor.get_device()`,
`.to("cuda:N")`/`.cuda(N)` (a copy when the index differs, identity when it
is the tensor's own device, a bare "cuda" keeps the tensor where it is) and
`device="cuda:N"` in factory functions (the tensor is created on N, not
copied there). `torch.accelerator.current_device_index/set_device_index`
follow.

## Out of scope for now

- Streams and events remain the per-device default stream; `torch.cuda.Stream`
  is still a no-op object.
- Kernels are compiled for the compute capabilities `query_cuda_cc` found;
  devices of a different architecture in one process are not handled.
- Memory swapping (`save_mem`) still assumes device 0.
