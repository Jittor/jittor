# Device Selection

One process can use every visible CUDA device. Placement follows torch's
rules, and the current device is switched at runtime -- the old
`jt.flags.device_id` setter that rewrote `CUDA_VISIBLE_DEVICES` and
re-executed the process is gone.

## Model

- **Every Var has a device.** `Var::cuda_device` (Python: `Var.device_index()`)
  is the CUDA index the Var is placed on. A new Var takes the current default,
  `jt.flags.device_id` (0 when unset). An op's outputs inherit its inputs'
  device.
- **Mixing devices in one op is an error**, raised when the op is built:
  `Expected all inputs of binary on the same CUDA device, got cuda:0 and
  cuda:1`. Scalar constants (one-element array ops) are exempt: they are host
  data and the executor places them wherever they are used.
- **`Var.to_device(i)`** returns the Var on device `i`, copying across devices
  with `cudaMemcpyPeer` (the `device_copy` op, differentiable: its gradient
  is a copy back). It is the only way data changes device.
- **`jt.flags.device_id = i`** is the current device: it calls
  `cudaSetDevice`, binds the per-device library state, and places Vars
  created afterwards on `i`. Other devices remain visible and usable.

## Runtime

The executor makes each op's device current before allocating its outputs
and running it, using that device's allocator chain (`get_allocator(temp,
device)` -- one SFRL cache per device over a per-device `CudaDeviceAllocator`,
whose `alloc`/`free` guard the current device themselves). `device_sync`
synchronizes every device the process has used.

Libraries keep per-device state through `register_device_switch_hook`: cuBLAS,
cuDNN, cuSPARSE and cuRAND bind `cublas_handle`/`cudnn_handle`/... to the
current device's handle on every switch (cuRAND generators share the seed);
the array and fetch ops keep a stream and event per device; cuFFT plans are
keyed by device. NCCL still selects one device per process and now records it
as the current device.

## Torch surface

`torch.cuda.set_device`, `current_device`, `device_count`, the
`torch.cuda.device(i)` and `device_of(tensor)` contexts, `device="cuda:N"` in
factories, `Tensor.device` with its real index, `Tensor.get_device()`,
`.to("cuda:N")`, `.cuda(N)` and `Module.to("cuda:N")` (in place, parameters
keep their identity) all map onto the mechanism above. `.to("cuda")` without an
index means the current device, as in torch.

Not covered: per-device streams and events as a user API
(`torch.cuda.Stream`), peer-to-peer access toggles, and memory statistics per
device.
