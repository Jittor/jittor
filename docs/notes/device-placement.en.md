# Devices and placement

This page clarifies three things: which accelerator a tensor is actually on, how to
move it to another one, and when Jittor will refuse you.

## A tensor's device is its own attribute

Each `Var` carries a `device_id`, denoting the accelerator it is on (or will be
computed on), fixed at creation:

- an operator's outputs **inherit the input's device**;
- source operators (`array`, `random`, `zeros`, etc.) land on the **current
  device**;
- a Var moved to host memory still remembers its device, and returns to the same
  one when it goes back.

The **current device** is `jt.current_device()`, set with `jt.set_device(i)` or
scoped with `jt.flag_scope(device_id=i)`. Setting it calls `cudaSetDevice` and
switches each library handle to that device. **The process does not restart**;
other devices remain visible and usable.

`jt.flags.device_id` is the raw state behind it, and the two are not the same
reading: at process start `jt.flags.device_id == -1`, meaning "nobody has chosen
yet", while `jt.current_device()` resolves that sentinel to `0` when an
accelerator is visible. Once `jt.set_device(i)` has been called, both are `i`.
Always read `jt.current_device()` to decide "which device am I on";
`jt.flags.device_id` matters only when distinguishing "unset" from "explicitly
set to 0" (`tests/runtime/test_runtime_device_state.py` writes it directly).

In a **build without accelerators** (`jt.get_device_count() == 0`, a CPU-only
session), `jt.current_device()` returns **-1**, not 0: the contract in
`src/runtime/device.h` is "the current device, or -1 when no accelerator is
visible", and answering 0 would claim a device `get_device_count()` says does not
exist. Any code that spans both builds must be written this way, and must pass
under both sides of `tools/run_test_suite.py --backend cpu|cuda`.

`Var.device` gives the `"cpu"` / `"cuda:N"` pair, while `Var.device_id` gives only
the number. A Var that has not executed yet has no allocation to ask about, so
`device` reports where it *will* land; the only case needing special handling is
a pending `.cpu()` — its destination is decided, but `device_id` deliberately
keeps the source device (so that it can return), so `x.cuda(3).cpu().device`
reads `"cpu"` before the sync, not `"cuda:3"`.

An operator executes on **the device where its output lives**: the executor first
makes that device current, then allocates outputs and launches kernels. At the end
of a run it waits for every device it touched and restores the caller's original
current device.

## There is only one way to change devices: `device_copy`

`Var.to_device(i)` is the `device_copy` operator, and is the only route by which
data changes devices. It:

- executes **on the target device's stream**, waits for the producer's event on
  the source device; the source device's stream must also wait for this copy
  before reusing that memory;
- is **differentiable**, with the gradient being a copy in the reverse direction;
- on device pairs where the hardware allows it, enables peer access once per pair.

User-facing spellings:

```python
b = a.cuda(3)            # 复制到 3 号卡；已在 3 号卡则返回自身
b = a.cuda("cuda:3")     # 同上，也接受带 .type/.index 的设备对象
b = a.to("cuda:3")       # 同上
b = a.to(device="cuda:3")
b = a.to("cuda:3", "float16")   # 换设备顺带换 dtype，顺序无关
b = a.to(other)          # 取 other 的 dtype 和设备，other 未同步也算数
c = b.cpu()              # 取回主机内存，返回新 Var，源不变
```

`.cpu()` allocates its destination buffer **on the host side** and does not occupy
an equal-sized device allocation.

Rejected spellings, each of which **raises** instead of quietly doing nothing:

```python
a.to(1)            # TypeError：裸整数不是设备，torch 在这里同样报错
a.to("cuda1")      # TypeError：拼错的设备名不会被当成 dtype
a.cuda(-1)         # RuntimeError：设备编号不能为负
a.cuda(99)         # RuntimeError：超出可见设备数
```

(The Torch compatibility layer is slightly more permissive: `Tensor.to(1)` /
`Module.to(1)` read as `cuda:1` there, because bare indices are common in
multi-GPU inference stacks, see section 31 of
`docs/results/2026-09-14-vllm-omni-h3-enablement.md`. Permissiveness is fine;
silent loss is not.)

**An unnumbered `cuda` means different things in the two layers**, deliberately:

| Spelling | Native `jt.Var` | Torch compatibility layer (`torch.Tensor`) |
| --- | --- | --- |
| `x.cuda()` / `x.to("cuda")` | its **own** device (`device_id`), i.e. "make sure it is on the device"; a host-parked Var thus returns to its original card | the **current device**, same as torch |
| `x.cuda(1)` / `x.to("cuda:1")` | device 1 | device 1 |

The native behavior is not an oversight: `.cuda()` reads as "put it back on the
accelerator it belongs to", and a Var always remembers which one that is. Within
one process the two kinds of Var follow their own rules — the one where
`type(x) is jt.Var` follows the left column, `torch.Tensor` follows the right.

## Mixed devices is a graph-construction-time error

Inputs of one operator landing on different devices **raises at graph
construction**, consistent with torch:

```
Expected all inputs to be on the same CUDA device
```

The check lives in `Op::init`, so backward operators created by `jt.grad` follow
the same rule — **a combination rejected in forward never produces a backward
that quietly mixes devices**.

The sole exception mirrors torch's CPU scalars: a Var that is **both not yet
computed and marked `_is_scalar` or `_is_constant`** (the `2` in `x * 2`, the `1`
that seeds a gradient) follows the operand it meets, together with the small
pending subgraph behind it. `_is_scalar` is the bit meaning "participate in dtype
promotion as a scalar", `_is_constant` is the bit used only for placement: factory
results clear the former and keep the latter at full shape, so `jt.ones(4)`
promotes as a tensor while still being able to follow an operand when unsynced.
Both conditions are required — looking only at element count would let through a
genuine one-element tensor the user actually stored on some device; looking only
at "not computed" would quietly move a `jt.array` the user deliberately built on
`cuda:0` and merely hasn't synced, where torch raises.

Known boundary: `jt.zeros(n)` / `jt.ones(n)` are implemented as
`unary(0).broadcast(n)`. The factory result clears `_is_scalar` on the broadcast,
but `_is_constant` passes through broadcast, so an unsynced `jt.zeros(1000)`
**still** follows an operand on another device, while torch raises. This is
**accepted** rather than patched: it is a compile-time constant, produced
bit-identically on any device, with no user-computed data being moved. Every path
that actually carries data (a `jt.array` with more than one element, or any value
already computed) is still rejected.

This exemption **does not survive an explicit `device_copy`**. The output of
`device_copy` carries `_placement_published`: the caller has named its device, so
it is no longer a pending scalar that may tag along. Before this,
`jt.ones(3).cuda(2)`'s output inherited `_is_scalar`, so

```python
a = jt.ones(3).cuda(1); b = jt.ones(3).cuda(2)   # 两个都没同步
(a + b).sync()      # 整段跑在 cuda:1 上，而且 b.device 读作 "cuda:1"
```

— an explicit `.cuda(2)` was silently rewritten, while the same expression raises
if synced first; `.cpu()` was lost the same way. `_is_scalar` itself is kept on
the copy output, because dtype promotion needs it (`x.cuda(1) * 2` must promote
the same as `x * 2`).

## Per-device state

| Resource | Granularity |
| --- | --- |
| Memory pool | one per device; the global instance is device 0's. Query usage per device: `jt.core.device_memory_used(i)` (bytes in use) and `jt.core.device_memory_reserved(i)` (bytes held by the pool, including cached free blocks); `i == -1` is the host side. `MemInfo.total_cuda_used` is the **sum over all devices** and cannot answer "how much is on device i" |
| Optimizer state | follows the device of its parameter: allocated as `zeros_like(param)` at construction, aligned once more before `step()`, so an existing optimizer keeps working after `Module.to("cuda:1")` (torch achieves the same by building state only on the first step) |
| cuDNN / cuBLAS / cuSPARSE handles | one per device; the global name always refers to the current device's handle |
| cuRAND generator | one per device, unified seed |
| cuFFT plan cache | one cache per device |
| Synchronization | waits for every device this run launched work on |
| NCCL | uses the same device switching; handles and memory pool stay consistent |

## Torch compatibility spellings

All of the following work with torch semantics: `torch.cuda.device_count()`,
`current_device()`, `set_device()` (non-CUDA devices raise `ValueError` as in
torch), the `torch.cuda.device(i)` and `device_of(tensor)` contexts,
`with torch.device("cuda:1")` as the default device for new tensors,
`Tensor.device` with real indices, `Tensor.get_device()`, `.to("cuda:N")` and
`.cuda(N)` (copy if different, return itself if already there), factory functions'
`device="cuda:N"` (**created directly on N**, not created then moved), `x.new_*`
and `*_like` inheriting the reference tensor's device, and `Module.to("cuda:N")`
taking effect in place while preserving parameter object identity. Unnumbered
`.to("cuda")` means the current device, as in torch.

`torch.device` itself parses as in torch: `torch.device(1)` is `cuda:1` (**not
cpu**), negative indices and nonexistent device types raise, and
`torch.device("cuda")` and `torch.device("cuda:0")` are two different objects.
`torch.cuda`'s per-device queries — `get_device_name(i)`,
`get_device_capability(i)`, `get_device_properties(i)`, `mem_get_info(i)`, the
`memory_allocated(i)` family, `current_stream(i)`/`default_stream(i)`,
`can_device_access_peer(a, b)` — answer for the index passed, and without an
argument refer to the **current device**, not device 0.
`with torch.cuda.stream(s)` is also a device context: within the block the current
device is `s.device`.

Two deliberate differences from torch, documented here rather than left to be
discovered:

- `with torch.device("cuda:3"):` also switches jittor's current device to 3
  (torch only changes the default device for new tensors;
  `torch.cuda.current_device()` is unchanged). This way native allocations made
  inside the block that bypass the frontend also land on 3, instead of scattering
  to device 0.
- `Tensor.pin_memory()` on a tensor already on an accelerator returns a **host**
  copy instead of raising (torch raises). Here a tensor without `device=` is
  already on the accelerator, so raising would break staging code that works fine
  on torch. The buffer is not page-locked, so `non_blocking` H2D stays
  synchronous; `is_pinned()` returns True consistently with that.

## Backend placement of tensors

The above is about "which device"; the standalone Torch frontend must also answer
"on which backend" — an explicitly `cpu` tensor must stay on CPU even when
`use_cuda=1` globally. This is a **native graph constraint**, not a Python-side
marker: the Var/Op graph and meta-operators are still shared with native Jittor;
there is no separate Torch executor.

- **Two placement states.** `Var::placement` is either `FollowRuntime` (the
  default for native construction, deferring to native Jittor's existing backend
  policy) or explicit placement (a `BackendId` plus a device index). Explicit
  placement survives host staging, materialization and changes to the Runtime
  default backend. `Var::device_id` still denotes the native accelerator
  affinity, and is `-1` for a Var explicitly placed on CPU. Physical residency
  follows **allocation**.
- **Graph-construction scopes.** `TensorPlacementScope` only changes thread-local
  graph-construction metadata: it does not call the Runtime's flag setters, does
  not synchronize, does not allocate and does not execute kernels. The Python
  binding's placement ContextVar carries the explicit factory device, and the
  frontend binding scope converts it into native construction state on each C++
  call and restores it on exit. Without a factory override, the placement of
  frontend inputs takes precedence over the Runtime default.
- **Propagation and conflicts.** `Op::propagate_device` makes outputs adopt the
  inputs' explicit placement; an explicit backend or device conflict requires an
  explicit copy. The Python surface marks native Vars with
  `_placement_published`, and a published tensor's placement cannot be changed by
  scalar redirection. Binary/ternary construction allows a 0-D tensor on CPU to
  be used with an accelerator operand: the generated construction adapter inserts
  a differentiable DeviceCopy before establishing members, the broadcast subgraph
  or node edges, leaving the source on CPU and copying the gradient back to CPU;
  the decision uses the actual 0-D shape, and the pre-existing `_is_scalar`
  promotion bit is retained rather than invented. An explicit accelerator scalar
  paired with a CPU vector is rejected.
- **Selection and execution.** Native dispatch queries and construction-time
  capability choices consult placement first, then the Runtime default (Random,
  Array, Where, Transpose, Argsort, ArgReduce and the reduction dtype policies all
  do). Fusion **does not merge** operators that requested different backends;
  each segment's compilation, graph optimization and runner scope use its graph
  placement, and the runner picks the backend/device allocator per segment.
  Explicit CPU outputs are never uploaded by the old CPU-fallback tail; genuine
  accelerator fallback keeps its existing policy. DeviceCopy handles the transfer
  itself, does not migrate sources inside the runner, and its gradient copies
  back to the input's explicit placement.
- **Frontend entry.** `tensor_frontend(type, device=..., like=...)` constrains
  construction without changing the Runtime default. Factory functions, the
  `Tensor`/`new_*` constructors, clone/deepcopy/pickle, `to`/`cpu`/`cuda`, and
  serialization's `map_location` all go through this boundary; device reporting
  and data-owner routing read the native placement.
- **Module forward.** The placement reference is the first tensor argument with
  explicit placement; when there is none, it uses its own explicitly placed
  tensors (parameters/buffers) rather than the Runtime default
  (`nn_frontend._placement_reference`; the native fast path uses the same logic
  in `first_placed_module_var`). For a module default-placed on CPU paired with
  unplaced accelerator inputs, forward therefore stays on the CPU where the
  parameters are; otherwise intermediates would land on the accelerator while
  parameters stay on the host, and the portable convolution path (taken under
  autocast) rejects mixed placement at its first binary operator. Explicit
  placement conflicts still raise per the rules above; this only decides the
  default when no input has placement.

`BackendId` gives ACL, ROCm and Corex builds the same CPU/accelerator split, but
at runtime each build/process still selects a single accelerator, **provides no
cross-accelerator-family copy**, and makes no claims about verification on
missing hardware. Targeted cases are in
`compat/tests/torch/test_native_tensor_placement.py`,
`compat/tests/torch/test_kernel_select_frontend_placement.py` and
`tests/structure/runtime/test_tensor_placement_scope.py`; host-side scope
nesting/thread isolation and translation-unit syntax checks are only
preconditions and do not prove tensors execute on the requested hardware.

## Out of scope

- **Streams and events**: only a per-device default stream. `torch.cuda.Stream`
  has device identity and `with torch.cuda.stream(s)` switches devices, but all
  logical streams serialize onto the same physical stream, so
  `wait_stream`/`wait_event` are honest no-ops rather than missing implementations.
- **Heterogeneous architectures**: kernels are compiled for the compute
  capability detected by `query_cuda_cc`; mixing devices of different
  architectures within one process is not handled.
- **Memory offloading** (`save_mem`) still assumes device 0.
- **Non-CUDA backends**: device placement and backend selection are two separate
  dimensions; the backend side is covered by "Backend placement of tensors" above,
  and how a backend is selected at build time is covered in
  [source architecture](../development/source-architecture.en.md) under "backend
  build configuration".
