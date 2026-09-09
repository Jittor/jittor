# 设备放置：两个分支的合并论证（整改期）

任务 4.02 把 `device-select` 与 `multi-device` 两个实现合并。本文保留当时的取舍
理由、标量豁免规则为何必须是合取，以及该规则仍存在的边界。用户侧的设备模型说明见
[`docs/notes/device-placement.md`](../../docs/notes/device-placement.md)。

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
  the user's data (`tests/backends/cuda/test_cuda_multi_device.py::
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
