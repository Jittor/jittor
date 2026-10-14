# Native sharded checkpoint files

Status: experimental native I/O and completed-step training continuation.
Reviewed: 2026-10-15. Baseline: `8e388d108eb83c1c438936709f29855f47ba2a69`. Owner:
serialization and distributed maintainers. Revalidate when the native Module,
Optimizer, RNG or process-group protocols change.

`jittor.distributed.checkpoint` uses native Jittor owners and never imports Torch
or its compatibility frontend. It partitions checkpoint payload files; it does
not shard training parameters, gradients or optimizer arithmetic. Native FSDP,
tensor parallelism, ZeRO and multi-node acceptance are separate capabilities.

## Training continuation

```python
from jittor import distributed as dist
from jittor.distributed import checkpoint

# The launcher owns the real communicator. Register a Store for multi-rank I/O.
dist.init_process_group("nccl", init_method="env://")
# Native model/optimizer train normally; every rank calls the checkpoint API.
optimizer.step(loss)
checkpoint.save_training_state(model, optimizer, checkpoint_directory,
                               extra={"completed_steps": 3})
extra = checkpoint.load_training_state(model, optimizer, checkpoint_directory)
```

The helper saves model FQNs and trainability flags, optimizer class, ordered
parameter bindings, defaults, moment buffers and counters, and rank-local
Python, NumPy, native CPU and visible CUDA generator states. Saving leaves the
original model flags unchanged. Loading applies saved flags through the native
Module owner and installs optimizer state through its existing public owner.
Native optimizer mathematics are not replaced. In particular, native Adam's
epsilon placement differs from `torch.optim.Adam`; numerical verification must
use the native convention.

Save after a completed optimizer step or `zero_grad()`. Pending accumulation is
explicitly refused because the native public optimizer state does not contain
its backward latch. Restore requires matching model FQNs, shapes and dtypes,
optimizer class and ordered FQN/group bindings, the same execution backend and
visible CUDA RNG device count, and the same global rank set.
Model/optimizer metadata and all payload files are validated before target
owners are mutated. Invalid generator state rolls back the current RNG state.
Custom optimizer load hooks and concurrent mutation of target owners are not a
transactional extension point.

CPU and CUDA native tensor payloads are supported. BF16 uses an explicit UInt16
storage view and restores the original BF16 dtype and raw bits; `numpy()` alone
exports BF16 numerically as FP32 and would lose original storage identity.
Unvalidated native backends are explicitly refused.

## Generic state trees and local tensor intervals

`save(state, directory)` accepts string-key dictionaries, lists, tuples, primitive
JSON values, NumPy scalars/arrays, native tensors and the two descriptors below.
No pickle globals are executed. Native tensor leaves load detached on the current
native device; training helpers separately restore recorded trainability.

```python
from jittor.distributed.checkpoint import RankLocal, TensorShard, save, load

# offset is a flat ELEMENT interval in the given global shape.
save({"weight": TensorShard(local_weight, global_shape, element_offset),
      "extra": RankLocal(per_rank_state)}, directory)
local = load(directory, tensor_shards={
    "/weight": TensorShard(target_tensor, global_shape, new_offset)
})["weight"].tensor
```

The target controls the returned local shape and native placement. Its original
contents are not modified. Dtype and native/NumPy owner must match the saved
leaf. Tensor paths use JSON-pointer escaping (`~0`, `~1`). Target intervals may
change, but rank-local state cannot migrate to an absent global rank. Training
RNG resharding and world-size changes are refused.

Plain tensor inputs are replicated: every rank snapshots the full input on the
host, verifies its full-byte digest agrees, and writes one balanced flat interval.
This path reduces file duplication but does not promise bounded host memory.
`TensorShard` instead consumes only an existing local input, validates global
coverage, and can restore a requested local interval without allocating a full
global output. Neither descriptor supplies training sharding.

## Completion and failure

All ranks must call `save` in the same order with the same path and overwrite
option, using a shared filesystem and a native process-group Store. The Store
exchanges status and metadata; each rank writes its own payload files. Payloads
and per-rank metadata are checked for missing ranks, mismatched replicas,
shapes/dtypes, overlapping or missing intervals and SHA256 corruption. Only
then does rank zero atomically replace `manifest.json` with a complete generation.

Failures notify participating peers and raise rather than produce a complete
marker. `overwrite=True` preserves the previous committed generation if a new
write fails. Old and incomplete generations remain for explicit operator cleanup.
Timeouts do not cancel I/O or establish resilience to worker death. Concurrent
saves to the same directory are unsupported. Generation files must remain
immutable while readers validate and restore them; filesystem crash guarantees
remain those of the deployed shared filesystem.
