# Native ZeRO stage-one optimizer

`jittor.distributed.ZeroRedundancyOptimizer` shards **optimizer state during
training**. Each rank instantiates an existing native Adam, AdamW or SGD for
only its owned parameters. Non-owned moments are never constructed. Model
parameters and gradients remain replicated; this is ZeRO stage 1.

```python
import jittor as jt
from jittor import distributed as dist
from jittor.distributed import ZeroRedundancyOptimizer, checkpoint
from jittor.nn.parallel import DistributedDataParallel

dist.init_process_group("nccl", init_method="env://")
model = DistributedDataParallel(jt.nn.Linear(3, 2))
optimizer = ZeroRedundancyOptimizer(
    model.parameters(), jt.optim.Adam, lr=0.01, betas=(0.85, 0.95))

loss = ((model(jt.ones((4, 3))) - jt.zeros((4, 2))) ** 2).mean()
optimizer.step(loss)
checkpoint.save_training_state(model, optimizer, "checkpoint")

# Construct fresh owners with the same ordered groups and DDP trainability.
fresh = DistributedDataParallel(jt.nn.Linear(3, 2))
resumed = ZeroRedundancyOptimizer(
    fresh.parameters(), jt.optim.Adam, lr=0.01, betas=(0.85, 0.95))
checkpoint.load_training_state(fresh, resumed, "checkpoint")
```

Every rank executes these operations in the same order, and the checkpoint
directory must be shared by all ranks. The public native launcher supplies
the actual CUDA/NCCL workers. CPU and CUDA singleton use does not require a
distributed Store; the current multi-rank implementation explicitly requires
CUDA/NCCL WORLD and a native Store. Subgroups, HCCL/MPI multi-rank training and
world-size migration are outside this implementation's validated contract.

## Ownership and communication

Parameters are assigned whole, in descending element count, to the least
loaded rank, with stable group/parameter-order ties. Small models may give a
rank no parameters; that rank creates empty algorithm state lists and still
participates in gradient synchronization and owner broadcasts. Whole-parameter
assignment does not guarantee balanced per-rank state for one large parameter.

The wrapper's full `param_groups` belong to the native base Optimizer, which
owns backward, gradient accumulation, clipping and the existing DDP reducer.
`local_optimizer` exposes the existing native algorithm's owned state for
inspection. Updates call its `step()` with no loss: they perform no second
backward or gradient reduction. Group overrides and current defaults are
forwarded, including removal of an override. New parameter groups, changing
parameter bindings/shapes/dtypes and repeated parameter aliases in optimizer
groups are refused; pass the module's deduplicated parameters for tied weights.
Rank agreement expects JSON-serializable host hyperparameters.

Before asymmetric owner updates, all ranks finish the exact collective-gradient
roots with `weak_sync=False`. Parameters are then broadcast in parameter order
from their owners. Store agreement checks rank configuration and trainability;
peers finish their last RPC with the native `arrive` contract before the
server-hosting rank enters a collective. Rank zero cleans the temporary keys.
These explicit boundaries favor correctness; no throughput claim is made.

DDP's existing `no_sync` contract is preserved: intermediate microbatches can
accumulate locally, and a synchronized backward must precede the update.
Native DDP fixes trainability when constructed. Construct a resumed DDP model
with the saved trainability; standalone ZeRO supports late unfreezing while
retaining the existing algorithm's per-parameter first-touch rules.

## Checkpoints and precision

The native training checkpoint saves full model state, ordered FQN bindings,
trainability and RNG through their existing owners, and saves optimizer state
as `RankLocal` owned tensors. Partition metadata records the algorithm, rank,
world size, ownership and parameter shapes/dtypes. Loading validates this
metadata and local state before mutating model parameters. An absent saved
group override also removes any stale fresh-owner override. Pending gradient
accumulation remains explicitly unsupported for training checkpoints.

The initial verified scope is native FP32 Adam/AdamW/SGD on CPU singleton and
single-host two-rank CUDA/NCCL, plus native BF16 parameter/momentum SGD with
raw BF16 checkpoint preservation. This BF16 SGD check does not establish AMP
with FP32 master parameters or BF16 Adam/AdamW support. Active calculations
are observed on CUDA before NumPy I/O; untouched frozen zero state is checked
separately after explicitly materializing its declared storage placement.

This API does not supply parameter/gradient sharding, native FSDP, TP, ZeRO
stages 2/3, elastic resharding, or two-host rendezvous acceptance. The broader
distributed capability request remains partially implemented. See the
[verification report](../results/2026-10-15-native-zero-optimizer.en.md).
