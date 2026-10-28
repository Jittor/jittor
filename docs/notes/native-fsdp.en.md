# Native FSDP storage, backward and checkpoint ownership

Status: bounded CUDA/NCCL correctness implementation. FP32 training and fixed-layout
fresh-process continuation have maintainer evidence. BF16 computation also passes
the recorded tolerances and exact continuation, but its small-gradient difference
from binary Torch remains an explicit numerical limitation. See the
[validation record](../results/2026-10-15-native-fsdp.en.md) for the measured source,
thresholds and conditions. Owner: Jittor core maintainers; review when storage,
collectives, optimizer gradients, precision policy or checkpoint encoding changes.

## Native entry points

`jittor.distributed.fsdp.fully_shard` replaces a native module's parameters with
rank-local FP32 storage. `FullyShardedDataParallel` is a native `nn.Module` wrapper
around that operation; `jittor.nn.parallel` exposes the same class lazily. Neither
entry point imports Torch or the compatibility package. Build the native optimizer
**after** sharding and start every rank from the same complete parameter values.
Different initial values are rejected before the parameter collectives.

A CUDA launcher must provide two or more real ranks and initialize the native
NCCL WORLD process group. The ordinary native training call remains:

```python
import jittor as jt
from jittor import distributed as dist
from jittor.distributed.fsdp import FullyShardedDataParallel, MixedPrecisionPolicy

# Run under jittor.distributed.launch with an isolated cache per rank.
dist.init_process_group("nccl", init_method="env://")
jt.set_global_seed(42)
module = jt.nn.Sequential(jt.nn.Linear(5, 7), jt.nn.Tanh(), jt.nn.Linear(7, 3))
module = FullyShardedDataParallel(module)
optimizer = jt.optim.AdamW(module.parameters(), lr=0.001)
inputs = jt.ones((3, 5))
target = jt.zeros((3, 3))
optimizer.step(((module(inputs) - target) ** 2).mean())
```

For the bounded BF16 policy, pass
`mixed_precision=MixedPrecisionPolicy(param_dtype="bfloat16", reduce_dtype="float32",
output_dtype="float32")` to the wrapper. Parameters and moments remain FP32 local
shards. Temporary all-gather inputs, gathered parameters and floating inputs use
BF16; reduce-scatter and optimizer gradients use FP32. This is a module policy,
and does not enable global autocast or change native optimizer update formulas.

## One sharding implementation

`distributed/fsdp/storage.py` owns parameter partitioning, padding, temporary
all-gather parameters, module attribute replacement and full-reference release.
`gradients.py` owns full-gradient reduction into the corresponding local shard.
The compatibility FSDP implementation delegates to these functions through its
own metadata and collective adapter. Its DTensor, frontend scope, mesh and Torch
optimizer surface remain in `compat/fsdp2`; native execution does not reach back
into that frontend.

For each parameter, the native implementation partitions the flattened element
interval evenly, padding the last interval if necessary. The optimizer owns the
local parameter objects throughout training. Forward gathers temporary complete
parameters; backward differentiates those exact graph objects, reduce-scatters
the gradients, synchronizes the actual collective results, and releases the
complete parameter references. Native SGD/Adam/AdamW update the local FP32
parameters through their existing update algorithms. Independent trajectory
acceptance currently covers SGD and AdamW, rather than every accepted optimizer.

The optimizer seam validates rank, world size, shape, dtype, device, trainability,
parameter identity and group controls. FSDP parameters bypass the legacy gradient
all-reduce and periodic full-parameter broadcast. The existing DDP and tensor
parallel seams remain separate. Stale pre-sharding optimizers, partial parameter
ownership, conflicting wrappers and an additional backward without a fresh
forward fail explicitly.

## Checkpoint leaves and continuation

The existing `jittor.distributed.checkpoint.save_training_state` and
`load_training_state` APIs encode each genuinely sharded parameter, gradient and
optimizer moment as a `TensorShard` with its complete shape and valid flattened
interval. Padding is excluded on disk and reconstructed as zeros on restore.
Buffers, optimizer scalars and FSDP layout metadata remain replicated and checked
across ranks; the complete model is not hidden inside `RankLocal`. Existing
rank-local RNG handling remains owned by the native checkpoint implementation.

Restore validates the original FSDP shape, master dtype, world size, shard length
and precision policy before mutating the model. The acceptance sequence uses
three distinct launcher invocations: independent-oracle parity; three steps plus
save and two live steps; and a fresh-process restore followed by those last two
steps. The last phase requires exact equality of output, input/parameter
gradients, local parameters, optimizer states, real dropout output and each
rank's Python, NumPy, native CPU and CUDA RNG traces.

## Deliberate limits

Multi-rank acceptance currently covers CUDA/NCCL WORLD with fixed world size and
per-parameter partitions, FP32 masters and optional CUDA BF16 computation. Native
CPU/world-one tests exercise arithmetic and validation contracts, and do not
establish distributed CPU support. Nested FSDP, aliased parameters, parameter
containers, composition with DDP/TP/ZeRO, extra optimizer input parameters,
retained-graph backward and world-size migration are refused or unsupported.
Other accelerator backends, multiple hosts, overlap and every optimizer variant
have no acceptance claim here. Local storage counts establish partitioning; they
do not establish peak memory savings or performance.
