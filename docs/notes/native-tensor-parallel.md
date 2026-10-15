# Native tensor-parallel linear layers

Status: bounded native column/row linear implementation. Verification baseline
and device scope are recorded in the [maintainer result](../results/2026-10-15-native-tensor-parallel.md). Recheck after
collective adjoints, native optimizer synchronization or checkpoint changes.

`jittor.distributed.tensor_parallel.ColumnParallelLinear` stores only its
rank's output rows and local bias. `RowParallelLinear` stores only its input
columns; its bias is replicated and added after summing partial outputs.
Feature counts must divide the WORLD size. Native FP32 CPU singleton and
two-rank CUDA/NCCL, with the BF16 CUDA training protocol below, are the
verification scope; these are not general DTensor,
sequence parallelism, tensor-parallel attention or arbitrary distributed ops.

```python
from jittor import nn
from jittor.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class MLP(nn.Module):
    def __init__(self):
        self.up = ColumnParallelLinear(256, 1024, gather_output=False)
        self.down = RowParallelLinear(1024, 256, input_is_parallel=True)

    def execute(self, x):
        return self.down(self.up(x).tanh())
```

For CUDA, initialize native NCCL WORLD before constructing the layers. Native
launchers can load the NCCL backend with
`jt.compile_extern.get_library("nccl", load=True)` and call
`jittor.distributed.init_process_group("nccl", init_method="env://")`.
CPU singleton execution needs no process-group setup.

Every TP rank supplies the same input and replicated loss, and calls forward,
backward and checkpoint in the same order. This is model parallelism, not data
parallel averaging. Unrelated input batches across TP ranks violate this
contract. `gather_output=True` produces full replicated column output;
`input_is_parallel=False` scatters replicated row input. These options permit
standalone layers as well as a column/nonlinearity/row pair.

| Boundary | Forward | Backward |
| --- | --- | --- |
| Replicated column input | Identity | Sum local input-gradient contributions |
| Replicated row output | Sum partial outputs | Identity |
| Gather column output | Gather local feature slices | Slice matching local gradient |
| Scatter row input | Slice matching local features | Gather local input gradients |

These TP-specific adjoints use native Function and native collectives.
In particular, replicated row outputs do not apply a second all-reduce in
backward. Native optimizer synchronization recognizes TP parameters and skips
legacy data-parallel reductions and periodic broadcasts for those shards.
Optimizer arithmetic remains owned by existing native optimizers; the
maintainer training validation uses native momentum SGD and AdamW with weight
decay. DDP managing
the same TP parameters is explicitly unsupported; a separate data-parallel
mesh axis and hybrid process groups require their own implementation.

The native `checkpoint.save_training_state` / `load_training_state` path keeps
genuinely sharded model and optimizer leaves rank-local and checks rank,
world, orientation, local/global shape and FQN metadata before mutation.
Only the layout rank field is rank-local; the other equal-partition metadata
must agree across ranks, and incompatible global shapes fail publication. Fresh objects must have
the same partition and optimizer bindings. This preserves training state;
it does not reshard a checkpoint to another world size. Checkpoint payloads
contain actual local tensors, without gathering full weights. Replicated row
bias and its optimizer buffers retain the replicated-equality checks; a
drifted bias must fail checkpoint publication. Optimizer class, ordered FQNs,
group hyperparameters and step metadata remain replicated and validated.

`partition()` returns primitive ownership metadata, without exposing live
mutable storage metadata. Parameters must retain their partition shape and
WORLD device during use. Uneven partitions, empty feature ranks, subgroup TP,
DP+TP, ZeRO+TP or FSDP+TP composition, sharded-vocabulary loss, mixed
precision policies beyond the verified BF16 protocol, larger-world and
multi-host acceptance remain unverified/outside
this bounded implementation.
No whole-process memory peak or throughput improvement is claimed.

## BF16 CUDA training with FP32 state

The [AdamW/BF16 result](../results/2026-10-15-native-tensor-parallel-adamw-bf16.md)
verifies actual BF16 input and weight operands of the CUDA linear products,
FP32 master parameters, input/parameter gradients and both AdamW moments.
Native mixed precision uses the existing flags; TP adds no autocast context
or optimizer arithmetic:

```python
import jittor as jt

optimizer = jt.optim.AdamW(model.parameters(), lr=0.003, fused=False)
with jt.flag_scope(auto_mixed_precision_level=3):
    with jt.flag_scope(amp_reg=jt.amp_flags.prefer_bfloat16):
        prediction = model(inputs)
    loss = ((prediction.float32() - targets.float32()) ** 2).mean()
    optimizer.backward(loss)
    optimizer.step()
```

Keep level 3 active across forward, backward and optimizer updates: core uses
it to publish gradients in each original variable's dtype. A forward-only
BF16 preference does not guarantee FP32 parameter gradients. Select the BF16
preference only for forward; loss and optimizer arithmetic retain FP32.
Replicated row bias is explicitly converted to the product's compute dtype
before addition, using the same native helper as ordinary functional linear.
The row reduction carries BF16 partial outputs; the column adjoint sums FP32
input-gradient contributions.

The maintained small-MLP acceptance compares all five training steps against
an independent, unsharded real Torch model, then checks exact native
continuation after saving at step three and restoring fresh objects.
BF16 has a separately frozen, fixture-specific precision budget calibrated
from real Torch full-model versus the same two-rank TP decomposition.
The FP32 comparison keeps its strict tolerances. BF16 acceptance does not
promise bitwise equality across frameworks or a universal error bound.

CPU singleton tests cover the corresponding output, gradient/state and
checkpoint semantics. Native CPU AMP's generic contraction can retain wider
input operands; these tests do not certify actual BF16 CPU GEMM input casting.
CUDA GEMM operand dtype and storage are checked before any NumPy readback.
Other AMP policies, hybrid meshes and larger or multi-host WORLD remain
outside this validation.
