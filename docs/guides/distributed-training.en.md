# Native Single-Machine Multi-GPU Training

Jittor's native data parallelism uses `jtrun` to launch processes, `jittor.distributed` to
manage the process group, and `jittor.nn.parallel.DistributedDataParallel` to synchronize
model parameters and gradients. The first NVIDIA backend is NCCL; MPI remains available for
existing legacy programs. This guide does not depend on `jittor.compat.torch`.

The training example in this guide is single-machine multi-GPU data parallelism. The
verification scope for native FSDP, TP, ZeRO, and sharded checkpoints is documented
separately in [maintainer results](../results/index.en.md); a successful launch does not
imply that training sharding is accepted. Jittor's backward and update entry points are
`optimizer.backward(loss)` and `optimizer.step()`; the Torch-style `loss.backward()` is not
part of the native DDP contract.

## Launch

After installing Jittor, list the visible devices explicitly and launch two ranks:

```bash
CUDA_VISIBLE_DEVICES=0,1 jtrun --nproc-per-node=2 examples/distributed/native_ddp.py
```

Set `NCCL_P2P_DISABLE=1` under the verified single-machine conditions without P2P. `jtrun`
first probes peer access among local devices; when each node exposes only one card, the probe
cannot determine P2P capability across nodes or other devices, so transport must follow the
actual hardware configuration. This guide does not claim network or communication
performance based on that.

`python -m jittor.distributed.launch` is the module entry point for the same launcher. Each
process receives `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT`; NCCL child processes see only the GPU assigned to them. The launcher
terminates the remaining ranks when any rank fails, and returns a nonzero status.

### Fixed-node control plane (experimental)

The Linux/NCCL entry point accepts `--nnodes`, `--node-rank`, and `--rdzv-id`. Each node runs
one `jtrun` with the same master address reachable from all nodes, a nonzero port, the job
id, and the local process count:

```bash
# 节点 0；MASTER_ADDR 指向该节点可被其他节点访问的地址。
jtrun --nnodes=2 --node-rank=0 --rdzv-id=training-job \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --nproc-per-node=2 --device-ids="<local-device-list>" train.py
# 节点 1，地址、端口、job id 和本地进程数保持一致。
jtrun --nnodes=2 --node-rank=1 --rdzv-id=training-job \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --nproc-per-node=2 --device-ids="<local-device-list>" train.py
```

The global rank is `node_rank * nproc_per_node + local_process_index`; `WORLD_SIZE` is the
total process count, `LOCAL_WORLD_SIZE` is this node's process count, and
`GROUP_RANK/GROUP_WORLD_SIZE` identify the node. Under a single-card mask, `LOCAL_RANK` is
still device index 0.

Node 0's launcher hosts the native TCPStore service; workers only connect to that service,
avoiding rendezvous being blocked while GPU collectives hold the Python GIL. NCCL unique ids
for WORLD and subgroups are exchanged through the Store, so nodes do not need to share a
root-info file. All WORLD ranks must call `new_group` in the same order; inconsistent rank
order is rejected before NCCL initialization starts. Node agents propagate exit and heartbeats
through the Store; the NCCL heartbeat file monitors only local workers. When an agent is
terminated, its local workers finish via the Linux parent-death signal.

This is a fixed-node control plane and does not support elastic restart, adding/removing
nodes, or multi-node HCCL. The control Store should live on a trusted network; the job id
only provides namespace isolation. `--timeout` bounds rendezvous and heartbeat waits; first
JIT should be warmed serially in advance, with an independent cache per rank. Log directories
are local paths on each node; existing shared-file-path requirements for checkpoints are
still determined by the checkpoint protocol, and the control plane does not transfer
checkpoint files automatically.

Multiple agents on a single machine can only verify the control protocol; actual cross-host
connectivity, network failures, training, and fresh-process recovery still need real
multi-host acceptance. Do not treat loopback or two local ranks as multi-machine success.

## Training loop

The [complete example](../../examples/distributed/native_ddp.py) below uses the native
`Dataset` and a one-layer linear model. `Dataset.batch_size` denotes the global batch size
summed across all ranks; each rank generates a common shuffle order from the same seed and
epoch, then splits each complete global batch by rank. To keep the complete local batch the
same size on every rank, the global batch size should be divisible by the world size; tail
samples are sharded by the ceil rule when `drop_last=False`, and the smaller side repeats
tail indices. Call `set_epoch(epoch)` each epoch to advance the shuffle, and derive a
reproducible worker seed from the seed, epoch, rank, and worker id.

```python
import jittor as jt
from jittor import distributed as dist
from jittor.nn.parallel import DistributedDataParallel

dist.init_process_group(backend="nccl")
model = DistributedDataParallel(MyModel())
optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

for batch in loader:
    loss = model(batch).mean()
    optimizer.backward(loss)
    optimizer.step()
```

When DDP is constructed it broadcasts parameters and broadcast-enabled buffers from group
rank 0; before each synchronized backward, the reducer mean-all-reduces accumulated
gradients in stable parameter order. The NCCL backend puts one round of parameter gradients
into a single collective bucket, reducing per-parameter stream joins and host launches;
`with model.no_sync():` buffers local gradients, and after leaving it a synchronous
`optimizer.backward(loss)` is needed before `optimizer.step()`. This implementation uses only
one device per process and does not move the model for the user; put the model on the current
rank's device first.

File writes, logging, and ordinary checkpoints are performed only by rank 0:

```python
if dist.get_rank() == 0:
    model.module.save("checkpoint.pkl")
```

To preserve Adam/SGD momentum and step counts across restarts, also save and restore optimizer
state; the state file is likewise written only by rank 0 and read by each rank after a
barrier:

```python
if dist.get_rank() == 0:
    model.module.save("model.pkl")
    jt.save(optimizer.state_dict(), "optimizer.pkl")
dist.barrier()
model.module.load("model.pkl")
optimizer.load_state_dict(jt.load("optimizer.pkl"))
```

`destroy_process_group()` closes the registered Store and clears Python process-group state.
The current NCCL/HCCL wrapper has no callable communicator teardown, so the underlying
communicator is released when the worker process exits. `async_op=True` currently returns a
`Work` for an already-completed synchronous operation, so it does not represent overlap of
communication and computation; timeout/cancel are also not implemented. MPI's `jt.in_mpi`,
`jt.rank`, `jt.world_size`, and the `mpi_*` aliases continue to serve legacy programs and are
not the interface for new native training scripts.
