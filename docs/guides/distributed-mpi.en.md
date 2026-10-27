# Distributed Training with MPI

Jittor can launch multiple data-parallel workers with MPI. Collective communication can use
NCCL on NVIDIA devices, and the training algorithm itself is still written in Python.

## Install and discover Open MPI

On Debian / Ubuntu, install the launcher and development headers:

```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

Jittor looks for `mpicc` on `PATH`. Specify it explicitly only when the compiler wrapper is
installed in a non-standard location:

```bash
export mpicc_path=/opt/openmpi/bin/mpicc
```

Launch the same training program with `mpirun`:

```bash
# 单进程
python train.py

# 四进程
mpirun -np 4 python train.py

# 指定两张 NVIDIA 卡
CUDA_VISIBLE_DEVICES=2,3 mpirun -np 2 python train.py
```

The dataset layer splits data across workers, supported optimizers synchronize gradients,
and synchronized BN layers exchange statistics. **Treat the batch size in the dataset
configuration as the global batch size**, and verify the behavior of custom samplers
yourself.

## Keep side effects on rank 0 only

Every worker executes the whole Python program. Restrict file writes and logging to rank 0:

```python
optimizer.step(loss)

if jt.rank == 0:
    writer.add_scalar("train/loss", float(loss.item()))
```

**Do not put Jittor operations inside a branch only some ranks enter.** All workers must
build compatible graphs and call collective communication in the same order, or they will
deadlock.

## Aggregate verification metrics

Reduce before using a value as a global metric:

```python
correct = local_correct(output, labels)
correct.sync()
if jt.in_mpi:
    correct = correct.mpi_all_reduce("add")

if jt.rank == 0:
    print("global correct:", int(correct.item()))
```

To run an operation in only one process without writing the branch by hand, use
`jt.single_process_scope()`:

```python
@jt.single_process_scope()
def write_summary(metrics):
    save_summary(metrics)
```

## Public MPI state and collectives

| Name | Meaning |
| --- | --- |
| `jt.in_mpi` | whether this process is in the MPI world |
| `jt.world_size` | number of workers; `1` when not under MPI |
| `jt.rank` | current worker index; `0` when not under MPI |
| `Module.mpi_param_broadcast(root=0)` | broadcast module parameters |
| `Var.mpi_reduce(op="add", root=0)` | reduce to one worker |
| `Var.mpi_broadcast(root=0)` | broadcast from one worker |
| `Var.mpi_all_reduce(op="add")` | reduce and send the result back to every worker |

Use `"mean"` when every worker contributes equally; otherwise reduce the **sum** and the
**sample count** separately.

## Synchronize statistics

Custom synchronized normalization can reduce moments across workers:

```python
x_mean = jt.mean(x, dims=[0, 2, 3], keepdims=1)
x2_mean = jt.mean(x * x, dims=[0, 2, 3], keepdims=1)
if jt.in_mpi:
    x_mean = x_mean.mpi_all_reduce("mean")
    x2_mean = x2_mean.mpi_all_reduce("mean")
x_var = x2_mean - x_mean * x_mean
```

Prefer a synchronized normalization layer maintained by Jittor when one fits; the example
above demonstrates the **call order for collective communication**, not a complete
replacement implementation.
