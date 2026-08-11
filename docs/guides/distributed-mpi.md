# Distributed training with MPI

Jittor can use MPI to launch data-parallel workers. Supported collective
operations may use NCCL on NVIDIA devices, while the training algorithm remains
in Python.

## Install and detect Open MPI

On Debian or Ubuntu, install the launcher and development headers:

```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

Jittor discovers `mpicc` from `PATH`. Set `mpicc_path` only when the compiler
wrapper is installed somewhere non-standard:

```bash
export mpicc_path=/opt/openmpi/bin/mpicc
```

Start the same training program under `mpirun`:

```bash
# One process
python train.py

# Four processes
mpirun -np 4 python train.py

# Two selected NVIDIA devices
CUDA_VISIBLE_DEVICES=2,3 mpirun -np 2 python train.py
```

The dataset layer partitions data across workers, supported optimizers
synchronize gradients, and synchronized batch-normalization layers exchange
statistics. Treat a dataset's configured batch size as the global batch size
and verify the behavior of custom samplers.

## Side effects belong on rank zero

Every worker executes the Python program. Restrict filesystem and logging side
effects to rank zero:

```python
output = model(images)
loss = nn.cross_entropy_loss(output, labels)
optimizer.step(loss)

if jt.rank == 0:
    writer.add_scalar("train/loss", float(loss.item()))
```

Do not place Jittor operators in a rank-only branch. Workers must construct
compatible graphs and call collectives in the same order or they can deadlock.

## Aggregate validation metrics

Reduce values before using them as global metrics:

```python
correct = local_correct(output, labels)
correct.sync()
if jt.in_mpi:
    correct = correct.mpi_all_reduce("add")

if jt.rank == 0:
    print("global correct:", int(correct.item()))
```

For an operation that should run in one process without manually branching,
use `jt.single_process_scope()`:

```python
@jt.single_process_scope()
def write_summary(metrics):
    save_summary(metrics)
```

## Public MPI state and collectives

- `jt.in_mpi` reports whether the process participates in an MPI world.
- `jt.world_size` is the number of workers, or `1` outside MPI.
- `jt.rank` is the current worker index, or `0` outside MPI.
- `Module.mpi_param_broadcast(root=0)` broadcasts module parameters.
- `Var.mpi_reduce(op="add", root=0)` reduces a value to one worker.
- `Var.mpi_broadcast(root=0)` broadcasts a value from one worker.
- `Var.mpi_all_reduce(op="add")` reduces and returns the result to every worker.

Use `"mean"` when each worker contributes equally; otherwise reduce a sum and a
sample count separately.

## Synchronized statistics

Custom synchronized normalization can reduce moments across workers:

```python
x_mean = jt.mean(x, dims=[0, 2, 3], keepdims=1)
x2_mean = jt.mean(x * x, dims=[0, 2, 3], keepdims=1)
if jt.in_mpi:
    x_mean = x_mean.mpi_all_reduce("mean")
    x2_mean = x2_mean.mpi_all_reduce("mean")
x_var = x2_mean - x_mean * x_mean
```

Prefer Jittor's maintained synchronized normalization layer when it covers the
use case; the example illustrates collective ordering rather than a complete
replacement implementation.
