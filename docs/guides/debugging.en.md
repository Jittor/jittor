# Debugging Jittor Programs

Jittor executes lazily and compiles operators at runtime. The diagnostics below all
**trade performance for more accurate failure information**, so enable them only while
reproducing an issue.

## Locate NaN and inf

Enable output checks and Python source tracing before starting the process:

```bash
export JT_CHECK_NAN=1
export JT_TRACE_PY_VAR=3
python reproduce.py
```

`JT_CHECK_NAN=1` stops when an operator produces an invalid floating-point value;
`JT_TRACE_PY_VAR=3` records the Python call site that produced the operator. **Both trigger
recompilation and slow execution significantly.**

## Turn asynchronous failures into local ones

Lazy execution reports an error only after the Python line that created the failing
operator -- which is why the stack often points at `print` or `.numpy()`. Temporarily switch
to immediate execution when building a minimal reproduction:

```bash
export lazy_execution=0
```

The in-process equivalent:

```python
import jittor as jt
jt.flags.lazy_execution = 0
```

## Diagnose out-of-memory

When a single iteration exceeds device memory, first use `jt.profile()` to see who is alive
at the peak (see [Performance and memory profiling](../notes/profiling.en.md)), then reduce
the model or batch size. Explicit synchronization plus collection helps determine whether a
temporary graph is being retained:

```python
for batch in dataset:
    train_step(batch)
    jt.sync_all()
    jt.gc()
```

CUDA convolution can limit cuDNN's workspace to reduce peak allocation, at the cost of speed:

```python
jt.cudnn.set_max_workspace_ratio(0.0)
```

If device memory **grows iteration by iteration**, check the graph and held-variable counts:

```python
for batch in dataset:
    train_step(batch)
    jt.sync_all()
    jt.display_memory_info()
```

`lived_var` or `lived_op` rising steadily usually means a graph is being retained, or a
global variable is still participating in differentiation.

> Device memory shown by `nvidia-smi` not dropping after release is **normal** -- the
> caching allocator has not returned blocks to the driver. Only `jt.clean()` actually hands
> it back.

## Debug segmentation faults

Build with debug information and attach GDB automatically:

```bash
export JT_BUILD_DEBUG=1
export JT_GDB_ATTACH=1
python reproduce.py
```

Jittor reads settings from the environment under only two namespaces: **`JT_BUILD_*`
decides what gets compiled, and `JT_*` decides how the compiled core behaves**.
`python -m jittor_utils.env_manifest` lists them all.

Most settings also accept the historical unprefixed lowercase names (`gdb_attach=1` still
works and prints a notice once at startup). `debug` is the exception: an ordinary English
word is more likely to be another tool's variable, so it is read only as `JT_BUILD_DEBUG`.

When reporting a native crash, include the minimal reproduction, complete logs, compiler
version, and device information.

## Manage the compile cache

Jittor keeps compiled kernels, downloaded toolchains, datasets, and weights in its cache
directory. After upgrading the compiler, driver, or system, clear the entire cache:

```bash
python -m jittor_utils.clean_cache all
```

To see finer-grained cleanup targets:

```bash
python -m jittor_utils.clean_cache help
```
