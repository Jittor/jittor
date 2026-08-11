# Debugging Jittor programs

Jittor executes graphs lazily and compiles operators at runtime. The following
diagnostics trade performance for more precise failure information, so enable
them only while reproducing a problem.

## Find NaN and infinity values

Enable output checking and Python source tracing before starting the process:

```bash
export JT_CHECK_NAN=1
export trace_py_var=3
python reproduce.py
```

`JT_CHECK_NAN=1` stops when an operator emits an invalid floating-point value.
`trace_py_var=3` records the Python call site associated with generated
operators. Both settings trigger recompilation and add substantial overhead.

## Make asynchronous failures local

Lazy execution can report an error after the Python line that created the
failing operator. For a small reproducer, temporarily switch to eager
execution:

```bash
export lazy_execution=0
```

The equivalent process-local setting is:

```python
import jittor as jt

jt.flags.lazy_execution = 0
```

## Diagnose memory exhaustion

If one iteration exceeds available memory, first reduce the model or batch
size. Explicit synchronization and collection can help establish whether
temporary graphs are being retained:

```python
for batch in dataset:
    train_step(batch)
    jt.sync_all()
    jt.gc()
```

For CUDA convolutions, limiting cuDNN workspace can reduce peak allocation at
the cost of performance:

```python
jt.cudnn.set_max_workspace_ratio(0.0)
```

If memory grows across iterations, inspect graph and held-variable counts:

```python
for batch in dataset:
    train_step(batch)
    jt.sync_all()
    jt.display_memory_info()
```

An increasing `lived_var` or `lived_op` count usually indicates a retained
graph or a global variable that still participates in differentiation.

## Debug a segmentation fault

Build with debug information and attach GDB automatically:

```bash
export debug=1
export gdb_attach=1
python reproduce.py
```

Include a minimal reproducer, the complete log, compiler version, and device
information when reporting a native crash.

## Manage the compilation cache

Jittor stores compiled kernels, downloaded toolchains, datasets, and weights in
its cache directory. After a compiler, driver, or system upgrade, clear all
cached state with:

```bash
python -m jittor_utils.clean_cache all
```

List narrower cleanup targets with:

```bash
python -m jittor_utils.clean_cache help
```
