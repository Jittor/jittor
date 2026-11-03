# Pipelined lazy execution

[Section contents](../index.en.md) · [中文](pipelined-execution.md)

Use `auto_flush_ops` to overlap CUDA execution with Python graph construction. Earlier launches can improve latency but materialize intermediates across graph segments, increasing peak memory. Measure both effects on your workload.

## Mechanism

`jt.flags.auto_flush_ops = N` flushes the pending graph early. Every operator
construction counts; once `N` operators have been created since the executor last
ran, the next `VarHolder` construction hands all current pending work to the
executor and returns, **without any device synchronization**. The device starts on
that segment while Python continues building the rest of the step.

The counter is anchored on each `Executor::run_sync`, so the split points land at
the same place every step of a training loop, and each segment compiles to the
same fused kernels as the previous step.

Lazy-execution semantics are preserved:

- **Only values held by Python are targets.** Intermediate values nobody holds are
  still computed only when some held value needs them, and dead-code elimination
  and fusion within a segment are exactly as before; a segment boundary can at
  most materialize one intermediate that a later consumer could otherwise have
  fused away.
- **Flushes do not nest**, and execution errors are never raised from the
  `VarHolder` construction that triggered them. A failed flush parks the pipeline
  and leaves the failing operator pending, so the caller's own `sync` raises the
  error — the same behavior as under lazy execution; a successful sync resumes the
  pipeline.
- **`jt.tape` outputs are never flush targets.** `tape_together` needs to attach
  these operators to the `Tapes` node after `jt.Function.execute` finishes, so it
  must find them still pending; `Tapes` now asserts this rather than corrupting
  the liveness bookkeeping.

**This switch only applies to CUDA.** CPU kernels execute synchronously on the
calling thread; flushing early would only lose fusion at segment boundaries.

```{warning}
**It is not a pure scheduling knob.** Flushing early changes memory residency, and
cuDNN's convolution algorithm is chosen by **timing candidates** — and the timing
depends on what is in memory at the time. So changing `auto_flush_ops` changes the
algorithm convolution picks, which **changes the gradients** (measured relative
difference `3.8e-4`, deterministic), while **the forward loss is bit-identical**.
For reproducible training numerics, call `jt.cudnn.set_benchmark(0)` first. See
the [numerics contract](../numerics/numerics-contract.en.md).
```

## Choose and measure the threshold

- `auto_flush_ops=0`: leave the whole graph pending until an explicit execution trigger. This gives the scheduler more opportunity to release intermediates and fuse work.
- A positive threshold: submit graph segments early. Smaller values create more boundaries; they do not guarantee lower memory usage.
- The CUDA default is 128. CPU execution does not benefit from this asynchronous overlap.

Compare 0 with the existing setting after warmup, keeping inputs, precision and cuDNN algorithm policy fixed. Record step latency and peak memory with [profiling](../../performance/profiling.en.md). Historical model-specific gains do not predict the gain for a different graph.

If the profile is dominated by Python or executor work, changing the threshold cannot remove that work. See [static graph replay](../../development/runtime/static-graph-replay.en.md) for reuse across compatible steps. Synchronizations inside the model can also drain the pipeline.


## External calls during capture

`graph_replay` and `capture_step` can only replay operations recorded in the native
graph. When directly writing outputs through pointers, launching external kernels,
or invoking callbacks with side effects, callers should mark a capture boundary
before that operation:

```python
jt.graph_replay_barrier("external library launch")
```

Capture reports the refusal reason and subsequent calls keep executing the
original function. Torch `Tensor.data_ptr()` and the Triton bridge launcher mark
this boundary automatically; other external calls still need to be marked
explicitly. A mark only affects the capture currently being built on the current
thread, and is propagated to nested captures. Calling it outside a capture does not
affect later pure native graphs.

When a native CUDA graph is already recording, both explicit and automatic replay
submit ordinary lazy operations, and the outer recording owns those operations; it
does not synchronize inputs inside, nor create another layer of device recording.

## Measure a steady-state step

Use [benchmarking](../../performance/benchmarking.en.md) for timing and [profiling](../../performance/profiling.en.md) to distinguish host work, device work and idle gaps. If using nsys, split the warmed-up timeline into steps. A whole-process kernel summary includes compilation, algorithm trials and shutdown, so it is not a steady-state comparison.
