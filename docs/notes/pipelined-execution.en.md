# Pipelined lazy execution

Jittor builds graphs lazily: Python creates meta-operators, and nothing actually
executes until a value is read or `sync` is called, at which point the executor
fuses and launches the entire pending graph at once. On a GPU this serializes two
things PyTorch overlaps — the device idles while the CPU builds the graph, and the
CPU waits while the device computes.

A measured BERT-base training step (RTX 4090, TF32 enabled on both sides, nsys
timeline split by step):

| | PyTorch | Jittor (fully lazy) |
| --- | --- | --- |
| GPU kernel time | 27.3 ms | 24.1 ms |
| Wall-clock time | 29.9 ms | 30.1 ms |
| GPU idle | 2.6 ms | 6.0 ms |

**Jittor's kernels are faster, but this step is slower** — the device sits idle
during the 9 ms Python spends building the forward graph, clearing gradients and
building the backward graph.

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
the [numerics contract](numerics-contract.en.md).
```

## Effect and boundaries

With `N` between 64 and 256, GEMM-dominated transformer steps catch up with
PyTorch's wall-clock time (milliseconds per step on the same device):

| Case | Lazy | Pipelined | PyTorch |
| --- | --- | --- | --- |
| BERT-base | 31.7 | 26.4 | 26.4 |
| ViT-base | 29.3 | 24.8 | 23.8 |
| Llama (8 layers, hidden 1024) | 43.4 | 39.1 | 37.8 |
| GPT-2 (8 layers, hidden 1024) | 42.2 | 38.7 | 40.2 |

The remaining gap comes from drains caused by synchronization points inside the
model itself (`transformers` reads mask values on the host three times per step in
BERT), and from the latency before the first segment launches.

**Pipelining hides device time behind CPU time; it cannot hide CPU time itself.** A
diffusers UNet2D step needs 22 ms of GPU work but 42 ms of CPU work — 16 ms in the
executor (fusion, scheduling, key construction and launching of about 1500
kernels), 16 ms building the backward graph, 9 ms building the forward graph — and
its wall-clock time does not change with pipelining. That case needs the work of
each step reused across identical steps instead of rebuilt every step — which is
what [static graph replay](../development/static-graph-replay.en.md) does
(`jt.graph_replay` and `jt.capture_step`, plus the automatic policy that is on by
default).

## How much memory it costs

Pipelining is not free: flushing early splits the graph into segments, intermediate
values live across segment boundaries cannot be freed, and peak usage rises. One
ResNet-50 training step (batch 32, 224×224, fp32, TF32 off, `set_benchmark(0)`,
RTX 4090, an external process polling `cudaMemGetInfo` for whole-device usage, same
ruler on both sides):

| `auto_flush_ops` | Peak usage | Time per step |
| --- | --- | --- |
| 0 (whole graph in one launch) | 3.574 GiB | 0.0881 s |
| 128 (default) | 5.104 GiB | 0.0600 s |

The default spends 43 % more memory for 32 % more speed. Under the same
configuration PyTorch (`cudnn.benchmark=False`) is 3.74 GiB, which is to say
**with pipelining off the two are roughly even on memory, and pipelining itself
brings the extra 1.5 GiB**.

Measured phase by phase, the excess is entirely in the backward segment; weights
and the forward side are nearly the same:

| Phase (cumulative) | PyTorch | Jittor |
| --- | --- | --- |
| Weights | 0.113 | 0.129 |
| + input batch | 0.133 | 0.148 |
| + forward | 2.816 | 3.025 |
| + backward | 3.262 | 5.104 |
| + optimizer state | 3.295 | 5.104 |

Running 11 more steps grows neither, so this is a steady-state peak, not a leak.
PyTorch's own `memory_allocated` count drops from 2.676 to 0.230 after backward,
meaning it returns each activation as soon as its gradient is done; Jittor's
segment boundaries let some activations outlive their original last consumer.

```{note}
**The direction is counterintuitive.** Being more thoroughly lazy
(`auto_flush_ops=0`) is the memory-saving side, not the wasteful side: when the
whole graph reaches the scheduler it can see every tensor's last consumer. Making
`N` smaller does not save more, it costs more — measured, `N=1` is 7.312 GiB and
`N=8` is 7.227 GiB. When you run out of memory, `auto_flush_ops=0` is the first
switch to try.
```

The above is a single point at batch 32. The absolute size of the gap grows with
batch and resolution; the ratio may not; and we have not measured which tensors
specifically live across segments — pinning that down requires `use_stat_allocator`
to see lifetimes.

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

## How to measure

The gate is `tests/compat/torch/test_ecosystem_speed.py`. To attribute a gap,
capture the harness runner's timeline with nsys, split into steps using a kernel
that appears a fixed number of times per step, then compare per-step kernel totals,
wall-clock spans and idle time. **Do not read a whole-run kernel summary as
steady state** — it includes cuDNN algorithm trials and process exit.
