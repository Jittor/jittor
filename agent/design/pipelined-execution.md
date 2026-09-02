# Pipelined Lazy Execution

Jittor builds a step lazily: Python creates meta-operators, nothing runs until
a value is fetched or `sync` is called, and the executor then fuses and
launches the whole pending graph at once. On a GPU this serialises two
things that PyTorch overlaps: the CPU builds the graph while the device is
idle, then the device computes while the CPU waits.

Measured on a BERT-base training step (RTX 4090, TF32 on both runtimes, per
step, nsys timeline cut at the step boundary):

| | PyTorch | Jittor, fully lazy |
| --- | --- | --- |
| GPU kernel time | 27.3 ms | 24.1 ms |
| wall time | 29.9 ms | 30.1 ms |
| GPU idle | 2.6 ms | 6.0 ms |

Jittor's kernels are faster; the step is slower because the device sits idle
for the 9 ms in which Python builds the forward graph, zeroes gradients and
builds the backward graph.

## The mechanism

`jt.flags.auto_flush_ops = N` launches the pending graph early. Every
operator construction is counted; once `N` operators have been created since
the executor last ran, the next `VarHolder` construction hands everything
pending to the executor **without a device sync** and returns. The device
works on that segment while Python keeps building the rest of the step. The
counter is anchored to every `Executor::run_sync`, so the flush points fall
at the same positions in every step of a training loop and each segment
compiles to the same fused kernels as the step before.

The semantics of lazy execution are kept:

- Only values Python holds are targets. An intermediate nobody kept is still
  computed only when a held value needs it, so dead-code elimination and
  fusion apply within each segment exactly as before; a segment boundary can
  at most materialise one intermediate that a later consumer would have
  fused.
- A flush is never nested, and an execution error is never raised from the
  `VarHolder` constructor it runs in. A failed flush suspends the pipeline
  and leaves the failing operators pending; the caller's own `sync` raises
  the error, as it would have under lazy execution, and a successful sync
  resumes flushing.
- An output of `jt.tape` is never a flush target. `tape_together` wires those
  operators into a `Tapes` node after `jt.Function.execute` has run and must
  find them pending; eager execution has always applied the same exclusion.
  `Tapes` now asserts it rather than corrupting liveness bookkeeping.

The flag only acts on CUDA. CPU kernels run synchronously on the calling
thread, so launching early would only cost fusion at segment boundaries.

## What it fixes and what it does not

With `N` in the 64–256 range, the GEMM-dominated transformer steps reach the
PyTorch wall time (ms per step on the same device):

| case | lazy | pipelined | PyTorch |
| --- | --- | --- | --- |
| BERT-base | 31.7 | 26.4 | 26.4 |
| ViT-base | 29.3 | 24.8 | 23.8 |
| Llama (8 layers, 1024 hidden) | 43.4 | 39.1 | 37.8 |
| GPT-2 (8 layers, 1024 hidden) | 42.2 | 38.7 | 40.2 |

The remaining difference is the drain at the synchronisation points the
model itself contains (`transformers` reads a mask value on the host three
times per step in BERT) and the latency before the first segment launches.

Pipelining hides device time behind CPU time; it cannot hide CPU time. A
diffusers UNet2D step needs 22 ms of GPU work but 42 ms of CPU work -- 16 ms
in the executor (fusion, scheduling, key construction and launch for about
1500 kernels), 16 ms building the backward graph, 9 ms building the forward
-- and its wall time does not move. That regime needs the executor's
per-step work cached across identical steps, which is the next stage of this
design.

## Measuring it

`tests/compat/torch/test_ecosystem_speed.py` is the gate. To attribute a gap
use an nsys timeline of the harness runner, cut into steps by a kernel that
recurs a fixed number of times per step, and compare kernel-sum, wall span
and idle per step; the whole-run kernel summary also counts the cuDNN
algorithm trials and process teardown and must not be read as steady state.
