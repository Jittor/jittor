# Making a repeated step stop rebuilding its graph in Python

- Status: implemented (`35cedb1a3` inference graph replay, `2f794e289` training step capture)
- Last reviewed: 2026-10-14
- Basis: [the 2026-09-14 parity report](../results/2026-09-14-jittor-vs-pytorch.en.md),
  [the 2026-09-24 real-model parity run](../results/2026-09-24-torch-compat-real-models.en.md)
- Recheck when: the capture guards, the automatic-policy thresholds, or the replay counters in
  `jt.profile` change

## Why this one

Across twelve parity items against real PyTorch the pattern was **device-bound wins, host-bound
loses**: convolution 0.66/0.85 and b8s256 training 0.51 all won; decode (batch 1, seq 1) lost
at 1.97, and d1024 decode lost at 2.05. The cause of the host-bound cases was measured five
times and the conclusion did not change:

1. **Phase breakdown**: of the 2.70 ms of one decode step, Python graph construction is about
   1.6 ms, kernel launches 0.54 ms, and planning only 0.12 ms.
2. **Per-operator device time** (measured the same way on both sides: enqueue hundreds of
   repetitions of the same operator, synchronize only at the end, then divide): gemm 1.00x,
   transpose 0.99x, elementwise 1.07x -- the device side is not the problem.
3. **Node count**: removing one broadcast node is worth 2% of the whole step; node count is not
   the lever.
4. **Frontend ceiling**: strip out the whole `nn.Module` / functional layer and a block goes
   from 202.1 to 151.4 us -- the frontend is only 25% of graph construction.
5. **The floor**: even with a zero-cost frontend, decode still costs 1.21 (graph construction)
   + 1.0 (executor) = 2.2 ms, which is still 1.6x against torch's 1.37.

So everything left is the fact that **the same graph is rebuilt in Python on every step**.
`ExecPlan` has long been a value that could be kept (`src/core/exec_plan.h`: planning only reads
the graph and writes batch numbers, it neither allocates nor launches); what was missing was
**letting the graph outlive execution, accept new inputs, and refuse to replay once it no longer
holds**.

## What exists now

Two entry points, both "capture once, replay afterwards"; the difference is whether what was
captured updates state:

| Entry point | What it captures | Implementation |
| --- | --- | --- |
| `jt.graph_replay(module, *example_inputs)` | one **inference** call: the module is a function of inputs and parameters, and the call leaves nothing behind | [`python/jittor/_runtime/graph_replay.py`](https://github.com/Jittor/jittor/blob/master/python/jittor/_runtime/graph_replay.py) |
| `jt.capture_step(fn)` | one whole **training** step: forward, backward and the optimizer update, together with the parameters, momentum and normalization statistics it rewrites | [`python/jittor/_runtime/step_capture.py`](https://github.com/Jittor/jittor/blob/master/python/jittor/_runtime/step_capture.py) |

The Torch frontend wires `torch.compile(..., mode="reduce-overhead")` to these two: a wrapped
module goes through inference replay, a wrapped function through step capture.

There is also an **on-by-default** automatic policy (`JT_AUTO_GRAPH_REPLAY=1`): it looks only
at the outermost module call under `no_grad`, steps in only after the same shape appears twice
in a row, accepts only calls whose input Vars total less than `auto_graph_replay_bytes`
(4 MB by default), and records a device graph only while the working set stays within
`auto_graph_replay_retain_bytes` (256 MB by default). **Input size is the cheap criterion
separating "graph construction saved by replay" from "extra input/output copies"**: measured,
an mlp with 1 MB inputs goes 0.14 -> 0.05 ms, batch-1 ResNet-50 (0.6 MB) 5.9 -> 0.7 ms, and
one SD1.5 denoising step (0.3 MB) 42 -> 20 ms, all worth entering; an 8 MB input,
device-bound mlp (0.49 -> 0.52 ms) is not. Everything else executes as usual.

The recording cap bounds **the extra working set of each device recording**, not the process's
total device memory; weights, other modules or caches of other signatures are outside this
per-recording budget; private fixed input and return buffers also add to the actual resident
cost. `0` means unlimited, and an explicit `jt.graph_replay` is not subject to the automatic
budget either. When the budget is changed through
`jt.flag_scope(auto_graph_replay_retain_bytes=...)`, the next automatic-policy-eligible
`no_grad` call releases that module's existing ordinary captures and stateful-object captures
and re-captures under the new budget; calling again after leaving the scope re-captures under
the restored budget as well. Exceeding the budget only refuses device recording; replay through
the executor and its memory-reclamation mechanism remains available, and calls that leave
`no_grad` keep building an ordinary differentiable forward.

## Guards: when a capture is invalidated, and when it is forbidden outright

A captured graph is **a snapshot of one set of shapes, one set of parameters and one Python
path**, and on its own it cannot tell that any of them changed. Every call therefore checks,
discards and re-captures once a condition no longer holds, rather than answering with a stale
result:

- the shape, dtype and count of the parameters;
- whether the module is still in the same training mode;
- whether any parameter was rebound (optimizer update, load, device migration) -- the graph
  reads the parameter Vars it captured, an update replaces them, and otherwise it would
  **quietly answer with the old weights**;
- whether the captured graph has not been finished yet. **This is the most important one**: a
  finished graph does not raise; it keeps answering with the values computed last time.

Two cases are **fundamentally not replayable**, and capture refuses outright instead of
answering wrongly: the graph **draws random numbers** (replay would repeat the same draw), and
the traced call **reads values back to the host** (after that the Python path depends on tensor
values, and the next call may take a different one). Training-step capture additionally
requires the Python-side accounting to declare itself -- `live_array()` (a small array refilled
before every replay, such as fused AdamW's step and learning rate), `on_replay()` (a hook run
before every replay), `guard()` (re-capture when a read value changed), `refuse()` (say
plainly that this step cannot be replayed). Device recording also checks before every launch
whether a leaf has moved (`graph_leaves_moved`).

`replay.stats` / `step.stats` report what actually happened -- how many times it replayed, how
many times it fell back, and why. `prof.replay` in `jt.profile` reads the same counters (see
[profiling and memory](../notes/profiling.en.md)). **A silent fallback is worse than no
replay**, so these counters are part of the conclusion, not optional debugging output.

## What it is worth, and two ways of measuring that fool yourself

The same decode step, with eager and replay measured in **separate processes**, with new inputs
each time, and every answer compared against eager:

| tf-d512-L8 decode b1s1 | Time |
| --- | --- |
| eager | 2.54–2.67 ms |
| replay | 1.57–1.79 ms |
| reference PyTorch | 1.37–1.43 ms |

About 1.5x, reproducible across runs; most of the gap is closed, but not all of it. Other
shapes are **not** listed, because they are not stable on this machine: the b8s256 forward
jumps between two states about 2x apart (14.06 ms or 6.49 ms), and the eager side does the
same, as does prefill. **Measure your own model.**

Two measurements that fooled the author:

- **Feeding the captured Var back in measures nothing.** When the input is the captured Var
  itself, replay skips the input copy, no new input is written, and the graph does not
  re-execute at all -- the call degenerates into copying the output it already holds, and the
  answer is still correct. "Replaying" this way on b8s256 gave 6.8 ms against eager's 14.0 ms.
- **Measuring both arms in one process measures ordering.** The arm that runs second starts
  from a warm state: running the exact same eager code in both arms (replay refused), the
  second measurement gave 1.57 ms and the first 2.74 ms.

**Replay is not free money.** A graph the runtime already overlaps well replays slower -- an
mlp forward is 0.33 ms eager and 0.42 ms replayed, and a convolution stack is a wash.
`measure=True` can A/B once automatically at capture time and permanently refuse if it loses,
but it is **off by default**: that measurement perturbs the very process it measures, making
every subsequent replay launch 219 kernels instead of 132 and the wrapper 30% slower -- larger
than the margin it is meant to protect.

## Where not to start

- **Do not start by "making the frontend thin".** The ceiling has been measured: the frontend
  is only 25% of graph construction, and removing all of it is not enough.
- **Do not use `Op::duplicate()` for replay.** It returns `nullptr` in the `Op` base class;
  only `BroadcastToOp` and `ReindexOp` implement it in the whole tree, so it is not a general
  primitive.
- **Do not count on reducing the node count.** Removing one broadcast node is worth 2% of the
  step.
- **Do not unfinish after the fact.** `finish_pending_liveness()` starts with
  `if (is_finished()) return;`; clearing the flag would make it **release the pending liveness
  of the same inputs a second time**, underflow the reference count, and then use-after-free.
  Keeping a graph means **not finishing on the first execution at all** (`keep_graph`), not
  undoing the finish afterwards.

## Tests

`tests/core/test_graph_replay.py`, `tests/core/test_graph_capture.py`,
`tests/core/test_graph_replay_retention.py`, `tests/core/test_step_capture.py` and
`tests/nn/test_graph_replay_multi_output.py` cover the guards, refusals and the multi-output
structure; the Torch frontend path is in
`compat/tests/torch/test_torch_compile_replay.py`.
**Every replay result must be compared against eager** -- the failure mode of replay is
"answers fast but answers with stale values", and it does not report itself.
