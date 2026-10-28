# Numerics contract

- Status: accepted
- Last reviewed: 2026-09-11
- Baseline: `2d716db31`
- Review trigger: when the test file behind any entry on this page goes red, or its
  assertion is loosened

This page answers one question: **for the same piece of code on CPU and CUDA, under
different scheduling settings, what must be identical and what is allowed to
differ?**

It is written for people who already have a model running and are starting to
suspect "are the two devices disagreeing because I did something wrong?". Each
entry has three parts — **what you see / why / what to do** — followed by **the
test file that pins it**. All numbers below are measured, with the source given in
each entry; the test file is the executable copy, and prevails when this page
grows stale.

Indexed by symptom:

| What you see | Which entry |
| --- | --- |
| `1/0` gives `nan`, and a fully masked attention row does not become `nan` | {ref}`IEEE special-value arithmetic <numerics-ieee>` |
| A tiny positive number becomes 0 on CUDA, `log` gives `-inf`, `count_nonzero` is off by one | {ref}`CUDA subnormals <numerics-subnormal>` |
| `x.max()` cannot see a NaN in the tensor; the sign of `maximum(-0.0, 0.0)` | {ref}`maximum/minimum vs max/min <numerics-minmax>` |
| float64 `log`/`sin` only have float32 precision on CUDA | {ref}`float64 unary math <numerics-float64>` |
| Only `auto_flush_ops` changed; the loss is bit-identical but the gradients differ | {ref}`convolution algorithm auto-selection <numerics-autotune>` |
| `profile_scope` reports 0 rows; changing a flag changes the result | {ref}`consequences of auto_flush_ops <numerics-flush>` |
| Summing sixteen million numbers on CPU is much worse than NumPy | {ref}`CPU reduction block accumulation <numerics-reduce>` |

What does not hold yet is listed separately in the {ref}`last section <numerics-open>` — **every entry there currently computes the wrong answer**.

(numerics-ieee)=
## IEEE special-value arithmetic

**What you see.** Arithmetic involving infinities and NaN gives well-defined
answers per IEEE-754 on both CPU and CUDA, consistent with NumPy:

| Expression | Result |
| --- | --- |
| `inf - inf`, `inf / inf`, `0 * inf`, `inf + -inf` | `nan` |
| `1 / 0` | `inf` |
| `-1 / 0`, `-inf / 0` | `-inf` |
| `inf * 1` | `inf` |
| `nan + 1`, `nan * 0` | `nan` |

`x != x` still works as the "is this NaN" test.

**Why it deserves its own entry.** Before 2026-09-10 it was not like this: the CPU
JIT kernels were compiled with `-Ofast`, which implies `-ffast-math` and hence
`-ffinite-math-only` — a promise that no operand is ever infinity or NaN. The
compiler optimized under that promise, and operands that **really were infinite**
went down whatever path the transformed code happened to produce: for float32
with length ≥ 4 (the vectorized path), `1 / 0` gave `nan` and `-inf / 0` gave
`nan`. A single element was correct, so scalar probing could not see it.

The dangerous case is not the obviously broken one. A fully masked attention row
subtracts its own `-inf` maximum; a finite value there produces a well-shaped but
wrong softmax instead of an obvious `nan`. The CUDA side was correct at the time,
so the two devices gave different answers for the same expression.

**What to do.** Nothing: CPU kernels now compile at `-O3` (commit `1e50d76c5`,
`KI-BACKEND-005`). The only thing to watch is that **adding `-Ofast` to your own
`cc_flags` / `kernel_flags` undoes this**, and the failure is silent.

On cost: this replacement showed **no measurable cost** on four shapes —
elementwise chains, `exp`/`sqrt` chains, `sum`, and matmul (4M elementwise chain
0.000320 → 0.000324 s, `sum` 4M 0.000737 → 0.000719 s). One shape did pay:
float32 `max`/`min` reductions were reassociated and vectorized under `-Ofast` at
28–31 GB/s, versus 14 GB/s under `-O3`. That item is recorded as `KI-OPS-006` and
has not been papered over.

**Tests**: [`tests/ops/test_ieee_arithmetic.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_ieee_arithmetic.py) —
ten expressions with exact IEEE-754 definitions, two devices, **length 8** (at
length 1 everything passes even when the flags are wrong), plus the `x != x` test
itself.
[`tests/structure/codegen/test_kernel_math_flags.py`](https://github.com/Jittor/jittor/blob/master/tests/structure/codegen/test_kernel_math_flags.py)
is its cheap companion: it names the flag so a reintroduction fails with a clear
message about what changed, and separately asserts the optimization level is still
present — otherwise "delete the flag and put nothing in its place" would satisfy
the first assertion while making things worse.

(numerics-subnormal)=
## CUDA subnormals are flushed to zero

This is a **decision**, not a defect, but it was not written down anywhere, so
every device comparison that hits it reads like a defect.

**What you see.** The smallest normal number in float32 is `1.18e-38`. Nonzero
values below it (subnormals) are preserved per IEEE-754 on CPU and **flushed to
zero by default** on CUDA:

| Value | CPU | CUDA default | CUDA `strict` | NumPy |
| --- | --- | --- | --- | --- |
| `1e-45` (subnormal) | preserved | **0.0** | preserved | preserved |
| `1e-40` (subnormal) | preserved | **0.0** | preserved | preserved |
| `1e-30` (normal) | preserved | preserved | preserved | preserved |

**Why.** The root cause is nvcc's `--use_fast_math`, which implies `-ftz=true`.
The consequence goes beyond the value itself: the input is already zero **before**
it enters the function, so `log(1e-45)` gives the correct `-103.28` on CPU and
`-inf` on the CUDA default; `count_nonzero([1e-45])` is 1 on CPU and 0 on the CUDA
default. A gradient that underflows into the subnormal range is exactly zero on
CUDA and a tiny nonzero on CPU, so the two devices take different update paths for
the same model — invisible until someone compares them side by side.

**What to do.**

```python
jt.flags.cuda_kernel_math = "strict"     # 去掉 --use_fast_math
```

This switch takes effect per process (`src/runtime/jit_policy.cc`), and switching
drains the pending graph first. Measurements show it **fully restores**
subnormals: the `strict` column in the table above is measured, not inferred.

Cost: measured on 2026-09-10 on an RTX 4090, two alternating rounds, minima taken:

| Operator (16M float32) | Default | strict |
| --- | --- | --- |
| `divide` | 218.6 µs | 218.8 µs |
| `sqrt` | 148.1 | 148.8 |
| `exp` | 150.0 | 150.2 |
| `log` | 148.1 | 148.2 |
| `mul-add` (control, unaffected by fast-math) | 218.6 | 218.6 |

**At this scale the cost is not measurable**: the differences are 0.1–0.8 %, the
same order as the control group unaffected by fast-math. The reason is that these
kernels are memory-bandwidth bound, so the ALU cycles fast-math saves were already
hidden behind memory access.

For compute-dense cases **no conclusion was reached**: after chaining
`exp/log/sqrt` into 8-layer and 32-layer chains, the difference between the same
tier across two rounds (1.01e-3 versus 4.16e-4, 2.4×) was **larger** than the
difference between the two tiers — that is variation in how fusion happens, not an
effect of the policy. So all we can say here is **not measurable**, not "no cost".
A reliable number needs a benchmark with a fixed fusion shape. The default was
therefore left alone: there is no evidence that flushing is the wrong trade-off.

**Tests**: [`tests/backends/parity/test_subnormal_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/parity/test_subnormal_contract.py)
asserts the table above. It is not saying which behavior is more correct, but
**what the current behavior is**: flushed by default, preserved under `strict`,
always preserved on CPU. If any of those three changes, that test goes red —
including for a good change such as "someone made strict the default", in which
case what should change is this document and that test, not letting them keep
describing a world that no longer holds. Its teeth are that the two tiers **must
actually differ**: pointing the `strict` case at the default tier makes it red.

(numerics-minmax)=
## maximum/minimum vs max/min: NaN and signed zero

**What you see.** Elementwise `maximum`/`minimum` and the `max()`/`min()`
reductions both handle NaN and signed zero per **NumPy's semantics**, identically
on both devices:

- **NaN propagates**. `x.max()` returns `nan` when the tensor contains a NaN,
  regardless of position, sign, or tensor size.
- **Signed zero is order-dependent, deliberately**. NumPy decides `maximum` with
  `>` alone, so `maximum(-0.0, 0.0)` is `+0.0` and `maximum(0.0, -0.0)` is `-0.0`,
  with `minimum` mirrored. Jittor now reproduces this bit for bit.
- Integers are unaffected.

**Why it deserves its own entry.** Before 2026-09-10 all three disagreed. For
`f = [nan, -inf, -0.0, 0.0, inf]` at zero, float32: CPU gave
`[nan, 0.0, -0.0, 0.0, inf]`, CUDA gave `[0.0, 0.0, 0.0, 0.0, inf]`, NumPy gave
`[nan, 0.0, 0.0, 0.0, inf]`. The reduction was worse and **the two devices agreed
by being consistently wrong**: on an array containing one NaN, `jt.max`/`jt.min`
returned `1.0` at n = 5, 4096 and 1048576, while NumPy returns `nan`. `x.max()` is
the usual way to tell whether a tensor has gone bad, and at the time it could not
see NaN at all.

The root cause was two spellings, each "obvious", but neither chosen for NaN
behavior: `std::max(a, b)` is `a < b ? b : a`, all comparisons against NaN are
false, so it returns the operand **written first**; the reduction folds
`acc = max(acc, x)`, so an incoming NaN is always in the second position and gets
dropped. CUDA's `::max` lowered to `fmaxf`, i.e. IEEE `maxNum`, which
**deliberately** returns the non-NaN one, so both positions were lost.

**What to do.** Nothing; `KI-BACKEND-004` (commit `0f037d790`) fixed it. Also
fixed along the way was `KI-BACKEND-007`: `jt.std([nan, 1.0, 2.0])` used to be
`nan` on CPU and `0.0009999999310821295` on CUDA, while `norm` was `nan` against
`1e-15`; those are composite reductions passing through the
`maximum`/`minimum` line, and once that line was fixed both devices give `nan`.

The cost is measured and **not zero**: CPU `max`/`min` **reductions** are 1.9–2.0×
slower (1M: 14.29 → 7.24 GB/s; 16M: 13.99 → 7.25 GB/s; `sum` as a control, 1.01×).
Elementwise `maximum`/`minimum` and softmax, the closest real-world case, show no
measurable cost. Half the remaining cause is not the comparison itself but the
reduce kernel indexing with a runtime stride, which blocks vectorization — an
eight-way partial sum matches `std::max` at unit stride. This is recorded as
`KI-OPS-006` and not yet closed.

**Tests**: [`tests/ops/test_minmax_nan_propagation.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_minmax_nan_propagation.py),
22 cases, including a **device-consistency class** — the same operands run on both
devices, compared with each other and with NumPy, so a future divergence cannot
slip through because "each device looks reasonable on its own". On the unfixed
tree, 16 of the 22 failed.

(numerics-float64)=
## float64 unary math is no longer narrowed on CUDA

**What you see.** float64 `log`, `sqrt`, `sin`, `tan`, `tanh`, `arctan`, `arcsin`,
`exp`, `erf` and so on **are computed at float64** on CUDA, bit-identical to CPU
and NumPy.

**Why it deserves its own entry.** Before 2026-09-11 they were not. Nineteen
entries in CUDA's expression table spelled the function name as the **float-only**
C variant — `::logf`, `::expf`, `::sinf`, `::tanhf`, `::erff`, etc. — regardless
of operand dtype. float64 operands were thus narrowed to float32, evaluated in
single precision, and scaled back up. Most inputs can hide this; it shows where the
answer itself is below float32 resolution:

```
log(1 + 2**-51)   CPU 4.4408920985006252e-16   CUDA 0.0   NumPy 4.4408920985006252e-16
```

`1 + 2**-51` is **exactly 1.0** in float32, and `log(1.0)` is zero. Not "slightly
off" — gone.

**What to do.** Nothing; `KI-OPS-007` (commit `2d716db31`) fixed it by adding the
same dtype dispatch to those nineteen entries. The text emitted by the float32
operand template is **word-for-word identical** to before, so the single-precision
path is unchanged by construction — measured float32 `log`/`exp`/`sqrt`/`sin`/`tanh`
on 1,000,000 elements show the same gap to NumPy (about 1e-7, caused by
`--use_fast_math`, before and after), and 23–25 µs per call, also unchanged.

**Tests**: [`tests/ops/test_float64_unary_math.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_float64_unary_math.py),
two devices. The inputs are chosen so the float32 answer is **qualitatively** wrong
(equal to zero, or equal to the input itself), not merely slightly less precise —
several tolerance-style tests passed on the old build. It also pins both "these
inputs really do collapse in float32" and "float32 did not change".

(numerics-autotune)=
## Convolution algorithm auto-selection changes gradients

This entry differs from the ones above: what it changes is not some operator's
semantics, but that **the same code trains different weights under different
scheduling** — and **the forward loss is bit-identical**, so any check that watches
the loss stays silent.

**What you see.** Measured, RTX 4090, five bottleneck blocks, 5.98 million
parameter elements:

| | `auto_flush_ops=0` | `=64` | `=256` |
| --- | --- | --- | --- |
| loss | 23080.078125 | 23080.078125 | 23080.078125 |
| gradient norm (algorithm cache 100, default) | 45839.379 | **45822.207** | 45839.379 |
| gradient norm (algorithm cache 100000) | 45839.379 | 45822.207 | **45822.207** |

Relative difference `3.8e-4`, **deterministic** (the same number three times for
the same configuration) — not a race, genuinely computing different things.

**Why.** The convolution algorithm is not chosen by heuristic; it **times
candidates and picks the fastest**, caching the result by shape
(`backends/cuda/kernels/cudnn/cudnn_conv_op.cc`; a code comment gives the reason:
heuristics pick an algorithm 2–3× slower on ordinary fp32 3×3). So which algorithm
wins depends on two things unrelated to mathematics:

* **what is in memory at timing time** — `auto_flush_ops` changes execution
  scheduling, and thus residency;
* **how many shapes have been seen before** — a full cache silently falls back to
  the heuristic
  (`if (fwd_algo_cache.size()>=max_cache_size) benchmark = false;`,
  `max_cache_size` defaults to 100), so `set_algorithm_cache_size` is also a knob
  that changes numerics.

**What to do.** For reproducible gradients, turn auto-tuning off first:

```python
import jittor as jt
jt.cudnn.set_benchmark(0)                    # 强制走确定性的启发式
# Torch 前端等价写法：
# torch.backends.cudnn.benchmark = False
```

Measured, gradients then no longer vary with `auto_flush_ops` (residual `2e-6`,
which is reassociation from different batch boundaries). The cost is that
convolution may pick a slower algorithm — exactly what that code comment says — so
this is **a trade-off, not a pure improvement**, and the default was left alone.

A comparable case is PyTorch's `torch.backends.cudnn.benchmark`: it has the same
property and says so in its documentation. The reason this section exists is to
write ours down too.

**A discipline for anyone running comparison experiments**: any numeric comparison
**across a flag that changes memory residency** must first call
`set_benchmark(0)`. Otherwise the `3.8e-4` band swamps smaller real differences,
and auto-tuning jitter gets misread as the effect of the change under test — that
is exactly how the first fix for `KI-EXEC-001` was wrongly rejected.

**Tests**: [`tests/backends/cuda/test_autotuning_isolation.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/cuda/test_autotuning_isolation.py)
pins the **boundary**, not the behavior: with auto-tuning off, gradients **must
not** vary with scheduling flags (this one has teeth — if something **else** starts
changing numerics in future, it goes red); and regardless of the switch, the loss
must be bit-identical. It does **not** assert "gradients differ when tuning is on"
— that depends on the machine, shapes and cuDNN version, and asserting it on
hardware where it happens to be stable today would produce false reds.

Unexplained residual: with tuning off, four bottleneck blocks still give
33063.723 at `auto_flush_ops=32` versus 33063.688 elsewhere, a relative `1.1e-6` —
three hundred times smaller than the algorithm effect, on the order of "the
accumulation order changed". Not chased further; recorded here so it is not
treated as zero. This is `KI-EXEC-003`, not closed.

(numerics-flush)=
## Consequences of auto_flush_ops

`jt.flags.auto_flush_ops = N` flushes the pending graph early: once every `N`
operators have been created, all current pending work is handed to the executor.
Its mechanism and
benefits are covered in [pipelined lazy execution](pipelined-execution.en.md); here
we cover only its numerical and observational consequences, because it reads like
a pure scheduling knob and is not one.

**It defaults to 128 and only applies to CUDA.**

Three things that have already happened:

1. **It changes gradients** — not because it computes anything wrong itself, but
   because it changes memory residency, and cuDNN's algorithm auto-selection
   depends on residency. See the {ref}`previous section <numerics-autotune>`. This
   is **the only one still open**.

2. **It once segfaulted a ResNet50-style backbone on CUDA.** Five bottleneck
   blocks deterministically segfaulted at `auto_flush_ops` 1, 16, 32 and
   **128 (the release default)**, while 64 and 256 happened not to — which looked
   like "above some graph-size threshold", but the threshold was not size, it was
   the split point landing on a tape for the first time. The root cause was that
   `Tapes` is a **pure control operator** (no `run`, no `jit_run`, reading no byte
   of its inputs), while `run_exec_plan` treated it by **compute-operator** rules:
   migrating every input to the device, asserting every input has storage. With the
   whole graph in one batch, the two never met. Fixed (`KI-EXEC-001`, commit
   `47cedd14d`): 0, 1, 16, 32, 64, 128, 256, 512 all run without crashing, the loss
   is bit-identical, and with cuDNN auto-tuning off the gradient residual across
   settings is at most `2e-6`.

3. **It once made `jt.profile_scope` return an empty report, silently.** A graph
   built before the `with` may already have run, so graphs above the threshold got
   0 rows while the results were correct. Fixed (`KI-EXEC-002`, commit
   `3b4e646e5`): `profile_scope` now sets `auto_flush_ops` to 0 within its scope
   (unless the caller overrides it explicitly), and the 64- and 32-slice cases went
   from **0 rows to 9 rows**; work that ran before the scope opened **cannot be
   recovered**, and it now raises a `RuntimeWarning` naming the reason — previously
   "ran very fast" and "measured nothing" were the same output.

**What to do.**

- For **reproducible training numerics**: `jt.cudnn.set_benchmark(0)`. Setting
  `auto_flush_ops=0` alone also makes one run self-consistent, but that gives up
  the pipelining benefit and does not prevent other residency changes.
- For **profiling**: build the graph inside `profile_scope`. That is the window you
  meant to measure anyway.
- For **comparing the numeric effect of two flags**: turn auto-tuning off first,
  see the previous section.

**Tests**: [`tests/backends/cuda/test_auto_flush_graph_split.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/cuda/test_auto_flush_graph_split.py).
Each setting runs in **its own process**, because the failure mode is a segfault —
in-process it ends the whole session rather than one test case, and later tests
silently do not run. It asserts the loss is bit-identical **and** compares
gradients: a fix that "stops the crash but has backward read the wrong bytes"
passes a "does it run" check and fails here. It disables cuDNN auto-tuning so the
`3.8e-4` band cannot hide smaller effects.

(numerics-reduce)=
## CPU reduction block accumulation

**What you see.** On CPU, float32 `sum`/`mean` relative error **does not grow with
element count**, and is smaller than NumPy's:

| Summing n copies of `0.1` | CPU before | CPU now | CUDA | NumPy |
| --- | --- | --- | --- | --- |
| 65,536 | 6.17e-4 | **5.96e-7** | 1.49e-7 | 1.49e-7 |
| 1,048,576 | 9.86e-3 | **5.96e-7** | 1.49e-7 | 9.69e-7 |
| 16,777,216 | 1.53e-1 | **5.96e-7** | 4.47e-7 | 1.61e-5 |

**Why it deserves its own entry.** Before, there was **a single serial
accumulator**, with error proportional to element count: summing 16.77 million
copies of `0.1` was off by **15 %**, and the `mean` of 1,000,000 `0.1`s was
0.09975. This needs no special input — a loss that averages over the batch hits it.

**What to do.** Nothing. `KI-BACKEND-006` (commit `fc2bfe46f`) changed it to
"multiple partial sums within a block + pairwise folding between blocks, with the
folding stack spanning the whole reduction loop nest". This fix is **both more
accurate and faster**: the same kernel on 64M float32 went from 4.90 to
**18.38 GB/s** (3.7–4.9×), from "4.4× slower than NumPy" to "on par with NumPy".
The only measured cost is +18 % JIT **compile** time for large fused reduction
kernels (once per kernel shape, disk-cached). The CUDA side is bit-identical — the
pass returns immediately for accelerators.

**Known boundary, read it as stated**: shapes reducing along the **outermost
dimension** are outside the rewrite. `(262144, 64)` along dim 0: 2.47e-3 both
before and after. This is not a regression — **NumPy gives exactly the same
2.47e-3 on that shape**, since it is also one serial chain per column — but it
falls outside the premise that "the reduced dimension is the innermost one".

`max`/`min` reductions are also out of scope: their fold is strictly associative,
so reordering brings no accuracy gain. Their problems are different; see the
{ref}`section above <numerics-minmax>` and the open items below.

**Tests**: [`tests/ops/test_reduce_accuracy.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_reduce_accuracy.py),
9 cases, both CPU and CUDA sets. It asserts the **growth shape**, not a threshold:
the relative error at 16M must be within 10× that at 64K (a serial accumulator
gives 248× over this 256× range, blocking gives 1×), and within 10× NumPy's error
on the same input; there is also a three-dimensional nested shape specifically to
catch fixes that "only changed the innermost loop". Fixed thresholds go red and
green with the machine and compiler regardless of defects, so none were used.

(numerics-open)=
## What does not hold yet

Each of the following has been reproduced and none are fixed — not wrong answers,
but throughput or residuals that vary with settings. Hitting them during a
comparison is not your fault.

| Symptom | Scope | Item |
| --- | --- | --- |
| float32 `max`/`min` reductions are about 2× slower than before the NaN fix (throughput only; the answers are correct) | CPU | `KI-OPS-006` |
| With cuDNN auto-tuning off, gradients still show a `1.1e-6` residual that varies with `auto_flush_ops` | CUDA | `KI-EXEC-003` |

The complete entries — severity, evidence, owner and exit conditions — are in the
repository's
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md).

## Related

- [float32 accumulation precision](float32-precision-policy.en.md): the TF32 / bf16
  tiers, and when these settings are read under the lazy graph.
- [Pipelined lazy execution](pipelined-execution.en.md): what `auto_flush_ops` is
  for.
- [Devices and placement](device-placement.en.md): confirm both sides really are on
  the device you think before comparing.
