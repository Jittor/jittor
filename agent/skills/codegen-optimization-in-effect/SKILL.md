---
name: codegen-optimization-in-effect
description: Prove an instruction-level codegen optimization is actually emitted and actually pays, before keeping, porting, or deleting it. Use for non-temporal stores, prefetch, FMA, alignment and vectorization passes, and for A/B timing a compiler-pipeline change on a loaded machine.
---

# Is the optimization in effect, and is it worth anything?

Two questions that get confused, and a third that decides what to do:

1. **Is it emitted?** Read it back out of the binary. Nothing else counts.
2. **Does the emitted form pay?** Benchmark that exact form, not the idea.
3. **What is the ceiling?** Benchmark the best possible form too, so a "no gain"
   verdict cannot be blamed on the measurement.

Most of the wasted effort in this area comes from answering 1 with a correctness
test, or answering 2 without 3.

## 1. "The result is numerically correct" is not evidence

A non-temporal store, a prefetch, an FMA contraction and an aligned load all
compute the same numbers as the code they replace. That is the whole point of
them. So a test that runs the kernel and checks the output can pass with the
optimization fully disabled, and it will read like proof that it works.

Read the instruction back instead:

```bash
objdump -d <built>.so | grep movnt      # or vfmadd, prefetch, vmovap...
```

Do this **before and after** the change, and record both counts. One count on
its own tells you nothing, because you do not know what the baseline was.

### The check must be shown to have teeth

A grep that finds nothing looks identical whether the optimization is absent or
the binary failed to build. Both of those happened in one afternoon:

- A build with `clang++ ... -fopenmp` failed with `cannot find -lomp`, and the
  script cheerfully reported `movnt = 0` — exactly the answer being hoped for.
  **Make the script refuse to report a count unless the artifact was built.**
- A test asserted the intrinsic appeared in the generated source, but guarded it
  with `if cc_type == "clang": ... else: ...`. The machine's default is g++, so
  the assertion that ran was the `else` branch, which asserts the pass did
  *nothing*. It had never checked a single instruction.

Prove the check can fail. Take the real artifact, inject the construct by hand,
and confirm the same command finds it:

```bash
sed -i 's|out\[i\] = \(.*\);|__builtin_nontemporal_store((\1), \&(out[i]));|' mutant.cc
clang++ $SAME_FLAGS mutant.cc -o mutant.so || { echo "BUILD FAILED, no conclusion"; exit 1; }
objdump -d mutant.so | grep -c movnt     # must be > 0, or the check is blind
```

## 2. Benchmark the form the compiler can actually emit

A source-level pass that rewrites one *statement* can only produce a
per-element intrinsic. Whether that survives into good code is a property of the
vectorizer, not of the idea. Measured on one AMD EPYC core, writing 256 MiB that
is never read back:

| store form | GB/s |
| --- | --- |
| ordinary store | 15.0 |
| per-element non-temporal, clang `__builtin_nontemporal_store` | 14.6 |
| per-element non-temporal, g++ `_mm_stream_si32` | 9.1 |
| whole-loop `_mm256_stream_ps` | 25.3 |

The idea is worth +69%. The form a statement-rewriting pass can express is worth
0% on one compiler and −40% on the other, because neither vectorizer will widen
a per-element non-temporal store back into a vector one. Without the last row
the honest reading would have been "non-temporal stores do not help here", which
is wrong; without the middle rows the reading would have been "keep the pass",
which is also wrong.

Ask the compiler directly whether the loop vectorized rather than inferring it:

```bash
clang++ -Ofast -march=native -c x.cc -Rpass=loop-vectorize -o /dev/null
```

Use **the flags the kernel actually ships with**, not the loudest ones. Jittor's
CPU kernels build at `-O3 -march=native` since KI-BACKEND-005, not `-Ofast`;
asking at `-Ofast` answers a question about a build nobody runs, and for
reductions the two answers differ — `-ffast-math` licenses the reassociation
that makes a reduction vectorizable at all.

Give every variant the same `__restrict__`, the same alignment and the same
buffers. A missing `__restrict__` on one variant silently makes the comparison
about aliasing instead of about the store.

## 3. A/B timing a compiler-pipeline change on a busy machine

To measure what a wrapper, an extra pass, or a second compiler invocation costs,
do not time the whole test suite. Take the exact command line out of the build
log, repoint its source, output and depfile at private paths, and run the two
forms directly. Every repetition then compiles for real, with no cache involved
and no need to wipe `$JITTOR_HOME`.

Two things decide whether the numbers mean anything:

- **Warm up, untimed, once per variant.** Faulting in the compiler binary and
  the headers makes the first sample more than twice as fast as the rest. With
  the warm-up the spread across three independent runs was 0.808–0.853s against
  1.013–1.055s; without it the same data gave "9.3% / 21.3% / 33.2%" depending
  on which statistic was quoted, and looked like load noise rather than an
  ordering effect.
- **Alternate which variant runs first.** On a shared machine, background load
  drifts over the length of a run and whichever variant goes first in each pair
  wins systematically.

Report an interval from at least three separate runs, not a single number, and
say which statistic it is. Compare min-to-min when the machine is contended: the
least-contended sample of each variant is the closest either gets to its true
cost.

## 4. When the verdict is "delete it"

A measured refutation is a real deliverable, but it needs the same rigour as a
fix, and it needs a test:

- Assert the thing is gone in a way that **failed before the change** — that the
  pass sources are absent, that the tuner no longer offers the candidate. Run
  those assertions against the pre-change files (`git show HEAD:<path>` into a
  scratch tree) and show them red.
- Keep a standing guard that would go red if it came back: disassemble the
  kernel that actually got built and assert the instruction is absent, with the
  measured numbers in the failure message so the next person does not have to
  rediscover them.
- Put the numbers in the audit entry and say plainly which documented claim they
  contradict. Here the audit said the chain cost "2–3x"; it measured 1.25x, and
  the task's "≥50% reduction" acceptance had been derived from that wrong
  multiple.

## 5. Two traps specific to finding the built kernel

- **Cached kernels are not regenerated.** A test that watches the log for
  "Generate" checks nothing from its second run onward. Watch for the library
  being opened instead — but note that a given jit key is opened only once per
  process, so give each test its own `compile_options` tag, and fail loudly on
  an empty list rather than passing vacuously.
- **Kernel file names are truncated** to fit the filesystem limit, so the
  interesting part of a fused op's name may simply not be there. Select the
  kernel by what its source defines (`#define op2_OP subtract`), not by name.
