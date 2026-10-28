# Cost attribution for `import jittor` on a hot cache

- Status: the attribution is accepted. The hot-cache gap it pointed to has been closed by
  build stamps for the core and the bundled libraries (hot-cache import is now under 1 s in
  both configurations); **on a cold cache and when switching configurations, import still
  compiles the whole core**, and that half remains open
- Date: 2026-09-04
- Baseline commit: `bf702127` (attribution); the numbers after the build stamps landed are
  taken from `d23f9bba`/`d41106d5a`
- Validation scope: an x86_64 + CUDA 12 development machine (sm_89), a hot cache under the
  same `JITTOR_HOME`, both the CPU-only (`nvcc_path=""`) and CUDA configurations; everything
  is `perf_counter` wall-clock time, no profiler. Not covered: ACL/ROCm builds, the
  breakdown of the cold cache
- Maintainers: build system maintainers
- Review triggers: changes to the module body of `python/jittor/build/compiler.py`, to the
  setup chain of `compile_extern.py`, to the cache key of `cache_compile`, to the
  parallelism strategy of `run_cmds`, or a significant change in the number of core
  translation units

## Conclusion

**The largest single item in a hot-cache import is neither probing nor dlopen, but the fixed
cost of the "core compilation" step when there is nothing to do**: 0.906 s of the 1.325 s in
the CPU-only configuration (68%), and, in the 2.457 s CUDA configuration, 1.636 s (67%)
split between the no-op validation of the bundled CUDA libraries and `cupy`. On a hot cache
the net effect of these steps is zero — the generated headers are byte-identical and every
compile command is judged up to date — yet the cost is paid on every import.

## Measurement method

Three complementary layers, all measured as `perf_counter` wall-clock time in a subprocess
(cProfile adds 40% on this path and changes the ordering):

1. Module bodies: `-X importtime`, accurate but with a module as the granularity;
2. Build fan-out: replace `jittor_utils.run_cmds` **before** `import jittor`, and record the
   command count and elapsed time of each fan-out (what needs measuring is inside the
   compiler module body, which has already run by the time the import returns);
3. Generators: call `gen_jit_flags` / `pyjt_compiler.compile` again after the import; they
   are idempotent, so the second call's time is the price of the first.

## Attribution table (CPU-only, 1.325 s)

| Item | Time | Share | What it actually does on a hot cache |
| --- | ---: | ---: | --- |
| `run_cmds` "Compiling jittor_core", 176 entries | 0.542 s | 41% | Starts a 16-process pool, reads each source file and all dependency headers, computes content hashes, compares `.key`, and returns on a match; the products are unchanged |
| `gen_jit_flags()` | 0.212 s | 16% | Globs 176 `.cc` files, strips comments character by character in pure Python, regex-extracts `DEFINE_FLAG`, and writes a byte-identical `gen/jit_flags.h` |
| `gen_pyjt` | 0.104 s | 8% | Scans 177 headers/sources and rewrites `gen/*.cc` |
| `run_cmds` "Compiling jittor_mpi_core", 7 entries | 0.044 s | 3% | The same no-op validation |
| `jittor.compat.triton` module body | 0.091 s | 7% | Imports `triton.runtime` |
| Everything else (numpy, dlopen, the various module bodies, `probe.json`) | ≈0.33 s | 25% | A single `probe.json` read takes <0.2 ms; probing is no longer a hot-path cost |

The CUDA configuration adds: 0.351 s of no-op validation over 49 commands for the bundled
CUDA libraries (cub/cublas/cudnn/curand/cufft/cusparse and `libcuda_extern`), 0.369 s for an
unconditional `import cupy` (of which 0.199 s is scanning 289 dist-info directories), and
0.204 s for the larger CUDA core dlopen.

Another commonly misread number: "40 s cold compile" does not only happen on an empty
cache. Within the same `JITTOR_HOME`, switching from the CUDA configuration to the CPU-only
configuration gives two different configuration fingerprints, each needing a full core — so
switching back and forth between gates pays it every time.

## What follows from this, and has landed

1. **Core compilation needs a "already up to date" fast path.** The core and the bundled
   libraries now both write build stamps (the `product_build_is_current` family in
   `python/jittor/build/compilation.py`): the stamp records the hashes of all source files,
   paired headers and the compiler module that generates the sources, and a hit skips the
   whole step. After it landed, hot-cache import went 1.28 → 0.80 s in the CUDA
   configuration and 0.52 → 0.38 s in the CPU-only configuration, and the compile fan-out of
   a hot import dropped from 60 entries to 0. On `e3c369acb`, a comparable machine (RTX
   4090, CUDA configuration) re-measured three times at 0.86–0.89 s.
2. **Third-party dependencies are no longer imported unconditionally.** `cupy` is imported
   lazily only on the first numpy→cupy conversion (`python/jittor/build/init_cupy.py`), and
   a plain `import jittor` no longer imports modules under `jittor.compat` either.
3. **`JITTOR_NO_BUILD=1`** turns "this import must not compile" into something assertable:
   any step that would compile raises instead of quietly compiling, which is how we ensure
   that what is being measured really is the hot cache.

The cost of a wrong judgement is asymmetric: a stamp that calls "needs rebuild" "up to date"
silently computes the wrong thing, while calling it "stale" only wastes time, so stamps err
on the side of recording more. Changing the compiler module (even just adding a comment)
invalidates the core build stamp; running `python -m jittor_utils.bootstrap` once fixes it,
and that is not a bug.

## Still open

- On a cold cache and when switching configurations, `import jittor` still compiles the
  whole core; moving core compilation to an explicit bootstrap or to the first operator call
  is not yet done;
- The no-op validation cost is proportional to the number of core translation units (about
  6–8 ms per entry wall-clock, 16-way parallel), so a change that adds TUs should report
  "+N TUs", not just "+0.1 s".

## Reproduction

```bash
JITTOR_HOME=<isolated JITTOR_HOME> TMPDIR=<isolated TMPDIR> PYTHONPATH=<worktree>/python \
EXPECT_JITTOR_SRC=<worktree>/python nvcc_path=<path to nvcc> \
python agent/skills/jittor-build-change-verification/measure_import_cost.py --json before.json
```

Setting `nvcc_path=""` switches to the CPU-only configuration (the first run pays that
configuration's cold compile once). Do not clear the cache between the two runs; run it
three times in a row and take the last two. For a reproducible way to create a cold cache and
for the design rules of build stamps, see §2.5–§2.7 of the same skill.
