# JIT cache and test duration diagnosis

[Section contents](../index.en.md) · [中文](jit-cache.md)

Use this page when the same test command suddenly takes longer. First choose the scope with
the [test system](test-system.en.md), and distinguish core bootstrap, external-library builds
and test execution. Establish cache hits before deciding that a test is too expensive.

## Diagnosis order

1. Record the commit, interpreter, compiler, backend selection, compile options, `JITTOR_HOME`,
   `cache_name` and actual `cache_path`. The same machine does not imply the same cache.
2. Locate the slow phase: core import, library loading, kernel compilation or computation.
   Keep compiler logs and child-process information; waiting on a build lock is not proof of a hang.
3. If needed, repeat the original command serially under the same configuration. Compare cold
   and warm timings as well as collected/executed, passed, skipped and failed nodes so that
   the amount of work is unchanged.
4. Only after establishing a warm cache should `--durations` guide test optimization or tier changes.
5. Use `-v` for long runs and check the [completion sentinel](test-isolation.en.md).
   A partial log after a timeout is not passing evidence.

Keep state and raw logs under `$JITTOR_LAB_ROOT/_state/<topic>/<run>/`, outside the source tree.
Tests and benchmarks use separate caches. Serialize the first build of a new JIT operation or
extension; independent concurrent tasks use distinct `JITTOR_HOME` or `cache_name` values.

## Cold-time attribution and warmup scope

The first case to reach a kernel or library pays its compilation cost. A cold duration ranking
therefore depends on collection order. It neither directly measures each test's compute cost
nor proves that a directory contains no expensive tests.

A fused kernel key depends on the actual subgraph: operator order, dtype, rank, strides and
broadcast forms. Warming common operators cannot replace the workload's fused graphs; library
`code` kernels and single-operator kernels may be reusable.

One historical `tests/nn` run produced 604 JIT kernels: 436 fused kernels (72%), 111 library
kernels and 57 single-operator kernels. The same suite took 37:03 cold and 1:02 warm, with
360 passed, 71 skipped and 948 subtests in both runs. These numbers illustrate attribution;
**they are not a current-commit performance baseline or a universal warmup ratio**.

## Cache identity: compiler probes must be locale-independent

The `archXXXXXXXXXX` component comes from the instruction set expanded by
`cc -march=native -Q --help=target`. gettext can translate its enabled/disabled states;
hashing that localized text creates different cache trees on the same machine.

The maintained constraints are:

- Run the probe in the C locale through `jittor_utils.c_locale_environment()`; callers need
  not standardize their language settings themselves.
- Use the memo slot `target_arch:c-locale:<cc>` instead of reusing translated text under
  the old `target_arch:<cc>` slot.
- Account for existing `probe.json` entries when changing a probe. Changing only the new
  computation does not invalidate an old memo.
- Other key inputs should be stable version numbers, architectures and flags, not localized prose.

The targeted regression is `test_the_instruction_set_key_does_not_depend_on_the_message_locale`
in `tests/build/test_build_config_cache.py`. Compare actual `arch*` paths and probe slots;
deleting every cache hides an identity problem rather than diagnosing it.

## Time external-library builds separately

`mkl` (oneDNN v3) loads lazily. Importing Jittor or running a simple addition does not establish
that the library needed for CPU matmul/convolution has been built. A source build can include
about 1200 translation units.

oneDNN build parallelism is constrained by CPU count and memory, with a default cap of 32;
`JT_ONEDNN_BUILD_JOBS` overrides it explicitly. Do not copy a large host's concurrency setting
onto a small runner. In an older isolated same-host A/B, two builds at parallelism 4 took
443/448 seconds, versus 87/87 seconds at 32. These observations do not promise a new host's time.

For library-load diagnosis, warm and time libraries separately with `probe_library(..., load=True)`.
This makes the work attributable and subject to a timeout; it does not reduce total work.
**Do not include MPI/NCCL/HCCL in generic warmup without the other ranks**: their loaders may
initialize communicators and wait for peers.

## Disk hits and the build lock

`Op::jit_run` checks the in-process `jit_ops` map first, entering `OpCompiler::do_compile` on a miss.
Even with a warm disk cache, a new process passes through the latter on its first use of a key.

A disk hit follows an unlocked path: compute the key from the current command and inputs,
compare it with the artifact's `<product>.key`, then `dlopen` on a match. Actual builds lock
inside `jit_compiler::compile()`. Generated source is an in-memory input; the hit path hashes
its string directly and must not write a source file just to check the key.

This fast path depends on publication order: `cache_compile()` atomically renames the complete
product into place before atomically publishing `.key`, both in the same directory. A matching
new key must never accompany an old or partial product. `tests/build/test_jit_cache_concurrency.py`
samples a slow rebuild externally and checks overlapping multi-process kernel sets for publication
order and numerical agreement. Such targeted stress tests do not authorize routine cold-cache contention.

Finish core import and first library loads such as `cutt` before measuring lock behaviour:
these may take a different lock path. Otherwise a slow "hit while locked" may actually measure
bootstrap or library construction. Python-side `lock_scope` around `compile_custom_ops`,
`compile_module` and `download_url_to_local` may still cover the whole call, including hits;
analyze these separately from the per-JIT-key fast path.

## Limits of parallel cache reuse

A warmed, matching configuration can be shared by runner-managed workers; actual compilation
still contends on the global lock. `use_parallel_op_compiler` controls in-process parallel
compilation. Adding pytest processes does not remove that lock. Independent tasks still isolate
caches under the repository collaboration rules to avoid mixing source, environment and build lifetimes.

Do not use hard-linked copies of warm caches as the default worker initialization scheme.
A new home's `jit_utils` bootstrap may rebuild and request a restart (historically exit code 3),
and can invalidate some seeded kernels. Prefer runner-managed caches with explicit source and
configuration identity.

This page extracts the mechanisms from the original cache investigation. Full historical
measurements and context remain in the
[historical cache chapter](https://github.com/Jittor/jittor/blob/a3783a04a0fdf921abfe1e3c0d7c93256ee70bd3/docs/development/test-system.en.md).
Record new measurements in GitHub issues/PRs. Changes to cache keys, locks or publication order
require fresh validation; historical speed conclusions do not carry forward automatically.
