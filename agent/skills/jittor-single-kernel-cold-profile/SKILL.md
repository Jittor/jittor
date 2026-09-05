---
name: jittor-single-kernel-cold-profile
description: Measure one generated C++ kernel cold-build cost by phase without changing Jittor compiler behavior.
---

# Single-kernel cold-build profiling

Use `profile_single_kernel.py` when a compiler or code-generation change needs a
small, reproducible cost breakdown. The report separates source generation,
Jittor `cache_compile`, direct compiler (`clang++`/`g++`), shared-library link,
and dynamic load. It is a measurement tool only; it does not change Jittor's
optimization or cache implementation.

```bash
PYTHONPATH=$PWD/python python \
  agent/skills/jittor-single-kernel-cold-profile/profile_single_kernel.py \
  --json /tmp/single-kernel.json
```

Use `--compiler clang++` to make the compiler phase explicit. Give the script a
dedicated `--work-dir` when retaining generated files for inspection. The
default temporary directory is removed after the report is written.

The JSON `phases` object is the source of truth. `cache_compile.available` says
whether the native `jit_utils_core` binding was available; when it is absent,
the script still records the equivalent command through the Python fallback and
marks that fact instead of presenting it as native cache timing.
