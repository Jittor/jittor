# Iluvatar Corex: Discovery and Follow-up Verification

This page records the **discovery contract** for the Corex backend. The current code can
inspect a Corex installation without importing the compiler or modifying Jittor global
state. Full compilation and performance verification require a supported Corex device and
belong to a separate gate.

## Configure the install path

Point `COREX_HOME` at the vendor SDK root (defaults to `/usr/local/corex` when unset). The
discovery code only looks for `$COREX_HOME/bin/clang++` and **does not create directories,
load libraries, or modify compiler options**.

The provider lives in the backend package's `__init__` (`backends/corex/` in a checkout,
`jittor.backends.corex` when installed). Discovery itself does not depend on the Jittor
runtime, so in a checkout you can load that module directly without even compiling the core:

```bash
export COREX_HOME=/opt/corex
PYTHONPATH="$PWD/python:$PWD/backends" python - <<'PY'
from corex import discover
result = discover()
print(result)
raise SystemExit(0 if result.available else 1)
PY
```

In an installed environment the equivalent is `from jittor.backends.corex import discover`
-- that path imports Jittor itself first.

For offline contract tests, create a temporary directory containing only `bin/clang++` (a
fake executable is enough). The returned result object reports the resolved home, compiler
path, availability, and a stable reason string.

## Handoff to hardware

**The repository currently has no trustworthy Corex hardware baseline.** On a supported
Iluvatar machine, first record the exact accelerator model, driver, SDK, and clang versions,
then run Jittor. Use a separate `JITTOR_HOME` and `TMPDIR`, then run a device-visibility
probe and a fused kernel. The result must show the Corex compiler path and **no CPU
fallback**.

```bash
export COREX_HOME=/path/to/corex
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/corex/jittor-home"
export TMPDIR="$JITTOR_LAB_ROOT/_state/corex/tmp"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
export CUDA_VISIBLE_DEVICES=<分配到的设备>
PYTHONPATH="$PWD/python" python -m pytest -q \
  tests/backends/corex/test_corex_discovery.py
```

This command **only covers the discovery contract**. A future hardware gate must also run
the corresponding CUDA-compatible numerical probes and record the vendor's device and
compiler versions. **Do not treat local offline results as Corex hardware verification.**
