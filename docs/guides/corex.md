# Corex discovery and later validation

This guide records the discovery contract for the Corex backend. The current
checkout can inspect an installation without importing the compiler or changing
Jittor globals. Full compilation and performance validation require a supported
Corex device and remain a separate gate.

## Configure the installation

Set `COREX_HOME` to the vendor SDK root. The discovery code looks for
`$COREX_HOME/bin/clang++`; it does not create directories, load libraries, or
change compiler flags.

```bash
export COREX_HOME=/opt/corex
PYTHONPATH="$PWD/python" python - <<'PY'
from jittor.extern.corex.corex_compiler import discover
result = discover()
print(result)
raise SystemExit(0 if result.available else 1)
PY
```

For an offline contract test, create a temporary directory containing only
`bin/clang++` (a fake executable is sufficient). The result object reports the
resolved home, compiler path, availability, and a stable reason string.

## Hardware handoff

The repository currently has no trusted Corex hardware baseline. On a supported
Iluvatar/Corex machine, record the exact accelerator model, driver, SDK, and
clang versions before running Jittor. Use distinct `JITTOR_HOME` and `TMPDIR`
values, then run the device-visible probe and a fused kernel. The result must
show the Corex compiler path and no CPU fallback.

```bash
export COREX_HOME=/path/to/corex
export JITTOR_HOME=/tmp/jittor-corex-home
export TMPDIR=/tmp/jittor-corex-tmp
export CUDA_VISIBLE_DEVICES=<allocated-device>
PYTHONPATH="$PWD/python" python -m pytest -q \
  tests/backends/corex/test_corex_discovery.py
```

That command is only the discovery contract. The future hardware gate must
also run the relevant CUDA-compatible numerical probe and capture the vendor
device/compiler versions. Do not treat this host's offline result as Corex
hardware validation.
