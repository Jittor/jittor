# ACL model diagnostics

These are opt-in maintainer probes, not pytest suites or hardware acceptance
results. The native GPT implementations exercise Jittor core; downstream
Transformers checkpoint validation belongs to
[Jittor Transformers](https://github.com/JittorEcosystem/jittor-transformers).
Historical observations and their limits are retained in
[the archived report](../../../docs/results/acl-manual-probes.en.md).

Run from an external state directory so weights and JSON never enter the source
tree. Select an allocated device in the environment before starting Python.
Use a separate Jittor cache for each backend, and compile the first run serially.

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
LAB_ROOT=${JITTOR_LAB_ROOT:-$(dirname "$REPO_ROOT")/jittor-lab}
mkdir -p "$LAB_ROOT/_state/acl-probes/run"
cd "$LAB_ROOT/_state/acl-probes/run"
export JITTOR_HOME="$PWD/cache"
export PYTHONPATH="$REPO_ROOT/python"

# Jittor interpreter; the generator builds a native model to enumerate shapes.
python "$REPO_ROOT/tools/probes/acl/xcheck/gen_weights.py"
python "$REPO_ROOT/tools/probes/acl/xcheck/run_jt.py" acl
# In a separate real-PyTorch environment, sharing this working directory:
<torch-python> "$REPO_ROOT/tools/probes/acl/xcheck/run_torch.py" --device cpu
python "$REPO_ROOT/tools/probes/acl/xcheck/compare.py" \
  xcheck_torch.json xcheck_jt_acl.json
```

`compare.py` checks loss and per-parameter **gradient norms** with explicit
absolute/relative tolerances. It fails on missing gradients, nonfinite values,
or excessive error. Equal norms do not establish elementwise gradient parity;
this cross-device diagnostic is not a same-device oracle acceptance test.
`tests/tools/test_acl_probe_comparison.py` checks the comparison contract without
requiring a device. Use the maintained backend and ecosystem gates for full
forward/backward acceptance.

For the one-card full-parameter Adam capacity probe:

```bash
python "$REPO_ROOT/tools/probes/acl/max_model_probe.py" 2 4 128 1 64 float32
```

Arguments are layers, heads, width, batch size, sequence length, and dtype.
Increase them deliberately; a failed allocation or backend operation exits with
an error. This does not measure LoRA, multi-card capacity, or throughput scaling.
No current hardware support claim follows from relocating these scripts.
