# Ascend 910B setup and validation

This guide configures a Jittor 2.0 source checkout for Ascend 910B-series
devices and verifies that an operation really executes through ACL. Installing
the Ascend driver, firmware, and CANN itself is outside Jittor's scope; start
from a host where the vendor runtime and compiler are already installed.

## Validated baseline

The maintained gate was validated on this combination:

| Component | Validated version |
| --- | --- |
| Device | Ascend 910B3 |
| Architecture | Linux aarch64 |
| Driver reported by `npu-smi` | 25.5.1 |
| CANN toolkit and `ccec` | 9.0.0 |
| Python | 3.9.25 |
| NumPy | 1.26.4 |
| pytest / pytest-timeout | 7.4.4 / 2.3.1 |

These versions are a reproduced baseline, not a claim that every other CANN
release is incompatible. Jittor declares Python 3.7 through 3.13 support;
Python 3.9 through 3.11 is the practical range for an Ascend environment whose
vendor packages may have narrower Python constraints.

## Check the device and toolkit

Confirm that the driver sees a healthy device before importing Jittor:

```bash
npu-smi info
```

Find the CANN environment script and load it in the current shell. Keep its
location configurable instead of embedding a host-specific installation path:

```bash
export ASCEND_HOME=/path/to/Ascend/cann-9.0.0
export CANN_SET_ENV="$ASCEND_HOME/set_env.sh"

# Some CANN set_env.sh releases expect these variables to exist under `set -u`.
: "${LD_LIBRARY_PATH:=}"
: "${CMAKE_PREFIX_PATH:=}"
source "$CANN_SET_ENV"

ccec --version
```

`npu-smi info` must succeed in the same execution environment as Python. In a
container, pass the required Ascend devices and driver libraries through using
the vendor container runtime instructions.

## Install the source checkout

Create a dedicated Python environment and install the checkout in editable
mode. The NPU validation gate also needs SciPy and the pinned pytest tools:

```bash
python -m pip install -e .
python -m pip install \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "pytest==7.4.4" \
  "pytest-timeout==2.3.1"
```

Run from the repository root. For development without an editable install,
place the checkout's Python package first:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

## Isolate JIT state

Jittor compiles its core and operators on first use. Put mutable state outside
the checkout and give every concurrent run a distinct `JITTOR_HOME` or
`cache_name`:

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
run_root="$JITTOR_LAB_ROOT/_state/ascend-910b/manual"
mkdir -p "$run_root"/{home,jittor-home,tmp,xdg-cache}

export HOME="$run_root/home"
export JITTOR_HOME="$run_root/jittor-home"
export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/xdg-cache"
export cache_name=ascend_910b_manual
```

Run the first JIT or extension compile serially. Tests and benchmarks must not
share a compilation cache. After changing CANN, the driver, the host compiler,
or ACL source, use a new cache name; do not diagnose the new stack with stale
binaries.

## Select a device

Restrict the process to allocated devices before importing Jittor:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
```

The visible device is renumbered inside the process. Do not select a device
that is already occupied by another workload.

## Run a real ACL probe

An import-only check cannot prove NPU support. The following probe requires ACL,
enables the accelerator flags, performs float32 matrix multiplication, and
checks both its independent result and the ACL compilation log:

```python
import numpy as np
import jittor as jt

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1

a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
b_np = np.arange(20, dtype=np.float32).reshape(4, 5)

with jt.log_capture_scope(log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
    actual = jt.matmul(jt.array(a_np), jt.array(b_np)).numpy()

np.testing.assert_allclose(actual, a_np @ b_np, rtol=1e-5, atol=1e-5)
messages = [entry["msg"].lower() for entry in logs]
assert any("compile acl op" in message for message in messages)
assert not any("fallback cpu" in message for message in messages)
print("ACL matmul passed")
```

Save the probe as `probe_acl.py` outside the checkout and run it only after
sourcing CANN:

```bash
python probe_acl.py
```

Jittor uses `use_cuda` as the common accelerator execution flag even when the
selected backend is ACL. `has_acl` is therefore the required discriminator
between Ascend and CUDA.

## Run the maintained NPU gate

The NPU nox session creates isolated state, checks `npu-smi`, runs a real ACL
matmul probe, and then executes the maintained backend and OpInfo tests. Nox
itself uses Python 3.11, while `JITTOR_CI_PYTHON` points to the pre-provisioned
Ascend Python environment:

```bash
python -m pip install -r requirements/dev-tools.txt

export CANN_SET_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export JITTOR_CI_PYTHON=/path/to/ascend-python/bin/python
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
python -m nox -s npu
```

The session resolves `python_config_path` from `JITTOR_CI_PYTHON` rather than
from Nox's own interpreter. If a direct launch sets `python_config_path`, it
must point to the config helper for that same Python version; a mismatched
helper produces an extension suffix that the hardware Python cannot import.

To run the same core tests directly:

```bash
export JITTOR_TEST_DEVICES=npu
"$JITTOR_CI_PYTHON" -m pytest -v --timeout=600 \
  tests/backends/npu/test_acl.py \
  tests/backends/npu/test_aclop.py \
  tests/backends/npu/test_acl_indexing.py \
  tests/ops/test_ops.py
```

Do not treat a passing CPU fallback as NPU coverage. For a new or repaired
operation, add a focused assertion that captures ACL execution or otherwise
proves the declared device performed the calculation.

## Run Qwen3-8B with Transformers

The maintained manual probe runs a local Qwen3 checkpoint through the Jittor
Torch shim and Transformers 4.56.2. Install the optional model dependencies and
keep the checkpoint outside the source tree:

```bash
python -m pip install "transformers==4.56.2" "jinja2==3.1.6"
export QWEN3_MODEL=/path/to/Qwen3-8B
export JITTOR_TORCH_SHIM=1

python tests/backends/npu/manual/run_qwen3_transformers.py \
  --model "$QWEN3_MODEL" \
  --dtype bfloat16 \
  --max-new-tokens 8 \
  --runs 3
```

Run this command only after the CANN, device-selection, and isolated-cache setup
above. The probe loads weights on CPU, explicitly migrates the model to the
visible NPU, prints `npu-smi` while the model is resident, performs greedy eager
attention generation with KV cache, and fails if generation logs an ACL backend
fallback or an unexpected CPU-compiled operation. CPU checkpoint
deserialization is expected and is not used as evidence for the model forward.

The validated Qwen3-8B checkpoint has 8,190,735,360 parameters. Float32 uses
32,376 MB of device memory on one 64 GB 910B3. Both float32 and bfloat16 report
accelerator-resident parameters, `has_acl=use_acl=use_cuda=1`, zero fallback,
and zero CPU-compiled operations. The maintained bfloat16 eight-token request
stops after `[19, 13, 151645]` and decodes to `4.` in every repeated run. Use
`--dtype float32` for the original one-token probe. These are correctness probes,
not throughput benchmarks.

## Current limitations

The maintained 910B gate deliberately skips these reproduced gaps instead of
allowing them to abort or stall the process:

- sub-32-bit integer `sum`, `prod`, `max`, and `min`, plus boolean `all` and
  `any`, lack a complete ACL reduction path; promote inputs to a supported
  width when possible;
- composed float32 `atan2` can raise a vector-core exception;
- complex `irfft` can stall;
- float32 `prod` can abort in the ACL reduction path;
- native FlashAttention tests require an optional `jt.nn.FlashAttention`
  implementation and are skipped when it is absent;
- Qwen3 bfloat16 is verified for eager, no-grad greedy inference. Fused ACL SDPA
  is additionally verified for Qwen3-0.6B with FlashAttentionScoreV2 prefill and
  IncreFlashAttentionV4 decode. Qwen3-0.6B float32 eager forward, causal-LM loss,
  and backward are also verified with zero fallback; the optimized RoPE result
  currently requires an external Transformers module patch to route Qwen3 to
  `jt.nn.rotary_emb`. Optimizer update, BF16 training, Qwen3-8B BF16 SDPA and
  training, sampling, quantization, and other model families remain separate
  capability gates. See the
  [Qwen3 forward/backward report](../../agent/results/transformers/2026-08-30-qwen3-ascend-training.md);
- ACL does not provide general float64 operator coverage, so float64 fallback
  is not accepted as evidence for an NPU operation.

Float16/float32 `arg_reduce` forward and value-output backward are maintained
ACL capabilities. Forward uses CANN MaxDim/MinDim and backward scatters the
upstream gradient to the selected first index; the real-device regression rejects
CPU compilation and fallback. See the
[focused verification report](../../agent/results/2026-08-30-npu-arg-reduce-backward.md).

See the [active known-issues ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)
for executable evidence and exit conditions.

## Troubleshooting

If ACL is not detected, verify `CANN_SET_ENV`, `ccec --version`, the Python
architecture, and the CANN library paths in the same shell. Do not catch and
ignore CANN registration or compilation errors; an incomplete backend setup
must fail visibly.

If an asynchronous device error is reported after the Python operation that
caused it, rerun only the smallest reproducer with:

```bash
export JT_SYNC=1
export trace_py_var=3
```

These flags are diagnostic and slow execution. Remove them after locating the
failing operation. For general JIT and memory diagnostics, see
{doc}`debugging`.
