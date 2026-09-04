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

## Check per-operator ACL synchronization

The ACL binary operator family uses the shared launcher contract for workspace
allocation, execution errors, and synchronization. This source-only change
still requires the 910B3 probe below; hosts without CANN and an Ascend device
must not report hardware validation.

The ternary SWhere family uses the same contract and intentionally retains its
historical asynchronous execution policy.

The single-step ACL reduce owners (ReduceSum, Mean, Amax, and Amin) also use
the shared launcher and retain synchronous execution. Product reductions with
multiple axes remain on their dedicated intermediate-buffer path.

Run this diagnostic on an Ascend 910B3 only after sourcing the CANN environment,
confirming the device is healthy, and selecting an allocated device:

The unary runner is the first family migrated to the shared launcher tail. It
keeps its historical asynchronous policy, while workspace allocation and ACL
launch failures now use the same auditable error path as the base runner. When
validating this migration, include one unary operation and confirm that its log
contains neither `fallback cpu` nor `execute launcher failed`.

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>

sync_run=1 python -m pytest -q -s \
  tests/backends/npu/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-sync-run.log"
```

`sync_run=1` makes every `BaseOpRunner` wait for `aclstream` immediately after
launch. If synchronization fails, Jittor raises an error containing the
operator name, numeric return code, and decoded ACL error. Keep the complete
log and extract the attribution line with:

```bash
rg "aclrtSynchronizeStream failed" "$TMPDIR/acl-sync-run.log"
```

The ACL Cumsum family uses the shared launcher and retains synchronous
execution. This remains source-only until the Ascend 910B3 probe is run.

The run is valid only when the ACL execution assertion passes and the log has no
CPU fallback. Treat either spelling below as a failed NPU verification:

```bash
if rg -i "fallback cpu|cpu fallback" "$TMPDIR/acl-sync-run.log"; then
  exit 1
fi
```

After diagnosis, repeat the same focused node with per-operator synchronization
disabled to verify the normal asynchronous path still launches on ACL:

```bash
sync_run=0 python -m pytest -q -s \
  tests/backends/npu/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-async-run.log"

if rg -i "fallback cpu|cpu fallback" "$TMPDIR/acl-async-run.log"; then
  exit 1
fi
```

`JT_SYNC=1` is the separate executor-wide compile-time diagnostic. It is not a
replacement for the `BaseOpRunner` `sync_run=1` check above and may rebuild the
JIT cache when toggled.

## Verify ACL tensor and workspace ownership

The ACL runner creates tensor descriptors explicitly and obtains its shared
aclnn workspace from Jittor's temporary allocator. On Ascend 910B3, source CANN,
check the device, and record memory before running a process that grows the
workspace through several normal matrix multiplications:

```bash
source "$CANN_SET_ENV"
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
npu-smi info | tee "$TMPDIR/before-workspace.txt"

sync_run=1 python - <<'PY' 2>&1 | tee "$TMPDIR/workspace-normal.log"
import numpy as np
import jittor as jt

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1
for width in (64, 128, 256):
    value = np.arange(width * width, dtype=np.float32).reshape(width, width)
    actual = jt.matmul(jt.array(value), jt.array(value)).numpy()
    np.testing.assert_allclose(actual, value @ value, rtol=2e-4, atol=2e-2)
print("ACL workspace normal path passed")
PY

if rg -i "fallback cpu|cpu fallback" "$TMPDIR/workspace-normal.log"; then
  exit 1
fi

npu-smi info | tee "$TMPDIR/after-workspace.txt"
```

The normal run is accepted only with correct values and no CPU fallback. After
the Python process exit, it must no longer appear in `npu-smi`; compare
`before-workspace.txt` and `after-workspace.txt` to confirm its workspace was
released rather than retained by an orphan process.

Do not manufacture an unbounded allocation on a shared NPU. When an existing
workload naturally reproduces a workspace failure, preserve its log and extract
the attribution with:

```bash
rg "ACL workspace allocation failed" "$TMPDIR/workspace-failure.log"
```

The error must report `workspace requested bytes`, the `workspace allocator`,
and the underlying allocation failure. A later small probe in a fresh process
must still pass without CPU fallback; otherwise the failed allocation did not
leave the global workspace in a retryable empty state.

Three runner failures now identify the stage and operator instead of continuing
with an invalid executor or checking the outer fused graph by mistake:

- `aclnn workspace-size query failed` includes the operator name, return code,
  decoded ACL status, and CANN's recent error text;
- `ACL operator has no registered launcher` means the runner name was absent
  from the ACL function table, for group and non-group runners alike;
- `current fused operator input is not allocated` names the queue item whose
  input invariant failed, rather than the enclosing fused operation.

On a 910B3, preserve these lines together with the surrounding `fallback cpu`
log. The normal matmul and workspace commands above must contain none of them;
an injected or naturally reproduced failure must stop that ACL runner before an
execute call uses an invalid executor.

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

- sub-32-bit integer `sum`, `max`, and `min`, plus boolean `all` and
  `any`, lack a complete ACL reduction path; promote inputs to a supported
  width when possible;
- composed float32 `atan2` can raise a vector-core exception;
- complex `irfft` can stall;
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

Full, single-axis, and multi-axis `prod` use CANN `aclnnProd`/`aclnnProdDim`.
Multi-axis reductions are lowered to ordered single-axis device reductions.
Float32 forward/backward and uint8/int8/int16/int32/int64 forward match
independent NumPy references on a real NPU without CPU compilation or fallback. See the
[product verification report](../../agent/results/2026-08-30-npu-product-reduction.md).

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

The ACL MatMul family uses the shared launcher while retaining its synchronous
execution policy. This source-only migration still requires the Ascend 910B3
probe and is not hardware validation on this host.

The ACL Expand family uses the shared launcher and intentionally retains its
historical asynchronous execution policy.

The ACL Floor family uses the shared launcher and retains synchronous
execution. This source-only migration still requires the Ascend 910B3 probe.

The ACL NanToNum family uses the shared launcher while retaining its nan,
posinf, and neginf attribute handling and synchronous execution policy.

The ACL Triu family uses the shared launcher while retaining its diagonal
attribute and synchronous execution policy.

The ACL Sigmoid forward family uses the shared launcher and retains synchronous
execution. Sigmoid backward is intentionally outside this slice. This remains
source-only until the Ascend 910B3 probe is run.

The ACL Transpose/Permute family uses the shared launcher while retaining axes
descriptor cleanup and synchronous execution.

The ACL Softmax forward family uses the shared launcher and retains synchronous
execution. Softmax backward is intentionally outside this slice. This remains
source-only until the Ascend 910B3 probe is run.

The ACL Embedding forward family uses the shared launcher and retains
synchronous execution. Embedding backward is intentionally outside this slice.

The ACL Embedding backward owner uses the shared launcher while retaining
numEmbeddings, paddingIdx, scaleGradByFreq, and synchronous execution.

The ACL Roll family uses the shared launcher while retaining shifts/dims array
cleanup and synchronous execution. This remains source-only until the Ascend
910B3 probe is run.

The ACL Gather family uses the shared launcher and retains its dimension
parameter and synchronous execution policy. Scatter is outside this slice.

The ACL ClampTensor family uses the shared launcher while retaining its three
input query and synchronous execution policy. This remains source-only until
the Ascend 910B3 probe is run.

The ACL Stack family uses the shared launcher while retaining tensor-list setup,
dimension handling, and synchronous execution. Tensor-list lifetime cleanup is
outside this migration slice.

These flags are diagnostic and slow execution. Remove them after locating the
failing operation. For general JIT and memory diagnostics, see
{doc}`debugging`.

The ACL Flip family uses the shared launcher while retaining axes setup and
synchronous execution. Its pre-existing aclIntArray lifetime issue is outside
this migration slice.

The ACL Scatter family uses the shared launcher while retaining axis/reduction
handling and synchronous execution. Gather remains outside this slice.

The ACL Concat family uses the shared launcher while retaining tensor-list and
dimension handling. SplitWithSize is intentionally outside this slice.

The ACL SplitWithSize family now uses the shared launcher while retaining its
split-size, tensor-list, and dimension handling. Tensor-list lifetime cleanup
is outside this migration slice.

The ACL Nonzero owner uses the shared launcher and retains synchronous
execution. The SWhere owner is outside this slice.

The ACL Range owner uses the shared launcher while retaining scalar creation,
cleanup, and synchronous execution. Scalar cleanup on exceptional exits is
outside this migration slice.

The ACL Dropout forward owner uses the shared launcher while retaining its
probability, training, seed, offset, and dual-output handling. Dropout backward
is intentionally outside this slice.

The ACL LeakyReLU forward owner uses the shared launcher while retaining its
negativeSlope scalar and synchronous execution. LeakyReLU backward is outside
this slice; scalar exceptional cleanup is unchanged.

The ACL LeakyReLU backward owner uses the shared launcher while retaining its
negativeSlope, selfIsResult, scalar cleanup, and synchronous execution.

The ACL ArgReduce owner routes both max and min execution through the shared
launcher while retaining is_max, dim, keepdims, and dual-output handling.

The ACLNN declaration header has an explicit include guard. This is a source
organization contract and does not constitute Ascend hardware validation.

The ACL Random owner routes uniform and normal execution through the shared
launcher while retaining seed, offset, unsupported-type diagnostics, and
synchronous execution.

The ACL SiLU forward owner uses the shared launcher and retains synchronous
execution. Backward, Swish, and SwiGlu remain outside this slice.

The ACL SiLU backward owner uses the shared launcher and retains synchronous
execution. Swish and SwiGlu remain outside this slice.

The ACL Swish forward owner uses the shared launcher and retains synchronous
execution. Swish backward and SwiGlu remain outside this slice.

The ACL Swish backward owner uses the shared launcher and retains synchronous
execution. SwiGlu remains outside this slice.

The ACL SwiGlu owner uses the shared launcher and retains synchronous execution.

The ACL BatchMatMul owner uses the shared launcher while retaining
cube_math_type and synchronous execution. This remains source-only until the
Ascend 910B3 probe is run.

The ACL TruthReduce owner routes both all and any through the shared launcher
while retaining reduce_all selection, keepdims, and RAII axes descriptors.

The ACL Conv2d forward owner uses the shared launcher while retaining its
group, bias, and convolution descriptor handling. Conv2d backward is outside
this slice.

The ACL Conv2d backward owner uses the shared launcher while retaining its
three-output gradient query and descriptor cleanup.

The ACL RmsNorm forward owner uses the shared launcher while retaining eps and
dual-output handling. Its gradient owner is outside this slice.

The ACL RmsNorm gradient owner now uses the shared launcher and retains its
multi-input, dual-output, and synchronous execution handling.

The ACL LayerNorm forward owner uses the shared launcher while retaining
normalizedShape, eps, three outputs, and descriptor cleanup. Backward remains
outside this slice.

The ACL Dropout backward owner uses the shared launcher while retaining its
scale query and synchronous execution policy.

The ACL Softmax backward owner now uses the shared launcher while retaining its
dimension query and synchronous execution policy.

The ACL RotaryPositionEmbedding forward owner uses the shared launcher and
retains its three input tensors and synchronous execution. Its gradient owner
is outside this slice.

The ACL RotaryPositionEmbedding gradient owner uses the shared launcher while
retaining its four-input, three-output query and synchronous execution.

The ACL Maxpool forward owner uses the shared launcher while retaining its
kernel, stride, padding, dilation, and ceil-mode descriptors. Avgpool and
backward owners remain outside this slice.

The ACL Avgpool forward owner uses the shared launcher while retaining its
pool descriptors, ceil mode, divisor, and padding semantics. Maxpool and
backward owners remain outside this slice.

The ACL UpsampleNearest2d forward owner uses the shared launcher while retaining
output-size RAII and synchronous execution. Backward is outside this slice.

The ACL UpsampleNearest2d backward owner uses the shared launcher while
retaining output/input-size RAII descriptors and synchronous execution.
