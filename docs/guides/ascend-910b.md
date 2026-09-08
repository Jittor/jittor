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
checks its independent result, device residency, and native fallback counter:

```python
import numpy as np
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1

a_np = np.arange(12, dtype=np.float32).reshape(3, 4)
b_np = np.arange(20, dtype=np.float32).reshape(4, 5)

fallback_before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    result = jt.matmul(jt.array(a_np), jt.array(b_np))
    jt.sync_all(True)
    assert result.location() == "device"
    actual = result.numpy()

np.testing.assert_allclose(actual, a_np @ b_np, rtol=1e-5, atol=1e-5)
assert jt.core.backend_fallback_count() == fallback_before
print("ACL matmul passed")
```

Save the probe as `probe_acl.py` outside the checkout and run it only after
sourcing CANN:

```bash
backend_fallback=error python probe_acl.py
```

Jittor uses `use_cuda` as the common accelerator execution flag even when the
selected backend is ACL. `has_acl` is therefore the required discriminator
between Ascend and CUDA.

### Runtime fallback policy

`jt.runtime.backend_fallback` accepts `error`, `warn`, or `allow`. Use
`backend_fallback=error` in the environment before starting every validation
process. `warn` and `allow` are explicit debugging policies, not NPU acceptance
modes. Only a preflight unsupported decision may request fallback; a launcher,
allocation, compilation, or execution exception must propagate after cleanup,
never be retried as CPU execution.

`jt.core.backend_fallback_count()` counts fallback attempts, including requests
rejected by `error`. Compare its value before and after the work in the same
process. `forbid_backend_fallbacks()` temporarily selects `error` and also
rejects a nonzero counter delta on normal exit, including when an inner caller
caught the rejection. An exception from the body propagates unchanged.
Materialization (`numpy()`/fetch) or `jt.sync_all(True)` must occur inside the
scope: it does not synchronize automatically. A lazy tensor created inside and
executed after leaving the scope is not covered.

Focused test nodes must use that scope around the operation and synchronization
and assert NPU execution/residency plus independent values or gradients. A
successful collection, skip, or absent log message is not this evidence. Keep
SDK and launcher logs for diagnosis only; do not grep CPU compilation or
fallback messages to decide whether validation passed. CPU reference/checkpoint
work outside the device computation is not itself a backend fallback.

`tests/backends/npu/conftest.py` installs this guard automatically for each
test when Jittor is already loaded at fixture entry. It preserves the original
setup/call exception instead of replacing it with a counter error. Tests must
still synchronize or fetch before returning: the fixture does not flush pending
work. Standalone scripts and tests outside `tests/backends/npu/` do not inherit
this fixture and must enter `forbid_backend_fallbacks()` explicitly. If Jittor
is first imported after fixture entry, the test also needs an explicit scope.

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
validating this migration, include one unary operation under
`forbid_backend_fallbacks()` and materialize its result inside the scope.
An `execute launcher failed` message is failure-attribution evidence, not a
replacement for the runtime counter check.

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
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

The run is valid only when the ACL execution assertion passes and the
`backend_fallback_count()` delta is zero under `forbid_backend_fallbacks()`.
The test must synchronize inside the scope; the log is diagnostic only.

After diagnosis, repeat the same focused node with per-operator synchronization
disabled to verify the normal asynchronous path still launches on ACL:

```bash
set -o pipefail
backend_fallback=error sync_run=0 python -m pytest -q -s \
  tests/backends/npu/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-async-run.log"
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

set -o pipefail
backend_fallback=error sync_run=1 python - <<'PY' 2>&1 | tee "$TMPDIR/workspace-normal.log"
import numpy as np
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1
fallback_before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    for width in (64, 128, 256):
        value = np.arange(width * width, dtype=np.float32).reshape(width, width)
        result = jt.matmul(jt.array(value), jt.array(value))
        jt.sync_all(True)
        assert result.location() == "device"
        actual = result.numpy()
        np.testing.assert_allclose(actual, value @ value, rtol=2e-4, atol=2e-2)
assert jt.core.backend_fallback_count() == fallback_before
print("ACL workspace normal path passed")
PY

npu-smi info | tee "$TMPDIR/after-workspace.txt"
```

The normal run is accepted only with correct values, device residency, and a
zero native fallback-attempt delta. After
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

On a 910B3, preserve these lines together with the surrounding SDK/launcher
log and the native fallback-attempt delta. The normal matmul and workspace
commands above must contain none of these failure diagnostics;
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
backend_fallback=error python -m nox -s npu
```

The session resolves `python_config_path` from `JITTOR_CI_PYTHON` rather than
from Nox's own interpreter. If a direct launch sets `python_config_path`, it
must point to the config helper for that same Python version; a mismatched
helper produces an extension suffix that the hardware Python cannot import.

To run the same core tests directly:

```bash
export JITTOR_TEST_DEVICES=npu
export backend_fallback=error
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

backend_fallback=error python tests/backends/npu/manual/run_qwen3_transformers.py \
  --model "$QWEN3_MODEL" \
  --dtype bfloat16 \
  --max-new-tokens 8 \
  --runs 3
```

Run this command only after the CANN, device-selection, and isolated-cache setup
above. The probe loads weights on CPU, explicitly migrates the model to the
visible NPU, prints `npu-smi` while the model is resident, performs greedy eager
attention generation with KV cache under `forbid_backend_fallbacks()`, and
rejects any native fallback attempt during generation. CPU checkpoint
deserialization is expected and is not used as evidence for the model forward.

The validated Qwen3-8B checkpoint has 8,190,735,360 parameters. Float32 uses
32,376 MB of device memory on one 64 GB 910B3. Both float32 and bfloat16 report
accelerator-resident parameters and `has_acl=use_acl=use_cuda=1`. Historical
zero-fallback log reports must be revalidated with the native counter and scope
on the current branch. The maintained bfloat16 eight-token request
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
  [Qwen3 forward/backward report](../results/transformers/2026-08-30-qwen3-ascend-training.md);
- ACL does not provide general float64 operator coverage, so float64 fallback
  is not accepted as evidence for an NPU operation.

Float16/float32 `arg_reduce` forward and value-output backward are maintained
ACL capabilities. Forward uses CANN MaxDim/MinDim and backward scatters the
upstream gradient to the selected first index; current real-device validation
must reject fallback attempts through the native policy and counter. See the
[focused verification report](../results/2026-08-30-npu-arg-reduce-backward.md).

Full, single-axis, and multi-axis `prod` use CANN `aclnnProd`/`aclnnProdDim`.
Multi-axis reductions are lowered to ordered single-axis device reductions.
Float32 forward/backward and uint8/int8/int16/int32/int64 forward match
independent NumPy references on a real NPU. Revalidation requires device
residency and zero fallback attempts within the guarded computation. See the
[product verification report](../results/2026-08-30-npu-product-reduction.md).

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

The ACL LayerNorm backward owner uses the shared launcher while retaining its
normalizedShape/outMask descriptors, three-output gradient query, and cleanup.

The ACL GroupNorm forward owner uses the shared launcher while retaining its
group, epsilon, and three-output query. GroupNorm backward remains outside this
slice.

The ACL GroupNorm backward owner uses the shared launcher while retaining its
output-mask descriptor, group attributes, three-output query, and cleanup.

The ACL MaskedSelect owner uses the shared launcher while retaining its
two-input mask query and synchronous execution.

The ACL Index owner uses the shared launcher while retaining its index query
and synchronous execution. SliceV2 remains outside this slice.

The ACL SliceV2 owner uses the shared launcher while retaining begins, ends,
steps, and axes descriptor handling with synchronous execution.

The ACL StridedSliceAssignV2 owner uses the shared launcher while retaining its
gradient memset branch and slice descriptor handling.

The ACL InplaceMaskedScatter owner uses the shared launcher while retaining its
tracked base-to-output device copy dependency and synchronous execution.

The ACL IndexPutImpl owner uses the shared launcher while retaining its
index-tensor-list handling and synchronous execution. Accumulate remains
outside this slice.

The ACL IndexPutImpl accumulate owner uses the shared launcher while retaining
its tracked output memset and index tensor-list dependency.

The ACL FlashAttention forward owner uses the shared launcher while retaining
prefix/qstart/kvstart RAII descriptors and synchronous execution. Backward,
incremental, and KV-cache owners remain outside this slice.

The ACL FlashAttention backward owner uses the shared launcher while retaining
its RAII descriptors, three gradient outputs, and synchronous execution.

The ACL incremental FlashAttention owner uses the shared launcher while
retaining block-table, actual-sequence, cache-view cleanup, and synchronization.
KVCacheMemcpy remains outside this slice.

The ACL AdamWList owner uses the shared launcher for each tensor update while
retaining its fused device-copy checks and single synchronization point after
the update loop.

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

The ACL Avgpool backward owner uses the shared launcher while retaining its
countIncludePad/divisorOverride semantics and descriptor cleanup.

The ACL Maxpool backward owner uses the shared launcher while retaining its
pool descriptors, ceil mode, output handling, and cleanup.

The ACL UpsampleNearest2d forward owner uses the shared launcher while retaining
output-size RAII and synchronous execution. Backward is outside this slice.

The ACL UpsampleNearest2d backward owner uses the shared launcher while
retaining output/input-size RAII descriptors and synchronous execution.

The ACL launcher migration boundary is now explicit: KVCacheMemcpy is a
per-token aclrtMemcpyAsync path, not a workspace/query executor owner, and must
remain outside BaseOpRunner::launch. The remaining 8.06 work is module-level
registration, attribute data plumbing, descriptor caching, and process_acl
removal with their stated prerequisites.

The obsolete ACL `op_idx_map` has no runtime consumers. Reduce dispatch keeps
its explicit operator ids in `acl_op_exec.cc`; the map definition and extern
were removed without changing dispatch semantics.

## Shared launcher migration is closed for the standard owners

The last four standard owners that still drove the aclnn execute call
themselves now use `BaseOpRunner::launch`: SWhere, Sigmoid backward, BatchNorm
forward, and BatchNorm backward. Each kept its synchronous execution policy,
BatchNorm kept its training/momentum/epsilon attributes and three outputs, and
BatchNorm backward still frees its output mask after the launch has
synchronised.

This matters beyond tidiness. Those four were the last places where a failed
`aclnn` execute only printed a line and returned, leaving the output var
undefined while the graph continued. Routing them through `launch` makes the
failure raise with the operator name and decoded ACL status.

No hand-rolled tail remains: all 71 `executeOp` owners are tail-free. The last
holdout was reduce prod, whose three paths -- whole tensor, one axis, and a
staged multi-axis reduction over intermediate tensors -- now each call `launch`,
the staged one asynchronously so its unconditional barrier still lands before
the intermediates are freed. Two owners keep a `syncRun()` of their own for a
reason the shared tail cannot express: the AdamW loop synchronises once after
its last step rather than once per tensor, and the staged product path after
freeing its intermediates. KVCacheMemcpy never had a tail; it is a per-token
`aclrtMemcpyAsync` path with no aclnn workspace executor.

The prod conversion also closed a silent failure: `ret = aclnnProd(...)` was
assigned and never read, so a failed product returned the untouched output
buffer as the reduction with nothing logged. The same shape applied to a failed
`aclnnMaxDim`/`aclnnMinDim` workspace query, which printed a line and returned
with both the values and the indices output left uninitialised. Both raise now.

Validate on an Ascend 910B3 after sourcing CANN and confirming the device:

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/npu/test_acl.py \
  tests/backends/test_acl_dtype_preservation.py \
  2>&1 | tee "$TMPDIR/acl-launcher-close.log"
```

The run counts only if all four owners actually executed on the NPU with
independent result/gradient checks. Each owner must synchronize inside
`forbid_backend_fallbacks()` and record a zero `backend_fallback_count()` delta.
Fallback attempts are NOT NPU validation, even if the request was rejected and
caught by the test. Preserve `execute launcher failed` and
`aclrtSynchronizeStream failed` diagnostics to attribute failures; log matching
is not the fallback acceptance gate.

Then repeat with `sync_run=0` to confirm the asynchronous path still launches
on ACL, applying the same runtime scope/counter checks.

Until that run exists, this is a source-only change. Hosts without CANN and an
Ascend device must not report hardware validation for it.

## Syntax-checking ACL sources without CANN

A host without CANN can still parse these sources against a generated stub SDK;
see `agent/skills/acl-host-syntax-check`. That check catches parse errors,
unknown identifiers, and a workspace query passed where a launcher belongs. It
cannot check the arguments of an `aclnnXxxGetWorkspaceSize` call, because those
signatures are not knowable without the SDK. It is not hardware validation.

## Python dispatch migration on another machine

The `2.0-refactor` Python dispatcher now takes its target from
`jt.core.dispatch_context(inputs)`. On a configured Ascend runtime this must
report `("acl_legacy", device_id)` while accelerator mode is enabled. A pending
or host-staged tensor is not evidence of a CPU execution target. This migration
has CPU and CUDA evidence only; the ACL registrations still need a real CANN
build and execution on the target machine.

After the CANN and device preflight above, run the existing RMSNorm, rotary,
SiLU and attention nodes with NPU selection, for example:

```bash
PYTHONPATH=python JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
python -m pytest -q -s tests/backends/npu/test_acl_torch_compat.py \
  -k 'rms_norm or rotary or silu or sdpa'
```

Record actual executed node ids, numerical/gradient comparisons, physical NPU
residency and zero CPU fallback using the checks above. A collected or skipped
node is not validation. Do not replace missing hardware evidence with the
host-only registry tests. Rebuild extensions against the current core.

Legacy `nn.backends.hooks.*`, `jt.cudnn` and `compile_extern.*` library attributes
are now read-only views. Backend integrations publish through
`_runtime.dispatch.register_kernel` and `_runtime.backend_libraries`; temporary
test overrides use `override_kernel`, which also restores absent registrations.
ACL implementations now register without `change_function` or public Module
replacement. The existing ACL source conversion remains until the separate
native SDK migration is complete.

### Custom ACL source migration

Custom SDK snippets must declare their source backend explicitly:

```python
result = jt.code(shape, dtype, inputs, cuda_header=acl_header,
                 cuda_src=acl_source, backend="acl")
```

The former `// aclop` marker is no longer a dispatch mechanism. The same
`backend="acl"` argument is required for preallocated `outputs=` snippets;
gradient CodeOps inherit it automatically. This does not select an NPU or make
an arbitrary CUDA kernel valid ACL code. Wrong accelerator targets fail before
execution; an optional `cpu_src` remains the CPU implementation.

The canonical Python SDK builders are in `jittor.backends.acl.kernels.ops`;
legacy `jittor.extern.acl.aclops` imports resolve as same-object aliases.
Rebuild native extensions against BackendOps ABI 2 and the updated CodeOp
constructor. On the target CANN machine, rerun indexing forward/backward and
writeback, normalization, convolution, explicit CodeOp outputs/gradients, and
HCCL nodes under the strict fallback policy above. Host syntax checks and
CPU/CUDA regressions are not evidence that these SDK calls passed on NPU.

Native basic Getitem/Setitem now use checked ACL device-copy plans for integer
indices, positive-step slices, new axes, ellipses, empty selections and broadcast
assignment. Validate chained forms such as `x[0][1] = value`, fp16/bf16 indexing
gradients with float32 upstream weights, and `return_x` on the target machine.
Copies must follow `aclstream` producers; a final synchronization alone cannot
repair a missing producer dependency. Non-equivalent storage overlap is rejected
before writes. Advanced/string/negative-step indexing and reduction assignment
are not implemented by this native copy entry; existing Python ACL paths may
handle variants before reaching it. Scalar broadcast currently uses repeated
device copies and has not been optimized or performance-validated.

## ACL attribute data-channel owner

The host-only attribute channel is defined by
`backends/acl/include/aclops/acl_data_channel.h`. It is intentionally
compiled before any CANN probe: the header owns the versioned record, typed
scalar/vector values, schema defaults, and deterministic cache key, but it does
not call ACL or construct an `aclTensor`. The corresponding source contract is
`tests/structure/test_acl_data_channel_contract.py` and can be run on a CPU
host:

```bash
python -m pytest -q \
  tests/structure/test_acl_data_channel_contract.py \
  tests/structure/test_acl_data_schema_normalizer.py
```

For the future generated-attribute consumer, `AclDataOwner::consume` supplies
a bounded `AclDataView` with typed scalar/vector accessors. This view is only
valid during the callback, so an eventual ACL adapter must copy values into its
`OpAttr`/descriptor arguments before returning. The host contract exercises
that callback and its wrong-type failure path; it does not construct an
`aclTensor`, invoke `aclnn`, or imply that the attribute channel is wired into
an existing NPU operator.

`AclAttrRunnerContract` freezes the next migration boundary: each registry
entry supplies a fixed list of `AclAttrBinding` names and types. Duplicate,
undeclared, or type-incompatible bindings fail as internal integration errors
before any record is decoded. Its host-only `consume()` callback is the shape
that a future `BaseOpRunner` adapter will use to copy values into `AclOpAttr`;
it does not include CANN headers or allocate device descriptors. The CPU
contract test therefore proves only registration and decoder behavior, not
NPU execution.

On an Ascend 910B3, the first attribute owner must be validated only after its
Python caller, generated `OpAttr` construction, and JIT key are migrated in the
same change. Source the CANN environment, select the allocated card, and run
the owner-specific node (the example below uses the current softmax/triu
cohort):

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
set -o pipefail
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/npu/test_acl_torch_compat.py -k 'softmax or triu' \
  2>&1 | tee "$TMPDIR/acl-attribute-data.log"
```

Record the 910B3 model, CANN version, selected device, exact node ids, and the
zero native fallback-attempt delta in the handoff. The owner-specific node must
enclose device execution and synchronization in `forbid_backend_fallbacks()`.
A host-only decoder pass does not close
8.06 and must not be reported as NPU hardware validation.
