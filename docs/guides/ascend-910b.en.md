# Ascend 910B: Configuration and Verification

This guide configures a Jittor 2.0 source checkout for Ascend 910B series devices and
verifies that operations **actually execute through ACL**. Installing the Ascend driver,
firmware, and CANN itself is outside what Jittor covers -- start from a host that already
has the vendor runtime and compiler installed.

## Verified baseline

The maintained gates pass on the following combination:

| Component | Verified version |
| --- | --- |
| Device | Ascend 910B3 |
| Architecture | Linux aarch64 |
| Driver reported by `npu-smi` | 25.5.1 |
| CANN toolkit and `ccec` | 9.0.0 |
| Python | 3.9.25 |
| NumPy | 1.26.4 |
| pytest / pytest-timeout | 7.4.4 / 2.3.1 |

These versions are a **reproduced baseline**, not a claim that "all other CANN versions are
incompatible". Jittor declares support for Python 3.7 through 3.13; for Ascend environments
where vendor packages may impose narrower Python constraints, 3.9 through 3.11 is the
practically usable range.

## Check the device and toolchain

Before importing Jittor, confirm that the driver sees a healthy device:

```bash
npu-smi info
```

Locate the CANN environment script and load it in the current shell. **Make the location
configurable**; do not hard-code host-specific install paths:

```bash
export ASCEND_HOME=/path/to/Ascend/cann-9.0.0
export CANN_SET_ENV="$ASCEND_HOME/set_env.sh"

# 某些 CANN 的 set_env.sh 在 `set -u` 下要求这两个变量已存在。
: "${LD_LIBRARY_PATH:=}"
: "${CMAKE_PREFIX_PATH:=}"
source "$CANN_SET_ENV"

ccec --version
```

`npu-smi info` **must succeed in the same execution environment as Python**. In a
container, pass through the required Ascend devices and driver libraries according to the
vendor container runtime's instructions.

## Install the source checkout

Create a dedicated Python environment and install it in editable mode. The NPU
verification gate also needs SciPy and pinned pytest tools:

```bash
python -m pip install -e .
python -m pip install \
  "numpy==1.26.4" \
  "scipy==1.13.1" \
  "pytest==7.4.4" \
  "pytest-timeout==2.3.1"
```

Run from the repository root. Without an editable install, put the checkout's Python
package first:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

## Isolate JIT state

Jittor compiles the core and operators on first use. Keep mutable state outside the
checkout, and give each concurrent run **a different `JITTOR_HOME` or `cache_name`**:

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

**First JIT or extension compiles must run serially. Tests and benchmarks must not share a
compile cache.** After changing CANN, the driver, the host compiler, or ACL sources, switch
to a new cache name -- **do not diagnose a new stack with stale binaries.**

## Select the device

Before importing Jittor, restrict the process to the device allocated to you:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>
```

Visible devices are renumbered inside the process. **Do not pick a device already occupied
by another workload.**

## Run a real ACL probe

**Verifying that import succeeds alone does not prove NPU support.** The probe below
requires ACL to be present, enables the accelerator flag, performs a float32 matrix
multiplication, and checks the independent reference result, device residency, and the
native fallback counter:

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

Save the probe as `probe_acl.py` outside the checkout, and **run it only after source-ing
CANN**:

```bash
backend_fallback=error python probe_acl.py
```

Even when the selected backend is ACL, Jittor uses `use_cuda` as the generic accelerator
execution flag. Therefore **`has_acl` is the criterion that distinguishes Ascend from
CUDA.**

### Runtime fallback policy

`jt.runtime.backend_fallback` accepts `error`, `warn`, and `allow`. **Set
`backend_fallback=error` in the environment before starting every verification process.**
`warn` and `allow` are explicit debugging policies, **not NPU acceptance modes**. Fallback
may be requested only when the pre-check stage judges the operation unsupported; exceptions
from the launcher, allocation, compilation, or execution must propagate after cleanup,
**never be converted into a retry on CPU**.

`jt.core.backend_fallback_count()` counts fallback **attempts**, including requests rejected
by `error`. Compare the value before and after the work within the same process.
`forbid_backend_fallbacks()` temporarily selects `error` and, on normal exit, **rejects a
nonzero increment in the counter** -- even if an inner caller caught that rejection.
Exceptions raised by the function body propagate unchanged.

**Materialization (`numpy()`/fetch) or `jt.sync_all(True)` must happen inside the scope**: it
does not synchronize automatically. Lazy tensors created inside the scope but executed after
leaving it **are not covered**.

Targeted test nodes must wrap the operation and synchronization in that scope and assert NPU
execution/residency plus independent numerical or gradient results. **Successful collection,
skips, or "some message is absent from the log" are not such evidence.** SDK and launcher
logs are diagnostic only -- do not grep for CPU compilation or fallback messages to decide
whether verification passed. CPU reference/checkpoint work outside device computation does
not itself count as backend fallback.

`tests/backends/acl/conftest.py` automatically installs this guard for every test when
Jittor is already loaded at fixture entry. It preserves the original setup/call exception
instead of replacing it with a counter error. Tests must still synchronize or fetch before
returning -- **the fixture does not flush pending work**. Standalone scripts and tests
outside `tests/backends/acl/` do not inherit the fixture and must explicitly enter
`forbid_backend_fallbacks()`. If Jittor is imported for the first time after fixture entry,
that test also needs an explicit scope.

## Per-operator synchronization checks

On Ascend 910B3, first source the CANN environment, confirm device health, select the
allocated device, then run:

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/acl/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-sync-run.log"
```

`sync_run` is off by default; per-operator synchronization is only a diagnostic: enabling it
(`sync_run=1`) makes every `BaseOpRunner` wait for `aclstream` immediately after launch, at
the cost of eliminating all host-device overlap (a four-layer transformer training step on
Ascend950PR goes from 5.05 ms to 10.39 ms, with bitwise identical results). On
synchronization failure Jittor raises an exception containing the operator name, numeric
return code, and decoded ACL error. Keep the complete log and extract the attribution line:

```bash
rg "aclrtSynchronizeStream failed" "$TMPDIR/acl-sync-run.log"
```

**A run is valid only when the ACL execution assertion passes and the
`backend_fallback_count()` increment is zero under `forbid_backend_fallbacks()`.** Tests must
synchronize inside the scope; logs are for diagnosis only. `execute launcher failed` is
evidence for failure attribution and **cannot replace the runtime counter check**.

After diagnosis, rerun the same targeted node with per-operator synchronization disabled to
confirm the normal asynchronous path still launches on ACL:

```bash
set -o pipefail
backend_fallback=error sync_run=0 python -m pytest -q -s \
  tests/backends/acl/test_acl.py::TestACL::test_float32_matmul_runs_on_acl \
  2>&1 | tee "$TMPDIR/acl-async-run.log"
```

`JT_SYNC=1` is another compile-time diagnostic switch that applies to the entire executor.
It **cannot replace** the `BaseOpRunner` `sync_run=1` check above, and toggling it may
rebuild the JIT cache.

## Verify ACL tensor and workspace ownership

The ACL runner explicitly creates tensor descriptors and obtains the shared aclnn workspace
from Jittor's temporary allocator. On 910B3, source CANN, check the device, record memory,
then run a process that grows the workspace through several normal matrix multiplications:

```bash
source "$CANN_SET_ENV"
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>
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

**The normal run is accepted only when the numbers are correct, the result is resident on
device, and the native fallback attempt increment is zero.** After the Python process exits
it must no longer appear in `npu-smi`; compare `before-workspace.txt` with
`after-workspace.txt` to confirm the workspace was released rather than held by an orphan
process.

**Do not create unbounded allocations on a shared NPU on purpose.** When an existing
workload naturally reproduces a workspace failure, keep its log and extract the attribution:

```bash
rg "ACL workspace allocation failed" "$TMPDIR/workspace-failure.log"
```

The error must report `workspace requested bytes`, `workspace allocator`, and the underlying
allocation failure. A small probe in a new process afterwards must still pass without CPU
fallback -- otherwise the failed allocation left the global workspace in a state that is not
clean and retryable.

Three kinds of runner failures now name the stage and operator, instead of continuing with
an invalid executor or inspecting the outer fused graph by mistake:

- `aclnn workspace-size query failed` -- includes the operator name, return code, decoded
  ACL status, and CANN's most recent error text;
- `ACL operator has no registered launcher` -- the runner name is not in the ACL function
  table (the same applies to grouped and non-grouped runners);
- `current fused operator input is not allocated` -- points at the queue item whose input
  invariant failed, rather than the outer fused operation.

On 910B3, keep these lines together with the surrounding SDK/launcher logs and the native
fallback attempt increment. **None of these failure diagnostics should appear in the normal
matmul and workspace commands above**; an injected or naturally reproduced failure must stop
that ACL runner **before** the execute call uses an invalid executor.

## Run the maintained NPU gate

The NPU nox session sets up isolated state, checks `npu-smi`, runs a real ACL matmul probe,
and then executes the maintained backend and OpInfo tests. nox itself uses Python 3.11, while
`JITTOR_CI_PYTHON` points at the provisioned Ascend Python environment:

```bash
python -m pip install -r requirements/dev-tools.txt
export CANN_SET_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export JITTOR_CI_PYTHON=/path/to/ascend-python/bin/python
export ASCEND_RT_VISIBLE_DEVICES=<分配到的设备>
backend_fallback=error python -m nox -s npu
```

The session resolves `python_config_path` from `JITTOR_CI_PYTHON` rather than from nox's own
interpreter. If `python_config_path` is set when launching directly, it must point at a
config helper for the **same Python version** -- a mismatched helper produces an extension
suffix that the hardware Python cannot import.

Run the same core tests directly:

```bash
export JITTOR_TEST_DEVICES=npu
export backend_fallback=error
"$JITTOR_CI_PYTHON" -m pytest -v --timeout=600 \
  tests/backends/acl/test_acl.py \
  tests/backends/acl/test_aclop.py \
  tests/backends/acl/test_acl_indexing.py \
  tests/ops/test_ops.py
```

**Do not treat a passing CPU fallback as NPU coverage.** When adding or fixing an operation,
add a targeted assertion that captures ACL execution or otherwise proves the declared device
performed the computation.

## Run Qwen3-8B with Transformers

The maintained manual probe runs a local Qwen3 checkpoint through the Jittor Torch shim and
Transformers 4.56.2. Install the optional model dependencies and place the checkpoint outside
the source tree:

```bash
python -m pip install "transformers==4.56.2" "jinja2==3.1.6"
export QWEN3_MODEL=/path/to/Qwen3-8B
export JITTOR_TORCH_SHIM=1

backend_fallback=error python tests/backends/acl/manual/run_qwen3_transformers.py \
  --model "$QWEN3_MODEL" \
  --dtype bfloat16 \
  --max-new-tokens 8 \
  --runs 3
```

**Run this command only after completing the CANN, device selection, and cache isolation
configuration above.** The probe loads weights on CPU, explicitly migrates the model to the
visible NPU, prints `npu-smi` while the model is resident, performs greedy eager attention
generation with a KV cache under `forbid_backend_fallbacks()`, and rejects any native
fallback attempt during generation. **Checkpoint deserialization on CPU is expected behavior
and is not evidence about the model forward pass.**

The verified Qwen3-8B checkpoint has 8,190,735,360 parameters. float32 occupies 32,376 MB of
device memory on one 64 GB 910B3. Both float32 and bfloat16 report parameters resident on
the accelerator with `has_acl=use_acl=use_cuda=1`. **Historical zero-fallback log reports
must be re-verified on the current branch with the native counter and scope.** The
maintained bfloat16 eight-token request stops at `[19, 13, 151645]`, and every repeated run
decodes to `4.`. Use `--dtype float32` to run the original single-token probe.
**These are correctness probes, not throughput benchmarks.**

## Current limitations

The maintained 910B gate **deliberately skips** the following reproduced gaps rather than
letting them abort or hang the process:

- integer `sum`, `max`, `min` below 32 bits, and boolean `all`, `any`, lack a complete ACL
  reduction path -- promote inputs to a supported width where possible;
- the composite float32 `atan2` may trigger a vector-kernel exception;
- complex `irfft` may hang;
- native FlashAttention tests need the optional `jt.nn.FlashAttention` implementation and
  are skipped when it is missing;
- Qwen3 bfloat16 has been verified for eager, gradient-free greedy inference. Fused ACL SDPA
  is separately verified on Qwen3-0.6B with FlashAttentionScoreV2 prefill and
  IncreFlashAttentionV4 decode. Qwen3-0.6B float32 eager forward, causal LM loss, and
  backward have also passed with zero fallback; the current optimized RoPE result needs an
  external Transformers module patch to route Qwen3 to `jt.nn.rotary_emb`. Optimizer updates,
  BF16 training, BF16 SDPA and training for Qwen3-8B, sampling, quantization, and other model
  families remain **separate capability gates**;
- ACL does not provide general float64 operator coverage, so **float64 fallback is not
  accepted as evidence of NPU computation**.

The float16/float32 `arg_reduce` forward and value-output backward are maintained ACL
capabilities: the forward uses CANN MaxDim/MinDim, and the backward scatters the upstream
gradient to the first selected index; current real-device verification must reject fallback
attempts through the native policy and counter.

Full, single-axis, and multi-axis `prod` use CANN `aclnnProd`/`aclnnProdDim`, with multi-axis
reductions lowered to ordered single-axis device reductions. The float32 forward/backward and
the uint8/int8/int16/int32/int64 forward agree with independent NumPy references on a real
NPU. Re-verification requires device residency and zero computation fallback attempts inside
the guard.

Executable evidence and exit conditions are in the
[active issue ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md).

## Debugging on Ascend

Jittor's own logs, `JT_SYNC=1`, and `sync_run=1` are covered above and in the
[debugging guide](debugging.en.md). The following are **CANN-side** switches: they belong to
the vendor runtime, do not override Jittor flags and vice versa, and take effect only in a
process that has sourced `set_env.sh`.

| Variable | Effect |
| --- | --- |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | Print CANN slog to standard output instead of the default log directory; this interleaves vendor errors and Jittor operator logs chronologically in one output |
| `ASCEND_GLOBAL_LOG_LEVEL` | Global log level for CANN components: `0` DEBUG, `1` INFO, `2` WARNING, `3` ERROR (default). `0` and `1` are extremely verbose and clearly slow execution; **use them only for minimal reproductions** |
| `DUMP_GE_GRAPH` / `DUMP_GRAPH_LEVEL` | GE (Graph Engine) graph dump: the former controls verbosity, the latter controls which stage's graphs are dumped; `DUMP_GRAPH_PATH` sets the output directory |
| `TASK_QUEUE_ENABLE` | The **torch_npu-side** asynchronous dispatch queue. `0` disables it, making dispatch synchronous, so host cost is attributed to the line that actually initiates it |

Two usage points:

- **GE graph dumps produce nothing for Jittor.** Jittor's ACL backend launches aclnn
  operator by operator (plus a small number of AscendC fused kernels), and **there is no GE
  graph in the process**. `DUMP_GE_GRAPH` is useful only on the side that actually builds
  graphs -- for example torch_npu graph mode used for comparison; looking for Jittor's graph
  with it will find nothing. To see Jittor's own graph use `jt.profile` (see
  [Performance and memory profiling](../notes/profiling.en.md)).
- **`TASK_QUEUE_ENABLE` is a knob on the reference side; do not change only one side.** It
  changes how torch_npu dispatches, and therefore its wall-clock time. When comparing
  timings **keep each side's default**; set `TASK_QUEUE_ENABLE=0` temporarily only when
  attributing host overhead, and state that the number was obtained with the queue disabled.

Diagnostic variables likewise fall into the "remove when done" category: they change timing,
and timings measured with them enabled are not evidence.

### Benchmark against torch_npu

Measure the end-to-end gap between the Jittor compatibility layer and `torch + torch_npu` for
the same model code with the maintained comparison suite (the reference side needs a separate
Python environment with a matching `torch_npu`; source CANN's `set_env.sh` on both sides
first):

```bash
source "$CANN_SET_ENV"
python benchmarks/torch_compat/run.py --device npu --gpu <分配到的设备> \
  --torch-python /path/to/torch-npu-env/bin/python
```

Here `--gpu` sets `ASCEND_RT_VISIBLE_DEVICES`, and device memory is taken from this process's
usage in the `npu-smi` process table. CPU fallback is forbidden by default; add
`--allow-fallback` only while investigating. For the measurement conventions, how the table
is read, and a probe that bypasses Jittor to call aclnn directly, see
[Benchmarking](../performance/benchmarking.en.md) and
[`benchmarks/acl/README.md`](https://github.com/Jittor/jittor/blob/master/benchmarks/acl/README.md).
**Benchmarks must use a different compile cache from unit tests, and must not share a card
with others.**

## Troubleshooting

**ACL not detected**: in the same shell, check `CANN_SET_ENV`, `ccec --version`, the Python
architecture, and the CANN library paths. **Do not catch and ignore CANN registration or
compilation errors** -- an incomplete backend configuration must fail explicitly.

**Asynchronous device errors are reported after the Python line that caused them**: rerun
only the minimal reproduction with

```bash
export JT_SYNC=1
export JT_TRACE_PY_VAR=3
```

These switches are diagnostic and slow execution; **remove them once the failing operation is
located**. For general JIT and memory diagnostics see the [debugging guide](debugging.en.md).

Developer contracts for the shared launcher, the attribute data channel, the descriptor cache
shell, and device acceptance criteria are in the
[ACL backend contracts](../development/acl-backend-contracts.en.md).
