# Native FSDP verification

## Two-host maintained entry

[`run_native_multinode.py`](../scripts/run_native_multinode.py) runs the existing protocol
with **two nodes, exactly one allocated CUDA device per node**. It neither
connects to hosts nor transfers files. Both hosts need the same clean Git SHA,
native and independent binary Torch interpreters, and an actually shared output
filesystem. Checkpoint transport and other world sizes are outside this entry.

On each host, set its real local paths and device allocation. `NODE_RANK` is
zero or one; `MASTER_ADDR` is the reachable node-zero address. `PRIVATE_STATE`
is local cache storage, outside both the checkout and `SHARED_OUTPUT`. Identical
local paths and device ordinal zero on different hosts are valid. Host identity
is sampled by the worker, never supplied by a CLI flag.

```bash
<runner-python> agent/skills/jittor-distributed-verification/scripts/run_native_multinode.py \
  --stage warm --source-root "$PWD" --source-sha "$SOURCE_SHA" \
  --python "$NATIVE_PYTHON" --oracle-python "$REAL_TORCH_PYTHON" --nvcc "$NVCC" \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --job-id "$JOB_ID" --device-id "$LOCAL_DEVICE" \
  --state-root "$PRIVATE_STATE" --shared-output "$SHARED_OUTPUT"
```

Repeat this **same command on both hosts**, changing `--stage` in this order:
`warm`, `collectives`, `parity`, `baseline`, `resume`. Before parity, node zero
also runs `--stage oracle`; it generates all four unchanged dense references.
Wait for both nodes to finish each distributed stage before starting the next.
Warm uses public singleton jtrun with actual rank zero/world one, private local
rendezvous, and the same cache as that node's later worker; serial JIT is not
represented as distributed rank initialization. Do not overlap the two warm
commands if the nodes are being simulated on one physical host.

Each training stage creates a fresh public `python -m jittor.distributed.launch`
agent and worker, invokes the maintained pytest node, requires actual execution,
and retains the existing numerical budgets. Fresh resume requires new workers,
models and optimizers. Failed/partial commands keep their claims and logs;
rerunning into the same output is refused. Use new output for a new source
commit or run. A node's private unit cache can persist with normal compiler
fingerprints, but must never be used concurrently or shared with a benchmark.
Neither existing results nor failures are overwritten.

After all stages finish on both nodes, node zero runs `--stage seal` with the
same arguments. It requires all actual node/oracle records, binds raw bytes,
then invokes the independent NumPy rechecker. `verification.json` is written
only after that recheck succeeds. Source snapshots include every tracked regular
file and Git symlink target text. Environment records use a strict allowlist;
they never dump the full inherited process environment.

`--stage plan` is a real host preflight with no CUDA/connectivity claim and no
artifact creation. `nox -s native_multinode -- <the same runner arguments>` is
the thin maintained session. Loopback or two different hostnames alone cannot
prove physical multi-host execution: retain external topology/network evidence
and the actual per-host commands before claiming that separate acceptance.

Native FSDP owns actual local parameters; its shared storage/reduce-scatter
implementation also serves the Torch frontend. Its initial acceptance scope is
CUDA/NCCL WORLD with fixed world size, FP32 master parameters and optional BF16
forward computation with FP32 gradient reduction. This does not establish peak
memory savings, throughput, another backend or world-size migration.

Generate references with an independent binary Torch interpreter. The tool
imports only the NumPy fixture before calling real Torch; it refuses the shim:

```bash
<real-torch-python> agent/skills/jittor-distributed-verification/scripts/native_fsdp_oracle.py \
  --source-root "$PWD" --device cuda --kind sgd --dtype float32 \
  --output-dir "$JITTOR_LAB_ROOT/_state/native-fsdp/oracle"
```

Generate both `sgd`/`adamw` and `float32`/`bfloat16` references. Run
`tests/backends/comm/nccl/test_native_fsdp_nccl.py` through the native launcher
on two actual GPUs, using separate prewarmed `JITTOR_HOME` values per rank.
Set `JITTOR_REQUIRE_NATIVE_FSDP=1`, `JITTOR_NATIVE_FSDP_TEST_DIR` to a shared
artifact directory, and `JITTOR_NATIVE_FSDP_ORACLE_DIR` to those references.
Use three **separate launcher invocations**, with
`JITTOR_NATIVE_FSDP_PHASE=parity`, then `baseline`, then `resume`. The baseline
saves after step three and continues two steps with actual dropout; the fresh
resume process must reproduce both subsequent steps and every rank's Python,
NumPy, native CPU and CUDA RNG arrays exactly. All runs forbid backend fallback
and prove device pointers and actual compute/collective dtypes. A host-resident
collective input may be uploaded only if that same Var was explicitly read back
by the trace beforehand; arithmetic and collective outputs must already reside
on CUDA before any pointer request.

The oracle protocol is fixed before measurement: exact shape and finite checks;
elementwise FP32 `atol=2e-6, rtol=2e-5` or BF16 `atol=4e-4, rtol=3e-2`;
per-parameter update L2 error at most `2e-6 + 2e-5 * oracle_norm` (FP32) or
`1e-6 + 0.03 * oracle_norm` (BF16). Each update must exceed ten absolute
floors to be informative. Every actual parameter/step also injects a zero
gradient and zero update and requires rejection. Per-step norms, errors and
budgets are saved beside the raw tensors. Elementwise absolute tolerance can
permit a large relative error on a tiny gradient: report that error explicitly
instead of calling a passing BF16 comparison exact gradient agreement. Effective
native matmul/cuDNN precision must both be `highest`, and the independent Torch
reference disables TF32 with `highest` precision.

For a low-precision residual, independently calibrate the same fixture with
real Torch FSDP2 before assigning the difference to distributed rounding:

```bash
<real-torch-python> -m torch.distributed.run --standalone --nproc_per_node=2 \
  agent/skills/jittor-distributed-verification/scripts/native_fsdp_torch_calibration.py \
  --source-root "$PWD" \
  --output-dir "$JITTOR_LAB_ROOT/_state/native-fsdp/calibration"
```

This tool uses actual Torch FSDP2/DTensor with BF16 parameters, FP32 reduction
and FP32 outputs and records both real ranks' five-step tensors and source
hashes. Keep this calibration beside the dense oracle. Identify the first
layer/output/gradient difference with fixed oracle weights; preserve the
original thresholds and earlier raw tensors. Check first and second derivatives
on each real device when repairing a core differentiation rule. CPU and CUDA
low-precision kernels can have different rounding boundaries.
