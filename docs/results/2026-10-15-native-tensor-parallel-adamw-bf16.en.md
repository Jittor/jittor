# Native TP AdamW and BF16 continuation

Status: verified bounded CPU semantics and single-host, two-rank CUDA BF16
training at `479a96093fad9a8c05adf7252984503c37403022`.
Maintainer: native distributed owner. Recheck after functional linear, CUDA
GEMM, AMP gradient publication, collective adjoints, native optimizer or
checkpoint ownership changes. This extends the
[FP32 momentum-SGD result](2026-10-15-native-tensor-parallel.en.md).

The worktree synchronized to `d62e0deba6b6dc74e4e4d9aaa12bb6baefec2d95`,
then retained the existing five TP commits as `642c60b1b`. Validation froze
that baseline plus this change; no source was merged during device runs.
The baseline includes the newer detach and CUDA RNG owners. Functional
linear, its AMP helper and CUDA Lt implementation were unchanged in this
patch. A later shared BF16 epilogue change needs the same fixed-input tests
and precision criteria rerun; this result does not certify that future code.

## Confirmed defect and bounded fix

`RowParallelLinear` added its FP32 master bias directly to the BF16 reduced
product. On CPU, with weight `[[1, 0], [1, 0]]`, input
`[[0.10009765625, 0]]` and FP32 bias `[-0.0373, -0.0369]`, this produced
`[0.06298828125, 0.06298828125]`. Independent Torch autocast linear produces
`[0.0625, 0.0634765625]`: its bias input is rounded to BF16 before addition.
CUDA already produced the Torch result before the fix. This was a CPU
counterexample, not a reproduced CUDA numerical defect.

TP now calls the existing `bias_for_compute_dtype` helper after the row
reduction. The helper and ordinary linear owner were not changed. Exact CPU
and two-rank CUDA regressions verify the output, input/weight/bias gradients,
FP32 master bias and zero backend fallbacks. No optimizer math or second
mixed-precision policy was introduced.

## Real runtime protocol

Both runtimes execute the same fixed Linear(3, 6), tanh, Linear(6, 4) model,
fixed weights and five input/target pairs. Native runs both gather/scatter
options. AdamW uses learning rate `0.003`, betas `(0.8, 0.95)`, epsilon
`1e-6`, weight decay `0.02`, with fused/foreach reference paths disabled.

- Real reference: independent Python, Torch `2.11.0+cu128`, binary `torch._C`,
  no Jittor import and no repository Python path. The full-model oracle uses
  ordinary unsharded Torch linear layers, autograd and AdamW.
- Native: Python 3.11.16, NumPy 1.26.4, CUDA 12.2.140, native NCCL 2.18.3,
  strict CUDA math, two real sm_89 GPUs. Tests and caches are separate from
  benchmark caches; each rank was warmed serially in its own cache.
- Reference TF32 and reduced-precision BF16 reduction are explicitly disabled.
  Dispatch observation records the actual linear GEMM input, weight and bias
  dtypes/devices, rather than observing only the model's final output.
- Native outer `auto_mixed_precision_level=3` covers forward, backward and
  optimizer update; only forward sets `amp_reg=prefer_bfloat16`. Actual CUDA
  GEMM input and weight operands are BF16, while masters, every input/parameter
  gradient, first moments and second moments are FP32. Each operand is checked
  for CUDA dispatch, device storage and driver-reported device ordinal before
  NumPy copies. A forward-only BF16 flag did not guarantee FP32 gradients in
  the retained initial probe.
- Communication observation confirms seven BF16 row-output sum reductions
  and seven FP32 column input-gradient sum reductions for five original and
  two resumed steps. Legacy optimizer reductions and broadcasts are counted
  at their actual owner and remain zero despite being due every step.

Each rank owns 28 parameter elements and 28 elements in each AdamW moment,
versus 52 parameters and 52 elements per moment in the full model. Moment
shapes equal the local master shapes; no full moment tensors are allocated
then sliced. The replicated row bias accounts for the difference from an
exact half. Activations and temporary compute casts are not optimizer state;
no whole-process peak-memory or speedup claim follows from these counts.

Every step records output, loss, input gradient, all parameter gradients,
updated parameters and both moments: 95 arrays per runtime/rank over five
steps. A checkpoint saved after step three restores into fresh native model
and optimizer objects. Both resumed steps match the uninterrupted native
trajectory exactly for every field, and remain within the independent full
Torch oracle's applicable precision criterion.

## Independent BF16 criterion

The original FP32 criterion remains `atol=2e-6, rtol=2e-5`; its BF16 failures
are retained as diagnostics. The BF16 criterion was frozen before native
post-fix acceptance. It uses an independent real-Torch comparison of the
full model against the same TP decomposition, including actual two-rank
Torch NCCL 2.28.9, both gather options, BF16 row partial-output reductions,
FP32 column input-gradient reductions, and CPU singleton calibration.
The full-model oracle remains the acceptance reference; TP calibration does
not replace it with a Jittor self-comparison.

For each individual array, compare maximum absolute error against
`8 * float32_epsilon + budget[field] * max(abs(full_reference_array))`.
With BF16 round-to-nearest unit roundoff `u = 1/256`, each budget is
`(ceil(independent_Torch_envelope / u) + 2) * u`. The two-unit margin allows
implementation rounding association. This is an empirical allowance for
this maintained fixture, not a numerical theorem for arbitrary models.
Native errors were not inputs to the budget calculation.

| Field | Torch full-vs-TP maximum relative Linf | Frozen budget | Native CUDA-vs-full-Torch maximum relative Linf |
| --- | ---: | ---: | ---: |
| Output | 0.005025 | 0.015625 | 0.005025 |
| Loss | 0.002498 | 0.011719 | 0.000873 |
| Input gradient | 0.007692 | 0.015625 | 0.010956 |
| Parameter gradients | 0.010695 | 0.019531 | 0.010152 |
| Updated parameters | 0.001858 | 0.011719 | 0.001998 |
| First moments | 0.006503 | 0.015625 | 0.006410 |
| Second moments | 0.012947 | 0.023438 | 0.011875 |

Values aggregate both ranks, gather options and all five steps; scales are
computed per individual full-reference slice. The reusable Torch TP
calibration script reproduced every array of the independently frozen
calibration exactly. Torch TP FP32 differs from its full model by at most
`3.05e-7` relative Linf; native CUDA FP32 differs from the full Torch oracle
by at most `3.11e-7` and passes the original elementwise criterion throughout.

CPU BF16 tests exercise native generic-contraction output and FP32
state/gradient semantics, and exact continuation. The CPU path does not
promise actual BF16 GEMM input casting: its generic contraction may keep
wider operands. The true BF16 compute claim above is limited to CUDA.
CPU's largest recorded field error is input gradient `0.015385` relative
Linf, within the independently frozen `0.015625` fixture budget.

## Commands and evidence

Configure private environments and caches as required by the worktree and
distributed verification skills. `<native-python>` must import this checkout;
`REAL_TORCH_PYTHON` must be a separately installed real Torch interpreter.
Do not expose the native Cu122/NCCL loader path to the Torch Cu128 reference.
The helper removes that path and the repository Python path in the child.

```bash
REAL_TORCH_PYTHON=<real-torch-python> JITTOR_REQUIRE_REAL_TORCH=1 \
  JITTOR_TORCH_SHIM=0 JT_BACKEND=cpu JT_USE_CUDA=0 \
  PYTHONPATH=python <native-python> -m pytest -q \
  tests/nn/test_native_tensor_parallel.py

# The wrapper sets isolated JITTOR_HOME/TMPDIR from RANK before importing
# pytest/Jittor, then forwards its arguments to pytest. Warm each rank first.
# JITTOR_TP_TEST_DIR is shared; rank-local artifact paths are separate.
REAL_TORCH_PYTHON=<real-torch-python> JITTOR_REQUIRE_REAL_TORCH=1 \
  <native-python> python/jittor/distributed/launch.py --backend nccl -n2 \
  --device-ids=<gpu-a>,<gpu-b> --log-dir <state>/logs --timeout 300 -- \
  <native-python> <rank-cache-wrapper> \
  tests/backends/comm/nccl/test_native_tensor_parallel_nccl.py

# Independent TP calibration: run using real Torch's two-rank launcher,
# without native PYTHONPATH/LD_LIBRARY_PATH; repeat with --gather.
<real-torch-python> -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/_helpers/native_tp_oracle.py --mode tp --device cuda \
  --precision bf16 --output <state>/torch-tp.npz
```

CPU TP: **14 passed, zero skipped** (11.89 s). CUDA TP: **13 passed per rank,
zero skipped** (18.60 s), launcher exit zero. This includes all previous eight
FP32 CUDA cases, the exact row-bias regression and four AdamW rows per rank.
Backend fallback count is zero for every native training row.
Core runner: **322 passed, 42 skipped, one existing xfail**, exit zero
(native 98/41/1 and Torch-mode 224/1). Repository layout passed.
Full structure: **1,459 passed, six skipped, two failed** (166.42 s).
The two failures are central toctree registration for the TP note and both
result pages, and the generated packaging manifest. Those central files are
reserved for integration; they were not changed here. Import layering,
cycle detection and child-process contracts passed. This branch does not
claim a fully green structure gate before the central registration step.

Raw logs, checkpoints, array traces, metadata and frozen hashes are
**unversioned** under `$JITTOR_LAB_ROOT/_state/native-tensor-parallel-bf16/`.
`frozen-bf16-budget.json` records independent calibration hashes and the
criterion formula; `mixed-shared-root-source.json` records the tested 1,115
native runtime files and six owned source/test hashes. CPU logs are
`logs/cpu-tp-adamw-shared-root.log`; CUDA logs are
`logs/formal-adamw-shared-root-two-rank/rank{0,1}.log`. Formal CUDA checkpoint
artifacts and 95-array traces are in `formal-shared-root-checkpoints/`.

The initial new harness incorrectly put checkpoint roots inside rank-local
artifact directories. Its Store waits failed because ranks used different
checkpoint identities; that run was stopped and retained without a pass
claim. The corrected harness uses a shared checkpoint root and separate
rank-local observations. This did not require a production checkpoint change.
Initial probes and a storage-observer timing failure are retained separately;
storage is checked after GEMM consumes its operands and before host readback.

No arbitrary TP operator, attention/sequence parallelism, uneven partition,
DP/ZeRO/FSDP hybrid mesh, larger WORLD, multi-host, non-CUDA accelerator,
other AMP policy, process peak memory or throughput acceptance is claimed.

## Shared Linear accumulator revalidation

Status: actual CPU and two-rank CUDA revalidation passed. The synchronized remote
baseline was `2c0d2681a7db089aa2e996f0ad8e15057fb8e35e`. The prior TP changes
`479a96093fad9a8c05adf7252984503c37403022` and
`b9c804386aaf4c56a186c823732d07b50bea1d8d` were retained and replayed together
with the complete shared Linear series `e64d4a27f`, `ead8bf803`, `e07a2e046`,
and `0c6837b90`. The resulting frozen source parent was
`6c194f0485943e5804e28b70bd237d2ac7c38f98`; the only subsequent test change
moves the CUDA operand observer to the actual cuBLAS call. The precision budget,
inputs, optimizer and checkpoint math remain those accepted above.

The biased BF16 Linear now returns a BF16 cast after adding bias to a FP32 GEMM
product. Indexing the final result's graph no longer identifies the GEMM. The
observer wraps the existing backend's `cublas_matmul`, records its actual inputs
and output, synchronizes the complete Linear, and checks dtype, CUDA dispatch,
resident storage and CUDA pointer ordinal before any NumPy conversion. It also
checks the native call arguments: biased BF16 uses `(False, True, True)` and a
FP32 product; biasless BF16 retains `(False, True)` and a BF16 product. FP32 calls
retain FP32 operands and product. The observer releases its temporary references
instead of retaining prior training graphs.

Each rank's cache was warmed serially, then the same two-rank test file was run
on the real CUDA devices. CPU: **14 passed**, 15.29 s. CUDA: **13 passed per
rank**, 31.52 s, launcher exit 0. This includes the existing eight FP32 tests,
the exact Row bias test, and all four FP32/BF16 AdamW and gather combinations.
Full parameter/input gradients, FP32 masters and moments, actual BF16 GEMM
operands, owned shard sizes, zero legacy optimizer collectives, zero fallback,
and save after three steps followed by an exact two-step fresh-object resume
remain checked. The core suite also passed: **322 passed, 42 skipped, one
existing xfail**.

The independent whole-model Torch oracle and the previously frozen BF16 budget
were reused without recalibration. Across both ranks and gather variants, the
maximum scaled errors were:

| Field | After shared Linear change | Frozen allowance |
| --- | ---: | ---: |
| Output | 0.00502513 | 0.015625 |
| Loss | 0.00249812 | 0.01171875 |
| Input gradient | 0.01095618 | 0.015625 |
| Parameter gradient | 0.01015229 | 0.01953125 |
| Updated parameter | 0.00239895 | 0.01171875 |
| First moment | 0.00650291 | 0.015625 |
| Second moment | 0.01294701 | 0.0234375 |

These are fixture-specific correctness results. A change to fused column bias
rounding can change subsequent TP rounding and does not imply every field's
cross-framework error decreases. Strict FP32 tolerances still fail for BF16 and
remain diagnostic; no performance or universal BF16 error-bound claim is made.

Unversioned evidence is under
`$JITTOR_LAB_ROOT/_state/native-tp-linear-revalidation/`: `frozen-source.json`,
`logs/cpu-14.log`, `logs/warm-rank{0,1}.log`, `logs/formal-launch.log`,
`logs/formal-two-rank/rank{0,1}.log`, `logs/core.log`, and
`formal-checkpoints/adamw-*/rank*/{result.json,native-trace.npz}`. The earlier
`native-tensor-parallel-bf16` evidence and delivered branch remain unchanged.
The launch command is the preceding command with this new artifact directory;
the same isolated per-rank compilation caches were reused after serial warmup.

The complete structure run finished with **1459 passed, six skipped, two
failed** (180.34 s). Its two failures are central integration work: the new
Linear and TP BF16 result pages are absent from the results toctree, and the
root generated `MANIFEST.in` needs regeneration. Import layering/cycles and
child-process contracts passed. Repository layout passed. The parent integrator
owns both central files; this report does not describe the structure run as
fully green.

## Parent integration

The parent combined the delivered TP changes and all four shared Linear
commits with native FSDP/checkpoint ownership on top of
`71c5be74ef6991b75fa79d73fe52eb04c0319d59`. The 36 delivered files matched
their source references exactly; 433 original TP and 141 shared-Linear
revalidation artifacts were rehashed successfully. These checks preserve the
measured source identity rather than relabeling earlier two-rank execution.

The frozen combined source passed the independent-oracle singleton CPU TP
cases within **56 native owner passes**, CPU core **322 passes**, actual CUDA
Linear/cuBLAS **20 passes**, and CUDA compatibility **24 passes**. Central
toctree and manifest registration is now present; full structure passed
**1470**, with six skips. Full command output, skip reasons and source hashes
are unversioned in
`$JITTOR_LAB_ROOT/_state/distributed-bf16-integration/`. The fixture-specific
BF16 allowance and the multi-node limitation remain unchanged.
