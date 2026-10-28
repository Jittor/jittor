# Dual-rank bf16 FSDP2 FULL save and continued-training verification

- Status: the CUDA FULL save, restore and fixed-input continued training bounded by this report pass; cross-framework bf16 numbers retain an approximate boundary.
- Date: 2026-10-15.
- Baseline: `ad47b58143f5c54e83114f923e23c0c69322130a`, with RNG `1e2ff1aa6b3a391c0c84747413c01bc2731e5626` and AMP `de23770c70f993b4d4fe818a763f9edca7486381` merged in sequence; isolated verification base `5274aeb6ecf1791029889263bc3ab6cc88324a4f`.
- Owner: Torch compatibility layer, FSDP2 and checkpoint maintainers.
- Review when: FSDP2 mixed precision, FULL checkpoint, RNG, Linear rounding policy, Accelerate or independent Torch versions change.

This work only adds regressions and evidence; no computation implementation changed. Two real CUDA ranks on one machine use fixed 4×4 inputs, 3×4 Linear weights and bias, and two AdamW groups. The FP32 row saves after one update; the new bf16 row first performs three updates, saves the FULL model and optimizer, then runs two steps as the expected trajectory and reruns both steps after restoration. The complete model, FQN optimizer state, actual gradients, loss and hyperparameters all enter each rank's record.

The Linear subclass observes input, weight, bias and output before Accelerate's output conversion; across the seven forwards all four tensors are bf16. CUDA driver pointer attribute 9 verifies that every observed tensor is really allocated on that rank's CUDA device. Each step checks the FP32 master shard, the FP32 actual gradients and FP32 `exp_avg` / `exp_avg_sq`; loss is explicitly FP32. FULL shapes are derived from the actual complete state, avoiding treating native flat local shapes as global shapes.

Restoration assertions require exact value-by-value equality and use no cross-framework numerical tolerance. Before saving, the Python, NumPy, Torch CPU and Torch CUDA random streams are advanced; the next values after restoration are also equal value by value. The training inputs themselves are fixed, so this is not additional coverage of dropout or randomly sampled training. Both Jittor ranks run in the fallback `error` scope, with a final cumulative fallback count of 0. Matmul / cuDNN TF32 are disabled on both sides, and the fixture restores the original configuration.

## Actual results

Python 3.11; pytest 7.4.4 and Accelerate 1.10.1 are identical on both sides. Jittor 2.0.0 declares Torch API 2.6.0; the independent reference is real binary PyTorch 2.11.0+cu128, confirmed not to have loaded Jittor and not to have activated the shim. Hardware is two NVIDIA RTX 4090 cards; Jittor uses GCC 11.5 and nvcc 12.2.140. CPU and the two CUDA ranks use separate JITTOR_HOME directories, and the first core, extension and bf16/FP32 NCCL operator warm-ups completed serially in order. No shared Python environment was modified and no new downstream library was installed.

| Verification | Result |
| --- | --- |
| Independent CPU Torch / Jittor bf16 forward and backward precheck | Identical outputs, real CPU, FP32 gradients; Jittor fallback 0 |
| Final fresh two ranks, FP32 × both runtimes | 1 passed per rank, both launchers rc 0, 0 skipped |
| Final fresh two ranks, bf16 × both runtimes | 1 passed per rank, both launchers rc 0, 0 skipped |
| FULL restoration, two-step continuation, four RNG classes | Exactly equal per rank in both runtimes |
| CPU Torch core whitelist plus FSDP mixed precision / optimizer checkpoint / AMP / planner adjacent selections | 253 passed, 1 accelerator skip, 257.56 s, including cold operator compilation |
| Complete `tests/structure` | 1426 passed, 4 skipped, 4 failed, 213.80 s |

The structure failures are this isolated base's existing missing ASV dependency, the ACL Dropout gradient manifest, and the results index / MANIFEST awaiting central maintenance; no claim is made that the complete structure gate passed. The new report also needs central updates to the index and manifest. The layout check and the diff whitespace check pass. After the report was written to disk, the documentation, packaging and private-path selections gave 25 passed, 2 failed, still the shared index and MANIFEST items; the private path and the remaining documentation contracts pass.

The test-source SHA256 of the final eight JSON files on both sides is
`5edc207e7892c8367cc91e17f54d45b6b68209c1bab7a891ec4aa47c2e3581af` in both cases.
FULL weight / bias shapes are `[3,4]` / `[3]`, and the local shard layouts are allowed to differ between the two owners.

## External numerical comparison and boundaries

The maximum absolute difference over the full FP32 trajectory is `1.862645149230957e-9`, passing the existing
`2e-6 + 2e-5 * abs(reference)` threshold. The bf16 maximum absolute difference is
`1.388988457620144e-4`; 194 / 188 scalar comparisons do not pass the FP32 threshold above, and the raw results are retained.

For every floating-point tensor, dividing the maximum absolute difference by the reference tensor's maximum absolute value gives the following bf16 full-trajectory upper bounds:
loss 0.736%, model 0.604%, gradients 2.381%, moments 1.882%. Relative to bf16 epsilon
`2^-7`, these are 0.942, 0.773, 3.048 and 2.409 times respectively; near-zero gradients and EMA cancellation amplify
this normalized error. These are measured values for this fixed five-step specification and are not turned into library-wide tolerances or EXACT commitments.

The minimal CPU/CUDA reproduction uses exactly the same bf16 x / weight / nonzero bias and compares fused
`F.linear` against `x @ weight.T + bias`. Jittor CUDA's output, loss, input grad,
weight grad and bias grad are bitwise identical to the latter in independent Torch; Torch's fused Linear has a different
rounding position, with a single-step output maximum difference of `4.8828125e-4` and a bias grad maximum difference of
`2.44140625e-4`. The gap therefore does not come from a wrong dtype, NCCL or a missing gradient chain; the existing Linear
`APPROXIMATE` fidelity is retained, and no kernel was expanded or training downgraded to FP32 for bitwise matching.

On CPU, a lazy fusion boundary was also found: without materializing the bf16 output, the loss declared as FP32 differs from Torch
unfused by at most `1.8621794879436493e-6`; after materializing the output first, the difference does not exceed
`1.1641532182693481e-10`, and all gradients still agree. This is retained as a CPU precision limitation, and the
compiler was not modified in this work; this report does not claim that CPU FSDP2 or all non-materialized bf16 graphs pass.

Running two precision rows consecutively in the same process, independent Torch once timed out after 180 s on the second `ProcessGroupGloo`
initialization, before entering bf16 computation. The earlier combined execution had also passed; final acceptance uses a new two-rank launcher for each
precision, and all final launchers exited normally. That oracle reinitialization race log is retained, and a process-group
initialization failure must not be used to infer a checkpoint computation failure.

## Reproduction entry

The test node is
`compat/tests/torch/test_accelerate_distributed_integration.py::test_fsdp2_full_model_optimizer_checkpoint_resume[bf16]`;
FP32 corresponds to `[no]`. The environment needs two visible GPUs, NCCL include/lib, a real reference interpreter and the
current source frontend. Set `ACCELERATE_USE_FSDP=true`, `FSDP_VERSION=2`,
`JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED=1` and a separate JSON output directory.

Jittor uses the existing `python/jittor/distributed/launch.py -n 2 --backend nccl --standalone`;
each child sets its own JITTOR_HOME / TMPDIR by RANK before import, then runs the pytest node above.
The reference uses `torch.distributed.run --standalone --nproc-per-node=2` with a byte-identical
external copy of the same node, `--noconftest` and an external rootdir to avoid the monorepo conftest activating the shim; the source-file
provenance directory is completed with read-only links. Each precision uses a new launcher.

Raw logs, the byte-identical reference copy, rank wrappers, separate caches and the eight complete JSON files are under
`$JITTOR_LAB_ROOT/_state/bf16-fsdp2-checkpoint/` and are not versioned. The key logs are
`logs/reference-exact-{no,bf16}.log`, `logs/shim-exact-{no,bf16}.log`,
`shim-exact-{no,bf16}-rank-logs/rank{0,1}.log`, `logs/final-comparison.log`,
`logs/final-quantification.log` and `logs/structure.log`.

Multi-node, TP, ZeRO, SHARDED/DCP planners, cross-runtime checkpoint interchange, ROCm/NPU are not verified,
and no claim is made from this that all of Accelerate or R17 is complete.
