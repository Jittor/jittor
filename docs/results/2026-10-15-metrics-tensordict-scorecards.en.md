# TorchMetrics and TensorDict component scorecards

- Status: measured CPU/CUDA components; shared-auditor result recorded below.
- Reviewed: 2026-10-15.
- Owner: Torch compatibility and downstream verification maintainers.
- Synchronized baseline: `ab006244a34c550a5f7235b1f65ec7d4165bde9d`.
- Clean measurement source: `90ee18a9916acda9b5e86ca665f09a231a41ced1`.
- Review when: maintained case builders, Module/tensor gradients, indexing, package versions/source identity, runner/training probe, precision policy, RNG seeding, device placement, or fallback ownership changes.

This is a new execution of the maintained registered cases, not a relabeling of the historical `5ad8a1929f2e4a88db39b4f0e570e4901615487d` arrays. The earlier [component report](2026-10-15-metric-state-and-tensordict-oracles.en.md) retains its original baseline. No framework, RNG, adapter, central inventory or shared auditor source changed in this batch.

## Exact scope and denominator

Each collection re-derives inventory from the clean release checkout. TorchMetrics 1.7.4 is `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` at `git@github.com:Lightning-AI/torchmetrics.git`; its complete top-level `__all__` axis has 105 members. TensorDict 0.10.0 is `c34264a58b12471dc3e2913ea586b32431872d4c` at `git@github.com:pytorch/tensordict.git`; its top-level axis has 60 members. These are bounded export denominators, not all metrics variants, methods, distributed facilities or an entire-library support percentage.

| Maintained case | Actual APIs and specification | CPU | CUDA |
| --- | --- | --- | --- |
| `torchmetrics_regression` | MeanAbsoluteError + MeanSquaredError, FP32 prediction/target `(4, 8)`; output + both input gradients; zero parameters | L2: 6 compared arrays, exact | L2: 6 compared arrays, exact |
| `tensordict_index_and_stack` | TensorDict + lazy_stack, batch `[2,3]`, FP32 sample `(2,3,4)`, integer/bool indexing, lazy stack/expand and Linear(4,3); both parameter gradients + input gradient | L3: 29 compared arrays, max abs 9.54e-7 | L3: 29 compared arrays, max abs 7.15e-7 |

Metrics is L2 only because there are no trainable parameters. TensorDict adds exactly three explicit SGD steps at learning rate 0.001: output, loss, both parameter gradients, sample gradient and both updated parameters at every step (21 arrays), including a real parameter update. This proves the declared tiny component composition, not Adam, an upstream Trainer, distributed synchronization, persistence or compilation support. Sampling is core 0, broad 4 scenario/device rows, tail 0. The same two API members are referenced by each device's one component scenario; unrelated exports receive no numerator. The shared auditor accepted all four rows with no rejection: **2/105 Metrics APIs at L2 on CPU and CUDA**, and **2/60 TensorDict APIs at L3 on CPU and CUDA**. The other 103/58 exports on each device remain unverified; API methods and variants outside the export axis are not part of these fractions.

## Execution and identity

Two task-owned Python 3.11 interpreters run the shared `verify_repo.py`, `_ecosystem_runner.py` and `training_probe.py`. The oracle is a real binary Torch 2.11.0+cu128, with `_C` SHA256 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`; it explicitly refuses the shim identity. Both sides load actual TorchMetrics 1.7.4 and TensorDict 0.10.0. A real SciPy 1.13.1 dependency directory takes precedence in the private environments; shared installations remain unchanged.

The new `component_measure.py` wrapper runs the maintained probe without replacing its computation. Inside each actual measurement interpreter it verifies every upstream package Python source (341 Metrics files, 36 TensorDict files), records all installed package files excluding bytecode caches (346 Metrics files and 42 TensorDict files), hashes the real oracle binary, and binds initial weights, source commit, probe/case/wrapper hashes, runtime flags, executable/PID and native worktree path. The collector refuses absent or changed bindings. All 20 raw measurement processes have distinct PIDs; a final cross-phase check confirms each repository's entire installed content is identical across both devices and all stages. Both runtimes must have identical complete installed-package hashes; this also binds installed generated files and native package binaries beyond the upstream Python subset.

CPU explicitly hides all CUDA devices and sets canonical/legacy CUDA flags to zero and nvcc paths empty. CUDA runs after CPU, uses one assigned physical GPU with actual resident-device assertions and forbids backend fallback. Both sides use TF32 off, cuDNN benchmark off and float32 matmul precision `highest`; actual JSON readbacks, thread settings and affinity must agree. Device-specific benchmark caches and a separate unit cache are task-owned; first JIT is serialized. Baseline seed 0 exports the reference's complete initial weights. Both seed-7 perturbation processes load those exact original weights, and the output must change. TensorDict training also loads that same seed-0 weight artifact on both sides. No diagnostic timing or allocator value is used to grant L4/L5, speed or memory conclusions.

CPU platform: Linux 6.1.188/glibc 2.34, AMD EPYC 9554 64-Core Processor, g++ 11.5.0, zero visible cards. Runtime threads and OMP/MKL/OpenBLAS are 2. Native auto_graph_replay is 1 with a 268435456-byte retention budget; this is the default policy, and these gradient-bearing correctness probes make no inference replay performance claim. Actual CPU TF32 and cuDNN benchmark readbacks are false, matmul precision highest, use_cuda/use_acl are zero. CUDA platform: one NVIDIA GeForce RTX 4090, driver 580.178.04, Jittor strict nvcc/CUDA 12.2.140 wheel stack and independent Torch CUDA 12.8. Both TF32 switches and cuDNN benchmark are false, matmul precision highest, use_cuda=1/use_acl=0. The raw platform includes the allocated physical-to-visible mapping, UUID, selected NVIDIA distribution metadata for both interpreters, compiler versions and actual flags. Read-only `/proc/PID/maps` snapshots corroborate the CUDA libraries loaded by actual measurement PIDs; they are supplemental software identity, not substitutes for resident tensor checks.

The final shared-schema artifact is `final-scorecard.json`, SHA256 `41a0826b495d63b9a989db7c04ecdc5239df3fed6557b24187721e6f0f48ae80`: four accepted records, zero rejected, 70 compared fields, and 58 unique raw artifact path/hash bindings. Each record also binds the two-stage base/perturb metadata, full gradients, platform, original weights and TensorDict training metadata. Source/framework content is unchanged from the synchronized `ab006` baseline; `measurement-source.json` preserves that comparison. The executable additions are measurement/collection tools and their host tests.

## Reproduction

Use two private environments with matching actual package installations. Use an executable launcher for each repository/runtime which forwards to the original interpreter through:

```bash
<runtime-python> agent/skills/torch-compat-repo-runbook/scripts/component_measure.py \
  --root <jittor-checkout> --repo <torchmetrics-or-tensordict> \
  --checkout <clean-release-checkout> -- "$@"
```

Pass these launchers as `verify_repo.py --oracle-python` and `--shim-python`, selecting the maintained case and `--seed 0 --repeats 3 --no-memory`. Run the maintained runner through each launcher again with `--seed 7 --weights <original-reference.weights.npz>`, saving `<case>.perturb.ref.npz` and `.perturb.shim.npz`. The independent comparison JSON records both raw results and the changed-output check. For TensorDict, run `training_probe.py` through both launchers with `--seed 0 --steps 3 --learning-rate .001 --weights <original-reference.weights.npz>` and stems `<case>.train.ref` / `.train.shim`. Run CPU first with `CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 nvcc_path='' JT_BUILD_NVCC_PATH=''`; configure the allocated GPU and an independent cache for CUDA. Set `JITTOR_ECOSYSTEM_TF32=0 JITTOR_ECOSYSTEM_CUDNN_BENCHMARK=0`, offline flags and equal thread settings in both stages.

Collect using the reusable skill tool, which re-enumerates both upstreams and invokes the actual shared `audit_coverage.scorecard`:

```bash
python agent/skills/torch-compat-repo-runbook/scripts/collect_component_scorecard.py \
  --upstream torchmetrics=<clean-metrics-checkout> \
  --upstream tensordict=<clean-tensordict-checkout> \
  --run torchmetrics=<cpu-metrics-run> --run torchmetrics=<cuda-metrics-run> \
  --run tensordict=<cpu-tensordict-run> --run tensordict=<cuda-tensordict-run> \
  --platform cpu=<actual-cpu-platform.json> --platform cuda=<actual-cuda-platform.json> \
  --baseline 90ee18a9916acda9b5e86ca665f09a231a41ced1 \
  --report docs/results/2026-10-15-metrics-tensordict-scorecards.md \
  --command '<complete two-runtime commands or retained task driver invocation>' \
  --output <outside-checkout-scorecard.json>
```

The first CUDA attempt stopped before candidate computation: strict wheel validation found cu128 runtime metadata ahead of the required cu122 stack. It remains intact under `attempts/cuda-wheel-mismatch/`. A separate task-owned CUDA shim interpreter with the existing cu122 directory first in its private search path resolved the environment; strict validation remained enabled. The CPU interpreter and all CPU measurements were preserved.

Validation of the reusable tools: 8 focused host tests; after fixing the new test's collection-time loader by moving it to a fixture, the relevant contract/tool run passed 42 tests and the complete structure run passed 1486 with 6 skips. Layout, whitespace and Ruff passed. No allowlist was broadened. The original structure failure is retained. After adding this report, the focused documentation/tool run passed 29 tests; the two remaining failures are its central results-toctree registration and regenerated root manifest. The integration owner maintains those shared files. Thus the full 1486/6 result describes the tool source before adding the report, not a claim that the unregistered documentation tree is green.

Raw arrays, original logs including failed attempts, full commands/environment recipes, platform JSON, inventories and hashes are unversioned under `$JITTOR_LAB_ROOT/_state/metric-tensordict-scorecards/20261015/`. No old raw file is rewritten.

## Main-branch integration verification

Integration synchronized `02c14ff7edde1203f6bb54230e6d0f8f0686e5e9`. Independent review rehashed all 139 sealed raw files and six delivered source files, recomputed all 70 numerical comparisons, checked the 20 measurement processes and complete installed-package bindings, and reran the shared auditor: four accepted records, zero rejected. The historical measurement baseline remains `90ee18a9916acda9b5e86ca665f09a231a41ced1`.

A negative experiment on copied TensorDict artifacts exposed a collector gap: both training runtimes could agree on a replacement initial-weight snapshot different from the baseline. The collector now requires each phase's actual weight-file SHA256 to equal its metadata and the original baseline snapshot. That same negative experiment is rejected after the fix; fresh collection of copies of all four original positive records still yields four accepted and zero rejected. The 139 sealed originals remain unchanged. Four focused regression cases cover valid, replacement, modified and missing snapshots; the related host/contract suite passed 99 tests.

On the integrated main worktree, the maintained Metrics and TensorDict gates passed on actual CPU (2 passed, zero skipped) and CUDA (2 passed, zero skipped). These checks validate composition with newer main-branch changes; they do not change the source identity of the earlier scorecard measurements. Integration logs and independent audit results are unversioned under `$JITTOR_LAB_ROOT/_state/metric-scorecard-integration-02c/`.
