# Installed-source VERL component training on CPU and CUDA

- Status: Both original tiny component protocols pass complete gradients, two input perturbations and three SGD steps on CPU and CUDA; bounded component L3.
- Measurement source: `73333a677dae984a3f577afb7d1586aecac5f490`; version-proof schema prerequisite `dbea21762d2df71b7c67dcf2ba9abb4f642944da`. This report is a later documentation commit.
- Upstream: clean VERL `e5a627cf55c94d85406556854c2c1a22db4ff5e7`, module version `0.10.0.dev`, actual distribution version `0.10.0.dev0`.
- Maintainers: Jittor Torch compatibility and downstream validation maintainers.
- Recheck when: upstream registries, installed payload, interpreters/providers, precision, component builders, measurement tools or training protocol change.

This fresh run retains the complete scope of the [earlier VERL protocol](2026-10-15-verl-registered-component-oracles.md)
and adds actual installed-package proof. Historical measurements remain under their original source identity;
no historical metadata was backfilled. No adapter, copied algorithm, upstream patch or model download was used.

## Actual source and installed package

The unmodified clean checkout was archived and built into `verl-0.10.0.dev0-py3-none-any.whl`, SHA256
`3bd60a64f76de6ff889951b1e994b1d08fa8aaff34b3b22da49a23e61006af0c`.
This is a self-built wheel from the pinned source, not an official PyPI wheel. Both interpreters import
that installed wheel payload. The inventory records 360 Python sources and 462 tracked package files;
the wheel ships 413 package files, including all 360 Python files and 53 data files. The remaining 49
tracked non-Python files are not represented as installed. Every runtime phase binds all source hashes,
all 413 actual installed payload hashes, the clean upstream SHA, wheel SHA and actual distribution metadata.

The module's literal `.dev` version remains unchanged. The wheel METADATA legitimately normalizes it
to `.dev0`. The shared proof retains both strings and accepts this difference only through strict
`packaging.Version` equality, complete wheel proof and explicit `distribution_version` in each runtime
and phase. Missing or unequal versions fail. Literal-equal and existing SCM proofs keep their prior
contracts; source-only inventory/audit entrypoints do not eagerly import packaging.

## Scope, gradients and frozen budgets

| Case | Real registered execution and shapes | Complete gradient scope | Three-step fields |
| --- | --- | --- | --- |
| `verl_ppo_gae_vanilla` | `gae` + `vanilla`; 3→1 Linear; features 4×6×3; 49-value loss/advantages/returns output | `policy.weight` (1×3), `policy.bias` (1), features (4×6×3) | 21 |
| `verl_distillation_k3` | registered packed `k3`; 28×3 features and 28×1 teacher; 4×6 token output | `student.weight` (1×3), `student.bias` (1), features and teacher | 24 |

GAE keeps its actual `no_grad` semantics. Its advantages and returns are compared; requested input
gradients come from the policy branch. Packed TensorDict inputs deliberately have no common batch
dimension. The k3 configuration does not start a distillation engine. Both teacher and student-side
input gradients are checked. Each runtime must produce the exact complete keys, shapes, FP32 dtypes,
finite values and a nonzero tensor for every requested gradient. Missing, extra, zero or broadcastable
but unequal-shaped fields fail.

Base seed 0 and perturbations 7/19 use identical initial weights. CUDA also reuses the CPU reference
initial weights. Perturbed outputs must change. Training uses seed 0, deterministic step input seeds
1/2/3, three real SGD updates at learning rate 0.001, and both named parameters remain trainable and
change. Every step compares output, loss, all parameter/input gradients and updated parameters.
The original stage budgets are preserved:

- Base seed 0 uses `max_abs_error / max(reference_field_max, gradient_block_max * 0.001, 1e-6)` strictly below 0.002 for output or 0.01 for gradients. The gradient block floor applies only to gradient fields.
- Perturbations and training use the original per-field sum: `max_abs_error <= 0.002 + 0.01 * max(reference_field_max, 1e-8)`.

There is no model-wide scaling, parameter-count dilution, convergence claim or changed tolerance.

## Platform, independent oracle and actual device

| Property | CPU | CUDA |
| --- | --- | --- |
| Runtime | Python 3.11; NumPy 2.2.6; Transformers 5.5.3; TensorDict 0.10.0; Ray 2.54.1 | same |
| Oracle | Genuine PyTorch 2.11.0+cu128 with its real binary extension; accelerators masked | same, actual CUDA 12.8 provider |
| Candidate | Jittor native CPU through Torch compatibility | real allocated NVIDIA RTX 4090; NVCC 12.2.140; driver 580.178.04 |
| Precision | FP32, TF32 OFF, matmul `highest` | FP32, TF32 matmul/cuDNN ON, matmul `high` |
| Conditions | Two runtime threads, cuDNN benchmark OFF, strict native kernel math, offline | same |
| Candidate execution | CPU dispatch and CPU physical storage | CUDA dispatch and device physical storage |

Candidate fallback policy is `error` and the actual count is zero. The observer records 304 tensor
observations across both runtimes and devices, including every exported output, gradient and training
parameter snapshot. Candidate observations check physical location after synchronization and dispatch;
they do not move tensors to a device to create apparent residency. CPU and CUDA benchmark caches and
unit-test caches are separate. No L4 or speed conclusion is made.

Two initial provider setup failures are retained separately. The old shared reference installation
was missing `_strobelight/compile_time_profiler.py`, before any numeric run. No shared package was
modified. A private complete official PyTorch wheel was then installed; its first import selected an
incompatible CUDA 12.2 runtime. Giving that private prefix the standard adjacent, real NVIDIA CUDA 12.8
package layout fixed the provider selection without changing Torch wheel bytes. The successful run's
private wheel SHA256 is `c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`;
11,841 original wheel files were verified, including binary `_C` SHA256
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`.
All 2,227 Python source paths match the wheel's exact source set. All 17 actual mapped NVIDIA libraries
were hashed. A post-formal check rehashed the original wheel payload, mapped providers and measurement
sources unchanged. These failures are environment failures, not passed numeric measurements.

## Results and maintained gates

| Device / component | Compared array pairs across all phases | Maximum absolute error | Maximum base normalized error |
| --- | --- | --- | --- |
| CPU / GAE + vanilla | 33 | 2.38418579e-7 | 3.65876907e-6 |
| CPU / packed k3 | 39 | 7.15255737e-7 | 5.00747165e-7 |
| CUDA / GAE + vanilla | 33 | 2.38418579e-7 | 0 |
| CUDA / packed k3 | 39 | 2.98023224e-7 | 2.98727005e-7 |

Both maintained nodes run the full three seeds and three-step training, with required dependency,
independent interpreter, complete source/wheel, device and zero-fallback checks. CPU nox actually
finished **2 passed, 0 skipped** (375.98 s pytest); CUDA nox actually finished **2 passed, 0 skipped**
(521.53 s pytest). Cold nox sessions took approximately 12 and 14 minutes respectively. These are
opt-in maintained gates; they do not establish nightly execution.

The common scorecard accepts all four records, with zero rejected or blocked records, after recomputing
all 144 array pairs and checking all 32 runtime-phase metadata files.
The automatic registry denominator, separately on each device, remains advantage estimators **1/14**
(`gae`), policy losses **1/12** (`vanilla`), distillation losses **1/8** (`k3`), Hydra entrypoints **0/5**.
Only the named tiny component scenarios receive L3. Ray/rollout workers, full PPO trainers, Hydra
training, model-level distillation, other platforms and whole-library coverage remain unvalidated.

Source validation: 97 focused host contracts passed; CPU core **322 passed, 42 skipped, 1 xfailed**;
`tests/structure` **1510 passed, 4 skipped**; lint and layout passed. Host negative tests cover strict
version normalization, both runtime/phase proofs, complete payload, missing/extra tensor fields,
shapes/dtypes/nonfinite/zero gradients, wrong marker/case/nonzero exit and wrong physical residency.
Thirteen additional mutations of copies of the actual metadata were rejected. Original evidence
was unchanged. A later isolated documentation check had 14 passes and one pending central toctree
registration for this new report; the integrator owns that index and the central manifest. An initial
documentation-check launch without the dedicated cache environment was stopped and retained separately;
it is not counted as a passed validation run.

## Reproduce and inspect evidence

The maintained session uses preconfigured independent interpreters and a clean source-built wheel;
it does not install or download downstream packages. Configure each interpreter to import the actual
installed VERL payload and its declared NumPy 2 / Transformers 5 dependencies first.

```bash
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_ECOSYSTEM_PACKAGE_SITE="$VERL_SITE" JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE="$VERL_SITE" \
  JITTOR_VERL_UPSTREAM="$VERL_CHECKOUT" JITTOR_VERL_WHEEL="$VERL_WHEEL" \
  JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_verl_training
# CUDA: use a fresh cache, JITTOR_TEST_DEVICES=cuda, explicit CUDA_VISIBLE_DEVICES,
# and nvcc_path pointing to the real allocated CUDA toolchain.
```

Exact environment names and actual command arrays are also preserved in the command artifacts;
use the session's declared required variables if reproducing on another machine. Actual nodes:
`compat/tests/torch/test_ecosystem_verl_training.py::EcosystemVERLTraining::test_verl_ppo_gae_vanilla`
and `::test_verl_distillation_k3`.

Raw artifacts live outside the repository at
`$JITTOR_LAB_ROOT/_state/verl-source-bound/20261015/`. Successful data is under
`formal-73333a677dae-provider-v3/{cpu,cuda}/`; the earlier directories preserve the two setup failures.
`clean-source-freeze-v1.json` binds the measurement sources. `scorecard-v1.json` contains regenerated
inventory, records and auditor results. `verl-sealed-raw-manifest-v1.json` fixes 240 files (127,073,470
bytes), SHA256 `ea4a86547d4f3f96a97d5010c746fc425ed6ad0dc016b777c057f09ec6b4b32e`.
The final evidence manifest additionally binds the final scorecard and documentation identity.
All original arrays, metadata, source proofs and preserved failure logs remain under their actual
source and provider identities.
