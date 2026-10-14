# Real VERL registry components on CPU and CUDA

- Status: Reproduced tiny component correctness and three-step training trajectories; no trainer or performance grade.
- Measurement baseline: `b367a25ff89b2496d4c34c57fe0a4b129b781dcd`; code includes profiler enums, required-gradient assertions, real multi-site dependency discovery, per-library missing-dependency enforcement and fail-closed runner/array validation.
- Upstream: clean VERL `e5a627cf55c94d85406556854c2c1a22db4ff5e7`, version `0.10.0.dev`.
- Maintainer: Jittor core / Torch compatibility maintainers.
- Review when: VERL registries/config/loss layout, dependency versions, case builders, precision policy or device backend changes.

This report executes the real registry functions from the pinned upstream source, using the same
builders under separate independent PyTorch and Jittor interpreters. No adapter, replacement registry,
mock, copied algorithm, downstream source patch or model download participates in the measurements.
The source checkout is clean and its runtime import origin is checked against the inventory.

## Exact scope and registry coverage

| Case | Actual upstream component | Output / parameter grads / input grads | Three-step trainable scope |
| --- | --- | --- | --- |
| `verl_ppo_gae_vanilla` | `get_adv_estimator_fn("gae")` + `get_policy_loss_fn("vanilla")` | 49 values (loss + all advantages/returns), 2 grads, `features` grad | `policy.weight`, `policy.bias`; 21 arrays |
| `verl_distillation_k3` | `get_distillation_loss_fn("k3")` | masked 4×6 token loss, 2 student grads, `features` and `teacher` grads | `student.weight`, `student.bias`; 24 arrays |

PPO uses a 3→1 Linear policy, a 4×6×3 feature tensor and varied response-mask lengths. GAE preserves its
upstream `no_grad` contract: rewards/values are detached, advantages and returns are numerically
compared, and the complete feature gradient comes from the actual policy branch. This is not a claim
that GAE is differentiable. The packed k3 case uses 28×3 student features and 28×1 teacher logits,
converted by the upstream packed-loss path into four six-token responses. The TensorDict has no common
batch dimension because packed tokens and per-sequence metadata have different leading dimensions.
It is a component protocol, not a validated trainer batch contract. Both student and teacher input
gradients are compared. `DistillationConfig(enabled=False)` avoids validating or starting a distillation engine; the registered k3 loss is called directly, and its nonzero student/teacher gradients are required.

The scorecard retains case, full registry refs, tiny shape and trainable parameter names in each
`verified_scenarios` row. The matrix applies separately to CPU and CUDA:

| Upstream axis | Verified members | Verified / automatic denominator |
| --- | --- | --- |
| `advantage_estimator` | `gae`, in the GAE+vanilla policy scenario above | 1 / 14 |
| `policy_loss` | `vanilla`, in that same scenario | 1 / 12 |
| `distillation_loss` | `k3`, in the packed student/teacher scenario | 1 / 8 |
| Hydra `entrypoint` | none | 0 / 5 |

Those denominators are derived from all upstream registrations at the stated SHA, not from CASES.
Other aliases of the same function are not counted automatically. The highest established level is
L3 within the named tiny component training scenarios. Ray workers, rollout, Hydra training entrypoints,
actual FSDP2 execution, full PPO, model-level distillation, NPU, ROCm and Corex are not validated here.
An ActorConfig `strategy="fsdp2"` value merely configures the selected loss; it does not establish
execution of a distributed engine. No L4/L5 or full-library claim follows.

## Fixed platform and protocol

| Field | CPU | CUDA |
| --- | --- | --- |
| OS / Python | Linux 6.1, glibc 2.34 / Python 3.11.16 | same |
| CPU / accelerator | AMD EPYC 9554; accelerators explicitly masked | one allocated NVIDIA RTX 4090 |
| Independent oracle | PyTorch 2.11.0+cu128, genuine binary `_C` | same, CUDA 12.8 |
| Candidate | Jittor 2.0.0 + source Torch compatibility layer | same, real NVCC 12.2.140 / driver 580.178.04 |
| Dependencies | NumPy 2.2.6, Transformers 5.5.3, TensorDict 0.10.0, Ray 2.54.1 | same real package sources |
| Precision | TF32 OFF, matmul `highest`, cuDNN benchmark OFF | TF32 matmul/cuDNN ON, matmul `high`, benchmark OFF |
| Threads / fallback | two runtime threads, identical affinity; policy `error`, count 0 | same policy; CUDA dispatch observed, count 0 |

The VERL revision declares NumPy >=2 and Transformers >=5.5.3, !=5.6.0, <5.13. Separate environments
honor those requirements; the established NumPy 1 / Transformers 4 ecosystem session remains separate.
Raw platform artifacts include compiler output, visible package metadata, real oracle binary origin,
allocated device mapping, actual precision, dependency versions and runtime conditions.

Inputs are offline, base seed 0. Additional seeds 7 and 19 reuse the complete base reference weights;
outputs must change and both runtimes must produce every requested parameter/input gradient. The
three-step protocol uses public `training_probe.py`, seed 0, SGD learning rate 0.001 and fresh fixed
reference initial weights. Step inputs use deterministic seeds 1, 2 and 3 in the same order on both runtimes; this compares an update trajectory, not convergence. Every step saves the output, scalar loss, complete gradients and updated
parameters; both trainable parameters change and there are no frozen parameters. All compared arrays
are finite. Correctness tolerances are absolute 2e-3 / relative 1e-2; actual errors below are much smaller.

## Numerical and gate results

| Device / case | Output + all gradients, max abs across seeds 0/7/19 | Three-step arrays / max abs |
| --- | --- | --- |
| CPU / `verl_ppo_gae_vanilla` | 2.38418579e-07 | 21 / 2.38418579e-07 |
| CPU / `verl_distillation_k3` | 5.96046448e-07 | 24 / 7.15255737e-07 |
| CUDA / `verl_ppo_gae_vanilla` | 1.1920929e-07 | 21 / 2.38418579e-07 |
| CUDA / `verl_distillation_k3` | 2.98023224e-07 | 24 / 1.1920929e-07 |

The dedicated pytest class actually executes both VERL nodes on each device, with explicit required
gradient keys. The CPU run also exercises the two profiler-enum contracts; the CUDA run additionally
executes four real profiler/allocator checks. JUnit evidence rejects any skipped VERL node. The tests
were run directly in the fully provisioned interpreters; nox wrapper session itself was not executed.
The nox session's command selection, required flags, dependency-version preflight, CPU masking and
explicit CUDA allocation are covered by host contracts.

- Public scorecard: four accepted records (two components × CPU/CUDA), zero rejected; each retains tiny shape, full registry references and named trainable scope. Both devices produce the matrix above.
- Actual dedicated parity/profiler nodes: CPU **4 passed**, CUDA **8 passed**, zero skipped. Both runs set `JITTOR_REQUIRE_VERL=1`, require an independent real Torch oracle and actual execution; the saved command evidence includes these flags. JUnit confirms both VERL test methods executed.
- Shared dependency, runner-protocol, array and required-gradient contracts, including the four existing noisy-marker cases: **54 passed**. CPU core on the final code baseline: **322 passed, 42 skipped, 1 xfailed**. Lint, diff, layout and internal Markdown link checks also passed.
- A negative audit uses copies of the final records/platform declarations, falsely relabels CUDA TF32 ON/high as OFF/highest, and regenerates the copy's platform artifact hash. Both CUDA records are rejected with `Platform/measurement precision mismatch`; both untouched CPU records remain accepted. Original raw arrays and measurement metadata are unchanged.

## Real breakpoints and fixes

1. Upstream profiler configuration enumerates `torch.profiler.ProfilerActivity.__members__`. The old
   compatibility constants lacked enum discovery. The fix belongs to the profiler facade and exposes
   matching names/values and ProfilerAction identity; enumeration is not a capability claim for every
   listed backend.
2. The shared harness restricted discovery to the Transformers directory when `PACKAGE_SITE` existed,
   excluding real dependencies in other source checkouts/sites. It now searches that explicit site and
   the interpreter's actual import path. A missing dependency fails only when its own import-name
   REQUIRE flag is set; requiring an available Transformers does not make missing optional PEFT fail.
   The dedicated VERL gate additionally refuses any missing dependency instead of skipping.
3. The shared comparison now requires declared gradient keys on both sides, so matching omissions
   cannot produce agreement. A host negative test covers oracle-only, candidate-only and joint omission.
   It also closes both NPZ archives before deleting the temporary directory, avoiding NFS open-file
   cleanup failure.

4. The shared harness now rejects a nonzero process exit before parsing a success marker, rejects a marker for another or missing case, and requires symmetric tensor keys, exact shapes and finite values on both sides. Before the fix, host counterexamples demonstrated false agreement for nonzero exits, extra candidate gradients, broadcastable shapes and NaN gradients. Both runtime branches and malformed-array variants are covered; the numerical thresholds and mathematics are unchanged.

## Reproduction and artifact binding

Use the [VERL runbook](../../agent/skills/verl-torch-compat/SKILL.md) for environment prerequisites and
`nox -s ecosystem_verl`. The source-only inventory and fixed-platform scorecard contract are documented
in [downstream scorecards](../development/downstream-scorecards.md).

```bash
python agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py \
  --repo verl --checkout "$VERL_CHECKOUT" --output "$STATE/verl-inventory.json"
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 "$SHIM_PY" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo verl --device cpu --oracle-python "$ORACLE_PY" --shim-python "$SHIM_PY" \
  --seed 0 --repeats 3 --no-memory --out "$STATE/cpu"
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TORCH_SHIM=1 REAL_TORCH_PYTHON="$ORACLE_PY" JITTOR_TEST_DEVICES=cpu \
  JITTOR_REQUIRE_VERL=1 JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TEST_REQUIRE_EXECUTION=1 "$SHIM_PY" -m pytest -q \
  compat/tests/torch/test_ecosystem_verl.py
python agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py scorecard \
  --upstream "verl=$VERL_CHECKOUT" --evidence "$STATE/scorecard-evidence.json" \
  --baseline b367a25ff89b2496d4c34c57fe0a4b129b781dcd --output "$STATE/scorecard.json"
```

Here `STATE` is an external directory below `$JITTOR_LAB_ROOT/_state/`; each device uses its own
compilation cache, and unit gates and numeric probes use separate caches. Repeat the same protocol for
CUDA in separate processes with an explicitly allocated visible device and a real toolkit. The
manifest retains exact commands for both perturbation seeds, every public three-step training command,
source inventories, driver/source hashes, raw arrays, platform and gate JUnit files.

The external run root is `$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/verl-components-20261015-final/`.
The following SHA256 values identify its final evidence; the manifest hashes each individual raw artifact.

| External artifact | SHA256 |
| --- | --- |
| `scorecard-evidence.json` | `bc15b70a10ab9950b3eae49beb76ef9b0f5b652f39aedd2142778b721501c0f6` |
| `scorecard.json` | `92ffe78afb57475d06cca47f58905792ad04392b63cf681da49a743ce00093fe` |
| `verl-inventory.json` | `83065316c7b8d7c8883876321d5a33d743bcdedb43181776efb8818c9eb9eba8` |
| `platform-cpu.json` | `3a2ddd74eab7dc872e601f5a3dc8e689327b248de7ac6cb098cefb03fbc391d4` |
| `platform-cuda.json` | `1d8145ee5775021e6d47e7582f17ca5e2c3b9b73f6d59c25775b51702c6f82f7` |
| `negative-precision-scorecard.json` | `0d75d3bab17e6368e334dae29a37833a6776713fbb1416a9ea8ad6a3262406ad` |

Raw data, logs, installed dependencies and compiler caches remain outside the repository. Intermediate
prototype, cache-bootstrap failures and interrupted baselines are archived separately and are not accepted scorecard records.
Tiny timing samples and skipped memory passes do not establish a speed or whole-process memory grade;
this report makes no performance claim.
