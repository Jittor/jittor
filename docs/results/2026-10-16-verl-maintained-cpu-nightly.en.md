# VERL maintained CPU nightly and independent installation prefixes

- Status: Nightly source wiring implemented; the equivalent maintained CPU session passed locally. No GitHub hosted workflow run is claimed.
- Date: 2026-10-16
- Baseline commit: fresh environment provisioning `d80422c89d29f2f6dd050847a2a5b0fda72194b5`; successful measured source `302900315cf3c4bd9449f166d1e1632feacfea94`. This report is a later documentation change.
- Owner: Jittor Torch compatibility maintainers
- Review when: the VERL pin, dependency profile, installed-source proof, component cases, numerical budgets, workflow or nox session changes.

## Maintained entry and scope

The existing maintained CPU matrix in `.github/workflows/ecosystem.yml` now includes `verl`,
alongside PEFT and Swift; the older general ecosystem job remains. It calls the real
`tools/provision_ecosystem_nightly.py --library verl --run` entry, rather than relying on a
developer's package directories. Mutable checkouts, wheels, three environments, HOME, temporary
files, caches and logs live under the explicit external state root or CI `RUNNER_TEMP`.

The two required nodes remain
`compat/tests/torch/test_ecosystem_verl_training.py::EcosystemVERLTraining::test_verl_distillation_k3`
and `test_verl_ppo_gae_vanilla` in that class. They run actual upstream registered k3,
GAE and vanilla policy-loss functions. This is component training, not a Ray worker, rollout,
Hydra entrypoint, CPU/backend extra or full VERL trainer. No algorithm, gradient request,
training step or numerical threshold was reduced.

The upstream is the clean official VERL source at
`e5a627cf55c94d85406556854c2c1a22db4ff5e7`. The inventory is regenerated from that checkout:
14 advantage estimators, 12 policy losses, 8 distillation losses and 5 Hydra entrypoints.
The bounded CPU scenarios exercise `gae` **1/14**, `vanilla` **1/12** and `k3` **1/8**;
entrypoints remain **0/5**. These axes are independent and are not summed into a library percentage.

## Fresh real environments and the reproduced failure

Provisioning at `d80422c89` created new candidate, oracle and controller environments with
133, 131 and 17 installed distributions, respectively. The runtime profile uses official
PyTorch **2.11.0+cpu**, NumPy **2.2.6**, Transformers **5.5.3**, Ray **2.54.1**,
TensorDict **0.10.0**, PEFT **0.17.1**, Accelerate **1.10.1**, Datasets **3.6.0** and
PyArrow **19.0.1**. Complete base dependency closures are resolved by pip, retained as locks
and checked in all three environments. Controller build tools remain separate from runtime
setuptools constraints. Optional full-trainer CPU/backend extras are not enabled.

The actual oracle loaded its binary Torch extension without the shim and reported CUDA
unavailable. NumPy 2 / PyArrow / Datasets interoperability was executed. CUDA was explicitly
hidden; the candidate selected the native CPU backend and fallback policy `error`, count zero.
The source-built VERL wheel is `verl-0.10.0.dev0-py3-none-any.whl`, SHA256
`609f257d53ea96dd31861af485f2dee9531abbcb9d49b2c69484871d716fba9b`.
Source/module version `0.10.0.dev` and actual distribution version `0.10.0.dev0` remain
unchanged and are checked through the existing PEP440 and full installed-wheel proof.

The first real maintained attempt failed **both nodes**, before input perturbations or
training, because the comparator required identical dependency `origin` strings. The
separate prefixes had identical actual installed payloads but different absolute paths.
Its **2 failed / 0 skipped, 126.01 s** result, original code and raw files remain sealed;
it is not a training success. A separate base-only diagnostic checked nine tensor pairs.

The fix first rehashes each of the five actual dependency package directories. The recorded
origin must be a real absolute file. VERL and any existing per-dependency distribution
binding must agree with their own side's origin root, version and complete installed payload.
Only then does cross-runtime identity comparison omit `origin` and `package` locations.
All source, wheel, distribution version, payload and newly introduced identity fields remain
checked. Raw metadata preserves the actual locations. Host negatives reject absent or invalid
origins, wrong-prefix bindings, changed real or recorded payloads, forged source/wheel/METADATA,
scope changes and unmatched new fields; they are metadata contracts, not numerical evidence.

## Successful new measurement

At clean source `302900315`, the same privately provisioned environments ran a **new** complete
maintained session: **2 passed, 0 skipped, 205.12 s**. Setup identity remains `d80422c89`;
the provisioner and dependency installation code did not change between these commits.
The existing task-owned benchmark cache was reused normally, without copying binaries or
claiming another cold build. Its first compilation had already run serially. Unit caches
are separate. All three environments passed `pip check` again after the new measurement.

| Required CPU node | Parameters / requested input gradients | Array pairs across all phases | Largest absolute difference |
| --- | --- | ---: | ---: |
| packed k3 distillation | student weight+bias / features+teacher | 39 | `7.152557373046875e-7` |
| GAE + vanilla PPO loss | policy weight+bias / features | 33 | `2.384185791015625e-7` |

Each node runs fixed initial weights with seeds **0, 7 and 19**, checks changed perturbation
outputs, then performs the original **three SGD updates**, learning rate **0.001**. All
requested parameter and input gradients are checked; both parameters must actually update.
The base gate keeps its original normalized output **0.002** / gradient **0.01** limits and
gradient-block floor. Perturbation and training fields retain their original
`max_abs <= 0.002 + 0.01 * max(reference_abs_max, 1e-8)` criterion. The table reports observed
errors and does not define a replacement budget.

The harness now retains **36 files, 13,426,995 bytes** for these two nodes, including every
side's NPZ and actual-source metadata, initial weights and two `gate-complete.json` records.
A record is written only after all four complete phases pass; failed directories retain their
raw data without a success marker. A post-run rechecker revalidated every bound file hash,
each side's installed source and **all 72 tensor pairs**, including complete training fields.
Actual descendant process records retain a strict environment allowlist, including private
HOME/JITTOR_HOME/TMPDIR and the hidden-CUDA selection; no complete environment dump is emitted.

## Reproduction and checks

From a clean checkout, with a healthy Python 3.11 and an external state root:

```bash
python tools/provision_ecosystem_nightly.py --library verl \
  --state-root "$STATE/fresh-verl" --run
```

Read-only exact source and wheel inputs can be supplied using `--source-cache` and
`--wheel-cache`; dependency resolution, source verification, actual installation and pip checks
still execute. The ordinary `python -m nox -s ecosystem_verl_training` entry remains reusable
with explicitly configured independent interpreters and installed package sites, as documented
in the [VERL skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/verl-torch-compat/SKILL.md).

The nox session allocates a new evidence directory per invocation outside disposable session
scratch. `JITTOR_VERL_EVIDENCE_DIR` can select another external output root; repeated case
outputs are rejected rather than overwritten. Failure after the final comparison is covered
by a negative test that checks retained raw files and the absence of a success marker.

At `302900315`, relevant host checks passed **85 / 0 skipped**, the child-process contract
passed **8**, canonical Ruff and layout checks passed, and complete structure plus compat
structure passed **1,624 / 4 skipped** in **223.09 s**. The first structure run had one new
test-fixture interpreter-spelling failure (**1,623 passed / 4 skipped / 1 failed**); it was
fixed before the successful run. Two intervening structure launches exited with the prescribed
`jit_utils` rebuild status before executing tests; the same correctly configured command then
passed. These unsuccessful invocations remain in the evidence.

After every measurement and structure process terminated, origin `9c6ee1dde` was integrated
without replacing its R33 changes. The VERL helper and measured component source bytes remain
unchanged. Host checks again passed **85 / 0 skipped**, with canonical Ruff and layout passing;
this documentation sync does not relabel the `302900315` numerical or structure results.

Unversioned evidence lives under `$JITTOR_LAB_ROOT/_state/ecosystem-verl-nightly/20261015/`:
`verl-v1/provision.json`, its locks/logs and `upstream-proof.json` preserve fresh setup;
`failed-v1-manifest.json` preserves the first real failure; `verl-v2/source.json` binds all
2,882 tracked source entries including Git symlink target text; `verl-v2/measurements/`,
`recomputed-comparisons.json`, actual process records and `unit-full/` retain the new checks.
The final `manifest.json` seals delivery evidence. Historical `73333a677` CPU/CUDA scorecards
remain unchanged. This new CPU maintained run is not relabeled as a new common scorecard,
CUDA run, whole-library support, performance result or an actual GitHub schedule execution.

## Central integration review (2026-10-16)

The integration owner reviewed the workflow, provisioner, maintained harness and
strict per-prefix identity changes, then integrated them at `902e733ed`. Original
setup and measurement remain `d80422c89` and `302900315`; this integration does not
relabel those executions. Independent replay without importing Jittor rechecked
**72 complete tensor pairs**, all eight phase seals and both actual installed
source bindings. All **2882 original Git blobs**, including symlink target text,
match the frozen measurement snapshot. The complete delivery seal matches all
**149 files / 20273226 bytes**, manifest SHA256
`284bab3ec2aafec0d5bfcfd82d86c50d3723385d11f8f106f464dcb84d19d0d7`.
All three actual candidate, oracle and controller environments also passed an
independent `pip check` after delivery.

The integrated main tree passes **74 focused host contracts**, core **322 passed /
42 skipped / one existing expected failure**, and complete structure plus compat
structure **1622 passed / six skipped** (208.99 s). The six actual structure skips
include two interpreter installation-resolution checks; they are not recast as
successful installed-package checks. Changed-Python Ruff, layout, generated
manifest and diff checks pass. Verification scripts, raw rechecks and terminal
logs are unversioned under
`$JITTOR_LAB_ROOT/_state/verl-nightly-root-integration-13a/20261016/`.
No old numerical packet or delivery seal was modified. All full-trainer, hosted
schedule, new-CUDA, shared-scorecard and performance boundaries above remain.
