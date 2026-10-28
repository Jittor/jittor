# Downstream coverage matrix and scorecard protocol

- Status: maintainer acceptance protocol; template placeholders are not validation results.
- Review date: 2026-10-14.
- Baseline: `cded3102037edc3663b8f2054a6ebc06da6ebfd2`, plus the commit containing this file.
- Owner: Torch compatibility and downstream ecosystem maintainers.
- Review when: the upstream registries, ecosystem runners, device fallback, graph replay,
  timing or process-memory measurement change.

Coverage breadth and L1–L5 depth are orthogonal. The highest level reached for a scenario must
have corresponding evidence; several successful imports do not add up to training support. This
protocol applies to every downstream; NPU-specific driver/operator investigation keeps using
the [NPU templates](npu-validation-templates.en.md); timing details follow the
[benchmarking specification](../performance/benchmarking.en.md).

## Upstream denominators

The entry point is [`registry_inventory.py`](../../agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py),
which reads only Git-tracked sources and imports neither Torch nor the downstream library. It
must run on a clean checkout; record the full SHA, remote, source version and the SHA256 of the
registration source files; re-enumerate for every scorecard and never accept a hand-written
denominator.

| Repository | Automatic source and breadth axes | What cannot be extrapolated |
| --- | --- | --- |
| Transformers | `MODEL_MAPPING_NAMES` model families; `MODEL_*_MAPPING_NAMES` task mappings | one configuration/task does not prove every checkpoint, MoE, multimodal or long-context variant of that family |
| PEFT | `PeftType` methods; `TaskType` tasks | type registration does not prove an implementation; configuration variants such as DoRA must be listed as separate concrete specifications, and model families reference the Transformers denominator separately |
| Diffusers | every optional branch of `pipelines._import_structure`: pipeline module families and APIs; AutoPipeline families/tasks listed separately; every public export of `models._import_structure` listed separately as `model_api` | the model export axis includes helpers, mixins, enums, functions and Flax, so it is not a model-family count; a single UNet module does not prove a whole pipeline; an AutoPipeline subset must not pass for the whole library |
| ms-swift | `register_model(ModelMeta)` model families; `SWIFT_MAPPING` tuners; ordinary/Megatron trainers and the CLI; the `task_type`/`rlhf_type` Literals | the ordinary and Megatron axes are separate; one tiny LoRA does not prove full-parameter/distributed support or every trainer method |
| verl | advantage-estimator, policy-loss and distillation-loss registrations; Hydra entry points | the entry points include training, evaluation and generation; registration does not prove a complete PPO/GRPO pipeline or every model |

What is enumerated here is **source declarations**, including conditional branches for optional
dependencies. Not installing a dependency must not shrink the denominator. When upstream adds a
new dynamic registration mechanism, or the parser fails on an unknown layout, update the parser
and its negative tests first, then give coverage numbers; never silently drop items that cannot
be parsed.

```bash
scripts=agent/skills/torch-compat-repo-runbook/scripts
python -S "$scripts/registry_inventory.py" --repo transformers \
  --checkout "$JITTOR_LAB_ROOT/downstream-coverage/upstream/transformers" \
  --output "$JITTOR_LAB_ROOT/_state/downstream-coverage/transformers-registry.json"
python -S "$scripts/audit_coverage.py" audit --json
python -S "$scripts/audit_coverage.py" scorecard \
  --upstream "transformers=$JITTOR_LAB_ROOT/downstream-coverage/upstream/transformers" \
  --upstream "peft=$JITTOR_LAB_ROOT/downstream-coverage/upstream/peft" \
  --evidence "$JITTOR_LAB_ROOT/_state/downstream-coverage/evidence.json" \
  --output "$JITTOR_LAB_ROOT/_state/downstream-coverage/scorecard.json"
```

Without `--evidence` it generates the complete inventory with a **zero verified numerator**,
meaning this run has no acceptable new evidence; it must not be read as "nothing ever passed
historically". Report each axis separately as `X / N @ upstream SHA`; do not add model counts,
task counts and method counts together. core/broad/tail sampling rules follow the
[adaptation workflow](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/downstream-library-adaptation/SKILL.md);
the report must list the scenario count per tier and the unverified cells, and the core matrix
must reach at least L3.

`gate_cases` in the audit records shared model CASES; `dedicated_gates` separately records
dedicated protocol entry points such as the DeepSpeed Engine and its nox session, instead of
disguising them as ordinary model cases. Both only prove that a gate is wired in;
`verified_level` stays empty, and an L level must be granted by actual execution evidence from
the current scorecard.

## Platform block required on every row

| Field | Required content |
| --- | --- |
| Identity | full Jittor commit; full upstream commit/version/remote; the actual downstream version on both sides must equal that source version; for a checkout without a release version, `source@SHA` may be used, with the real wheel and all source hashes proving the actual SCM version as described below |
| Host/device | OS, CPU, card model, card count, visible index mapping; for CPU write zero cards explicitly and disable CUDA/ACL explicitly |
| Software stack | driver, CUDA or CANN, nvcc, gcc, nvidia wheel versions; write `not-applicable` for fields that do not apply |
| Actual flags | `use_cuda`, `use_acl`, graph replay, TF32, HF32; NPU/ROCm and similar may still use Jittor's `use_cuda` scheduling switch, so record the real values |
| oracle | torch/torch_npu version, binary extension source, `is_shim=false`; "another Python" alone is not enough |
| Inputs | seed, `HF_HUB_OFFLINE=1`, tiny/real; model, batch, sequence, dtype, precision, train/infer/decode |
| Sources | in-tree gate node, dated results report, complete reproduction commands; external paths and SHA256 for both sides' outputs, the comparison result and the platform snapshot |

Same-card parity uses two separate processes and never substitutes a Jittor CPU/device
self-comparison for the oracle. CPU① must explicitly disable CUDA; device③ must prove
residency and `fallback_policy=error,count=0`. Missing, skipped, xfailed, environment-blocked
and failed each get their own record, and an all-skip run cannot count as a pass. Platform
snapshots are for reviewing identity and **cannot replace an assertion about the device the
computation ran on**.

### Actual installation source for release versions

A registry-based L2-and-above record cannot enter the numerator just because both version
strings match. The regenerated inventory provides `package_sources` for every Python file of
each target library, plus `package_files` including non-Python files; these fields neither
change the registry axes nor the denominator. Both sides' raw measurements for base, perturb and
L3 training must include `downstream_distribution`, or `downstream_distributions` keyed by
library name, covering the record itself and every `registry_refs.repo`.

Each binding contains `repo`, `version`, `source_sha256` (the inventoried source path) and
`installed_files_sha256` (the path relative to the installed package root). Every required
Python source must match; additionally declared sources may only come from the same inventoried
`package_files`, and the corresponding installed files must match item by item. When the same
library uses both the singular and plural fields at once, the two must be identical.

If the artifacts bind this library's `upstream-wheel` or `<repo>-official-wheel`, the auditor
independently reads the wheel's single METADATA, checks the package name, version and all
inventoried sources, and then requires that the actual installed file list exactly equal the
wheel's complete package contents, including data, C++, CUDA files and generated Python files;
the binding must also include that artifact's `wheel_sha256`. A role name does not mean the
wheel necessarily comes from PyPI; the report must state truthfully whether it was built from
pinned sources. When there is no wheel, any extra installed Python files must also come from
the inventory.

When the source module version and the wheel METADATA differ only by PEP 440 spelling
normalization, keep both original values: `version` remains the literal version of the actual
module/inventory, and the additional `distribution_version` must equal the single METADATA
Version of the original wheel. For example, the VERL source `0.10.0.dev` has a real build
product that declares `0.10.0.dev0`. This branch is allowed only when the independently read
wheel version and the source version are strictly equal under `packaging.version.Version`; base,
both perturbations and training must each explicitly bind the actual `distribution_version` and
the complete wheel payload for every runtime. Missing fields, differing dev/rc/local versions,
wrong metadata or missing installed files are all rejected. The original literal-equality and
SCM branches are unchanged; do not rewrite `__version__` or label a development build as a
formal release. `packaging` is loaded only when this non-literal proof is needed; a missing
dependency fails outright, and the pure source enumeration entry point still works under
`python -S`.

For older measurements that lack these original bindings, the current audit refuses to grant a
coverage numerator; keep their original numerical results and measurement SHA rather than
backfilling historical evidence from today's installation directory. Protocol-type records
keep their own separate strict validators.

### Limited distribution-name branch for the same source

The clean `setup.py` of MMEngine/MMCV declares the official package name and the lite package
name through environment conditions. The inventory adds `distribution_names` and binds the
SHA256 of the tracked `setup.py`, the actual name expressions and **all** finite branches and
conditions; it is not a hand-written alias allowlist. Only literal names and explicit
`os.getenv` equality/inequality conditions are parsed; dynamic names, unknown conditions,
multiple setup files and dictionary expansion are all rejected. The auditor re-parses the
expressions, and a missing branch or a mismatched setup source hash fails.

This kind of source proof must supply the actual wheel. The auditor independently checks that
the single METADATA Name is among the names the source declares, and every runtime/stage's
`downstream_distribution` or plural binding must also carry the verbatim `distribution_name`,
exactly equal to that actual Name; this branch additionally requires an explicit
`distribution_version` exactly equal to the wheel METADATA Version, even when the module
version's literal value is the same. The package's import prefix, repo identity, version and
the verification of the complete Python/non-Python payload stay unchanged. A missing field,
only the repo name, an undeclared name or an incomplete payload is rejected. A real installation
of `mmcv-lite` can prove the corresponding pure-Python scenario; it does not prove that
`mmcv._ext`, `mmcv.ops` or the compiled operators exist; do not modify METADATA or install an
empty extension to pass for the complete package.

### Source without a release version and SCM wheels

If the regenerated inventory version is exactly `source@<上游完整 SHA>`, the actual
installed package may carry an SCM development version built from that commit. Both sides'
`downstream_versions` must keep the same real installed version; do not modify the package's
`__version__`, label a development commit as a formal release, or write `source@SHA` as the
actual distribution version. Existing release versions still require the original strict
version equality and cannot be relaxed through this branch.

Such records additionally bind a raw `.whl` artifact with `role=upstream-wheel`. The auditor
first checks its external path and SHA256, then reads the single METADATA, requires the
distribution name to match the repository name and the Version to match both sides' actual
versions, and verifies byte-for-byte the hashes of **all package Python sources** in the
inventory's `package_sources`, not just the smaller registry/target-class `sources` subset. The
`src/` prefix of a source path maps onto the wheel package root; missing files, duplicate ZIP
members, wrong sources or wrong versions are rejected. The source identity keeps the original
SHA/remote; the wheel does not replace the upstream commit.

Both sides' raw measurement metadata for base, perturb and L3 training must record the actual
installed contents:

```json
"downstream_distribution": {
  "repo": "<仓库名>",
  "version": "<实际安装的 SCM 版本>",
  "wheel_sha256": "<上面原始 wheel 的 SHA256>",
  "source_sha256": {"<package_sources 源码路径>": "<实际安装文件的 SHA256，逐项完整记录>"},
  "generated_source_sha256": {"<wheel 中额外生成的 Python 文件路径>": "<实际安装文件 SHA256>"}
}
```

The auditor requires this binding to match the wheel and the regenerated sources exactly. A
missing runtime, a missing training measurement or a missing source file cannot be compensated
for by the existence of the wheel. Python files additionally generated into the wheel, such as
`_version.py`, must also be bound completely; when there are no extra files the field is `{}`.
The version/source proof itself grants neither numerical, device nor L-level credit.

## The L4 speed block

L4 cumulatively requires L2 numerics and L3 multi-step loss/parameter trajectories; a probe that
only runs forward/backward cannot be promoted to L3. Use real sizes on the same card, the same
model specification and the same precision; both sides use the same warmup, timing count and
statistics. Warm up at least one round and time ten; production conclusions should add samples,
interleave reruns and record machine load according to the benchmarking specification.

Every row must carry `jittor_seconds`, `oracle_seconds`, the recomputed ratio, a
pre-declared `ratio_limit`, and **agreement and tolerance immediately next to the speed**; PURE
eager/default graph-replay modes must not be mixed, and platform flags and measurement regimes
must be identical. With no concurrent load, record warmup/iterations/statistic. CUDA memory is
the whole-process NVML peak on both sides, NPU uses process-level npu-smi, and CPU uses process
RSS; lacking the corresponding sampling means lacking L4 evidence, and allocator
allocated/reserved values must not be substituted. Allocator live/pool may be an extra column,
with its source and sampling interval described in the raw snapshot. L5 additionally needs zero
fallback on the real accelerator.

PRs report time only; speed-limit assertions are for nightly/acceptance runs on fixed hardware.
Diagnostic measurements may report raw times, but a "performance target met" conclusion must
not be derived from a timing table that lacks the platform or the numbers.

## Machine-checkable evidence structure

`evidence.json` is `{"schema_version": 2, "records": [...]}`. Below is the **structural
template** for L2; the angle brackets must be filled in by an actual run. The input-perturbation
check must change the input under fixed weights and re-run the comparison; "running the same
input twice" must not be recorded as verified. When integer tokens do not request input
gradients, write requested=0 explicitly; every requested parameter and floating-point input
gradient must be checked.

```json
{
  "repo": "transformers", "case": "<真实case>", "status": "passed", "level": 2, "tier": "broad",
  "jittor_commit": "<40位SHA>",
  "upstream": {"commit": "<40位SHA>", "version": "<源码版本>", "remote": "<实际上游URL>"},
  "downstream_versions": {"candidate": "<源码版本>", "oracle": "<源码版本>"},
  "registry_refs": [{"repo": "transformers", "axis": "model_family", "member": "<自动registry中的键>"}],
  "device": "cpu", "observed_device": "cpu",
  "platform": {
    "os": "<OS>", "cpu": "<型号>", "accelerator_model": "not-applicable", "card_count": 0, "card_indices": [],
    "driver": "not-applicable", "cuda": "not-applicable", "cann": "not-applicable", "nvcc": "not-applicable",
    "gcc": "<版本>", "nvidia_wheels": {},
    "precision": {"matmul": false, "cudnn": false, "cudnn_benchmark": false, "matmul_precision": "highest"},
    "jittor_flags": {"use_cuda": 0, "use_acl": 0, "graph_replay": "eager", "tf32": false, "hf32": false},
    "oracle": {"is_shim": false, "torch_version": "<版本>", "torch_npu_version": "not-applicable", "binary_extension": "<真实_C.so>"}
  },
  "inputs": {"seed": 0, "hf_hub_offline": true, "shape_kind": "tiny"},
  "numerics": {
    "passed": true, "finite": true, "checked_tensors": 1, "missing_tensors": [], "input_perturbation_verified": true,
    "max_abs": 0.0, "max_scaled_relative": 0.0, "atol": 0.0001, "rtol": 0.001,
    "parameter_gradients": {"requested": 1, "checked": 1}, "input_gradients": {"requested": 0, "checked": 0}
  },
  "evidence": {
    "command": "<完整命令>", "report": "docs/results/<dated-report>.md", "gate_nodeid": "<文件>::<真实测试节点>",
    "artifacts": [
      {"role": "reference", "path": "<checkout外原始输出>", "sha256": "<SHA256>"},
      {"role": "candidate", "path": "<checkout外原始输出>", "sha256": "<SHA256>"},
      {"role": "comparison", "path": "<checkout外数值比较>", "sha256": "<SHA256>"},
      {"role": "platform", "path": "<checkout外平台快照>", "sha256": "<SHA256>"},
      {"role": "base-comparison", "path": "<verify_repo的原始JSON报告>", "sha256": "<SHA256>"},
      {"role": "perturb-comparison", "path": "<固定权重输入扰动比较JSON>", "sha256": "<SHA256>"}
    ]
  }
}
```

L3 adds `training: {steps, loss_trajectory_passed, parameter_trajectory_passed, trainable, frozen, trainable_parameters, frozen_parameters}`, where the last two are lists of actual parameter names; its artifacts additionally need `training-reference-metadata` and `training-candidate-metadata`, pointing at the raw JSON of both sides' training probes. L4 adds `speed` (the fields of the previous section, plus `model/batch/sequence/dtype/precision/phase`, `graph_regime`, `no_concurrent_work: true`, `memory: {source, jittor_peak_bytes, oracle_peak_bytes}`); the device evidence adds `fallback: {policy: "error", count: 0}`. The machine check covers structure, SHAs, members, gate-node existence, numerical/level prerequisites and sampling conventions; it **does not replace the maintainer's check that the raw results really came from that command**.

Schema v2 reads the hashed real measurement JSON stage by stage: `cases[].oracle_result/jittor_result` of `base-comparison`, `reference_result/candidate_result` of `perturb-comparison`, and both sides' training metadata for L3. The perturbation JSON top level must contain `case`, `status=passed`, `input_perturbation_verified=true`; both sides' results use the shared runner's fields and must not be replaced by hand-written precision labels. Actual device, the TF32 matmul/cuDNN switches, `matmul_precision`, runtime conditions and the platform `precision` must be the same at every stage; the platform JSON must be exactly consistent with the record's platform block. L3 additionally checks the actual Git SHA, the step count, and the trainable/frozen parameter names and counts. Each artifact role is unique. Neither rewriting only the report labels nor recomputing the SHA of wrong platform labels can pass these checks.

The new shared runner and training probe also emit, in every actual measurement process, `dependencies.<module>.installed_files_sha256`, recording the contents of all files in the actual package directory, including sources, data and extension binaries, excluding `__pycache__`, `.pyc` and `.pyo`. A single-file module records only that file and does not walk neighbouring distributions. A missing actual module file, multiple roots in a namespace/search path, or symlinked directories inside the package are rejected explicitly, so that an incomplete enumeration is not mistaken for a complete installation source. This collection happens outside the computation's timing interval.

These hashes prove what was actually installed in this run; they must still be checked against the pinned upstream source or the original wheel to prove provenance, and the mere presence of the field does not grant an L level. For older measurements missing the field, keep the original mathematical results and the original identity boundary; do not recompute from today's installation directory and backfill historical measurements. The extra canonical-source binding for Metrics/TensorDict and the strict per-stage binding for SCM wheels remain.

`coverage[].verified_scenarios` keeps the case, its repo, the verified L level, the complete cross-library `registry_refs`, tiny/real, and the actual trainable/frozen parameter scope. A family numerator must be read together with its scenarios: Llama+LoRA at L3 must not be written as dense Llama full-parameter L3, and for an L2 scenario without training proof the parameter-update scope is explicitly null. Schema v1 is no longer used for new level grants; regenerate the v2 bindings from the raw measurements.

By default evidence is still bound to the current HEAD. Auditing historical reports may pass `--baseline <完整40位Jittor SHA>` explicitly; that SHA must be a real commit in the local Git, and the output records `baseline_policy=explicit-historical`, the measured baseline and the current `auditor_commit`. This only re-audits evidence for an old commit; it cannot prove the current code also passes. Never change the SHA in an old measurement to the current HEAD; new acceptance for the current tree must re-run the computation.

The existing [`verify_repo.py`](../../agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py)
reuses the ecosystem runners and can quickly produce both sides' raw metadata and numerical
diagnostics. It does not exercise input perturbation, multi-step optimizer trajectories or
complete platform/process-peak collection, so its output is explicitly marked
`scorecard_eligible=false`; do not automatically convert a single case's `passed` into a
complete scorecard row.

Multi-step supplementary evidence can be produced with
[`training_probe.py`](../../agent/skills/torch-compat-repo-runbook/scripts/training_probe.py):
the same registered builder, with an independent oracle saving the complete initial weights,
which the candidate loads before running the same SGD. The tool preserves the upstream tuner's
trainable/frozen partition and outputs per-step losses, parameter/input gradients and the full
parameter trajectory; frozen parameters changing, missing requested gradients, non-finite
values or device mismatches fail outright. It takes host snapshots of the raw gradients and
parameters to pin down the semantics of lazy updates; it makes no performance claim. For the
concrete two-interpreter commands see the
[runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/SKILL.md#多步训练证据).

## Results report template

```text
# <repo> <model/method> coverage scorecard
Status: measured / blocked / partial
Date: <date and timezone>
Baseline commit: <full SHA>
Owner: <maintainer role>
Review when: <upstream/compat/runtime/driver/measurement changes>

Upstream: <remote, full SHA, source+installed version, source hashes>
Platform: <固定平台块；仅引用外部文件还不够，关键值必须在报告可读>
Sampling: core <x> scenarios / broad <y> / tail <z>; selection rationale

| Model family / method / exact spec | CPU① level | Device③ level | Speed ratio / agreement | Status / evidence |
| ... | ... | ... | ... | ... |

Breadth: <X / N @ upstream SHA, 分轴分设备；未验证/未抽查数量>
Commands: <完整重现命令，包括明确CPU禁CUDA和独立缓存>
Artifacts: <lab相对路径、SHA256、未版本化>
Limits: <未覆盖的API/精度/设备/级别；缺硬件和环境错误分别写>
```

Keep unverified cells when there is no acceptable evidence. The negative test for this protocol
is [`test_downstream_coverage.py`](../../tests/tools/test_downstream_coverage.py).
