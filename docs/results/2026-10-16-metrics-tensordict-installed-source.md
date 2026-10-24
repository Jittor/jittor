# Metrics and TensorDict: fixed official installation evidence

- Status: fresh CPU/CUDA component measurements; shared-schema admission is recorded below.
- Reviewed: 2026-10-16.
- Owner: Torch compatibility and downstream verification maintainers.
- Measurement source: `91242395f555f3c24f60df665c3ea66e13d01caf`.
- Measurement remote base: `53d710647b0b44564af6fe476e0b16cf298b53b2`; later synchronized `2f8895f1a656bcdd66c280dbdbe222eb8268f56e` after all measurement processes terminated. The later synchronization does not relabel the measurements.
- Review when: library source/wheel pins, installation metadata, registry extraction, component/training builders, precision/device/fallback handling, snapshot observation or source-closure validation changes.

This batch closes an actual installation-evidence gap. It changes measurement and validation tools, with no model mathematics, optimizer, adapter or numerical-budget change. The [historical report](2026-10-15-metrics-tensordict-scorecards.md) keeps its original `90ee` arrays and historical auditor outcome. A read-only audit using the then-current `4ce` guard accepted its two Metrics rows even when existing non-Python resource hash claims were coherently replaced; its two TensorDict rows already failed because the generated `_version.py` lacked source authority. These are different findings. Under this batch's stricter guard, all four unchanged historical rows are rejected for missing fixed official-wheel evidence. No installation or platform field was backfilled into those old measurements.

## Actual component scope

Every measurement and collection re-enumerates the complete declared top-level public API axis from a clean pinned checkout. No optional-dependency filter or smaller handwritten denominator is used.

| Library and upstream | Declared axis | Actual maintained specification | CPU | CUDA |
| --- | --- | --- | --- | --- |
| TorchMetrics 1.7.4, `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` | 105 top-level public APIs | MeanAbsoluteError + MeanSquaredError; FP32 prediction and target `(4,8)`; output and both input gradients; zero parameters; seed 0 and fixed-weight seed 7 | L2, 6 array pairs, max abs 0 | L2, 6 array pairs, max abs 0 |
| TensorDict 0.10.0, `c34264a58b12471dc3e2913ea586b32431872d4c` | 60 top-level public APIs | TensorDict + actual lazy_stack; batch `[2,3]`, FP32 sample `(2,3,4)`; integer/bool/lazy indexing and Linear(4,3); both parameter gradients and sample gradient; seed 0/7 plus three SGD steps at lr .001 | L3, 29 array pairs, max abs 9.5367431640625e-7 | L3, 29 array pairs, max abs 7.152557373046875e-7 |

The unchanged acceptance budget is absolute .002 or scaled-relative .01, with the original per-phase global floor. An independent NumPy audit checked all 70 actual arrays, finite values, exact field/shape/dtype sets, every requested nonzero gradient, both parameters' real updates in both runtimes, changed outputs under input perturbation and complete equal initial arrays across CPU/CUDA. It also checked 20 distinct measurement PIDs, 148 tensor observations and identical committed-source maps across all workers. The maps bind 1,609 selected tracked framework/probe files, including Git symlink target text; a separate complete Git archive and prerequisite-based bundle preserve the measurement tree.

The component coverage is **2/105 Metrics APIs at L2** and **2/60 TensorDict APIs at L3** on each of CPU and CUDA: core 0, broad 4 device/scenario rows, tail 0. The remaining 103/58 exports are unverified. The TensorDict wheel's native extension is included in installation authority; this is not a claim that all extension kernels execute in this composition. No full-library, distributed, persistence, other dtype/backend, L4 or L5 conclusion follows.

## Actual platform and installation authority

| Platform field | CPU | CUDA |
| --- | --- | --- |
| OS / CPU | Linux 6.1.188, glibc 2.34; AMD EPYC 9554 | Same host and CPU |
| Allocated cards / mapping | Zero, `CUDA_VISIBLE_DEVICES=''`; availability read back false | One NVIDIA GeForce RTX 4090; actual physical UUID and visible/logical mapping in every worker artifact |
| Driver / toolkit / compiler | Driver/CUDA/nvcc/CANN not-applicable; g++ 11.5.0 | Driver 580.178.04; native CUDA/nvcc 12.2.140; independent Torch CUDA 12.8; g++ 11.5.0; CANN not-applicable |
| Precision / native flags | use_cuda/use_acl 0/0; TF32 and cuDNN benchmark off; highest | use_cuda/use_acl 1/0; both TF32 switches and cuDNN benchmark off; highest |
| Graph / threads | Default graph replay 1, retention 268435456 bytes; runtime and OMP/MKL/OpenBLAS threads 2 | Same graph policy and thread settings |
| Independent binary oracle | Genuine Torch 2.11.0+cpu in its own prefix | Genuine Torch 2.11.0+cu128 in another own prefix |
| Speed / whole-process memory | Not measured for L4 | Not measured for L4 |

Each actual worker captures the shared `_runtime_platform.py` header, selected NVIDIA distribution versions, compiler readbacks, precision, native flags, executable/PID, complete runtime paths and source closure. The candidate/reference HOME, JITTOR_HOME, TMPDIR and XDG_CACHE_HOME are separate; benchmark and unit compilation caches are separate, and initial compilation is serialized. Every CUDA snapshot checks the real driver pointer ordinal and allocated physical UUID; native snapshots also require actual device residency/dispatch. Both devices forbid fallback and read back count zero. Supplemental `/proc` library maps cover nine of ten actual CUDA measurement PIDs, excluding compiler children; they corroborate the fresh runtime prefixes and never substitute for tensor evidence.

Four new private compute environments resolve complete dependencies and pass `pip check`; NumPy is 1.26.4 and SciPy 1.13.1. No shared installation or old sealed prefix was modified, and no JIT cache was copied. The independent Torch installations additionally match their fixed official wheel inputs, actual METADATA and all `torch/` payload files: 11,781 CPU and 11,677 CUDA files, excluding bytecode caches. The actual RECORD file is hashed and its `torch/` rows are checked for payload hashes and sizes. These installation counts are not model coverage.

The whole-wheel SHA256 binds the input archive. Installed-payload verification here covers the complete `torch/` package, not every prefix member of that wheel: `functorch/`, `torchgen/` and other dist-info payload files are outside this additional oracle proof. This batch uses normal private pip installations and retains their actual pip RECORD bytes; it does not restore the archive RECORD over the installed file.

The downstream pins are fixed official PyPI artifacts:

| Package | Whole official wheel SHA256 | Actual package payload |
| --- | --- | --- |
| TorchMetrics 1.7.4 | `9298ad0e893b0cf2956bee95b0f7eecdc65205ab84ddb4f6762eff157c240518` | 341 tracked Python sources, 346 installed files, including non-Python resources |
| TensorDict 0.10.0 cp311 manylinux | `bccb6a7d7f02a8c83ca41815943bfa284da7d28019ab4276135809bda5ba05a6` | 36 tracked Python sources, 42 installed files, including generated `_version.py` and `_C.so` |

The new `_metric_td_installation.py` keeps library pins and TensorDict's actual setuptools-scm declaration. The clean upstream release tag is verified at the exact commit; the pinned wheel's generated version is 0.10.0 and its actual commit_id is **None**, not an invented upstream SHA. Its bytes remain protected by the fixed whole-wheel hash. The shared `_installed_source_proof.py` rechecks the selected complete package payload, selected distribution root, original wheel METADATA and that package's real pip RECORD hash/size rows; vLLM keeps its original pins and public helper behavior through a thin caller.

## Reproduce and audit

Use the maintained [`component_measure.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/scripts/component_measure.py), [`verify_repo.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py) and [`training_probe.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/scripts/training_probe.py). Install core/frontend normally into the candidate prefix and keep a separate genuine binary Torch prefix. Pin both upstream commits, official remotes and wheels above; fetch TensorDict's actual `v0.10.0` release tag and verify its peeled commit. A tagless checkout honestly inventories as `source@SHA` and is refused for this release profile.

Each runtime's executable launcher must restore its own four runtime paths after `verify_repo` sanitizes child environments, then forward to:

```bash
<runtime-python> agent/skills/torch-compat-repo-runbook/scripts/component_measure.py \
  --root <clean-measurement-checkout> --repo <torchmetrics-or-tensordict> \
  --checkout <clean-pinned-upstream> --wheel <fixed-official-wheel> -- "$@"
```

For CPU explicitly hide CUDA and set both canonical/legacy CUDA flags to zero and both nvcc paths empty. For CUDA select the assigned physical card, enable strict native wheel validation and explicitly restore the actual `JT_BUILD_NVCC_PATH` in **both** launchers: the independent-oracle sanitization otherwise removes it. Use TF32 off, cuDNN benchmark off, highest, offline flags and equal thread settings. Forward base `--seed 0 --repeats 3 --no-memory`, repeat the maintained runner with seed 7 and the original full reference weights, and run TensorDict training with `--steps 3 --learning-rate .001` and those same weights. Use new output directories; the collector refuses changing existing comparisons.

```bash
scripts=agent/skills/torch-compat-repo-runbook/scripts
python "$scripts/collect_component_scorecard.py" --root <auditor-checkout> \
  --upstream torchmetrics=<clean-metrics> --upstream tensordict=<clean-tensordict> \
  --wheel torchmetrics=<metrics-wheel> --wheel tensordict=<tensordict-wheel> \
  --run torchmetrics=<cpu-metrics> --run torchmetrics=<cuda-metrics> \
  --run tensordict=<cpu-tensordict> --run tensordict=<cuda-tensordict> \
  --platform cpu=<cpu-worker-derived-platform> --platform cuda=<cuda-worker-derived-platform> \
  --baseline 91242395f555f3c24f60df665c3ea66e13d01caf \
  --report docs/results/2026-10-16-metrics-tensordict-installed-source.md \
  --command '<complete retained two-runtime matrix invocation>' --output <new-scorecard-json>
```

The collector re-derives both inventories and invokes actual `audit_coverage.scorecard`; `final-scorecard.json` (SHA256 `80f3cb67e9a9321474fa548908c38737b304fd8aa4b6d4fd4c294cebdd5896b3`) contains **four accepted records, zero rejected**, and their bindings. Shared platform files retain stable actual binary identity; process-specific PIDs stay in each original packet. It cannot substitute a CPU-less CUDA row or metadata-only claim for the measured protocols.

## Regression evidence and boundaries

Focused installation/component/platform tests passed 107 cases, including coherent changes to actual resource/generated-version/native payload bytes, honest RECORD repairs and matching captured claims: the immutable wheel still rejects all replacements. Additional shared source/vLLM/Accelerate tests passed 131. Four original c997 vLLM actual CPU/CUDA installation packets retain byte-for-byte equal bindings after extraction. The original c7 Accelerate audit remains two accepted/two blocked/zero rejected; all 48 original platform headers replay identically. This is format/guard compatibility, not a numerical rerun or an upgrade of old blocked evidence.

The first CPU attempt passed Metrics then stopped before TensorDict execution because the fresh checkout lacked the actual release tag. The first CUDA attempt stopped in reference platform capture because the external launcher had not restored the explicit compiler path. Both attempts and original invocation scripts remain intact; successful v2 measurements use separate output directories and the same clean `91242395f` source. Actual copied-artifact negatives also reject all four rows when the fixed wheel is omitted, existing resource/generated payload claims change coherently, the explicit distribution version changes to 9.9, or committed source hashes change. Removing a full parameter-gradient array is rejected by the component collector. Removing CPU rows retains the independently valid CUDA numerical records but yields zero accelerator coverage numerator; this does not turn absent CPU evidence into a runtime failure.

On the later clean validation source `2e0bdac97585df049da56d7a61dccaef5aeff3b3`, the dedicated unit environment passes CPU core: 322 passed, 42 skipped and one expected failure. Complete structure checks finish with 1,628 passed, four skipped and two failures: the new report's central toctree entry and generated MANIFEST registration. Those two files remain with the integration owner; this is not a fully passing structure claim. The complete original output and exact failed nodes are retained in `mandatory-status.json`. After all processes terminated, remote `bc4dc0bc01a3930d3fad22326da0f9b8c70d444f` was integrated at `0bcf640b140e764db828ef81bb57e67c93f46338`; that synchronization does not relabel any numerical or core/structure result.

All raw arrays, metadata, commands, official-wheel selections, environment locks, negative experiments, independent rechecker and source archive/bundle are unversioned under `$JITTOR_LAB_ROOT/_state/metric-td-installed-source/20261016/`; `evidence-seal.json` binds the retained files. This batch did not execute or trigger a hosted nightly, change the repository's default branch or grant unseen hardware/backend coverage. The maintained nodes remain the original Metrics and TensorDict CPU/CUDA `EcosystemParity` nodes.

## Independent integration audit

The integration audit rehashed every file in the delivery seal: 421 files and
110,079,967 bytes, with seal SHA256
`7073bd2146bbe2d456cdc7fa0224fff4443edbac98ca20ad0d9e7fee1727298a`.
It independently recomputed all 70 array pairs, all 20 distinct worker identities,
148 physical observations and the 1,609 selected source hashes against the original
`91242395f` Git objects. Both parameters changed at every TensorDict training step;
the recorded gradients reproduce the SGD updates within two float32 ULPs.
The complete historical archive also matches all 2,906 tracked Git files byte for
byte, including executable modes and symlink targets; its prerequisite-based bundle
verifies against the retained Git history. Re-enumerating both real upstream
checkouts reproduces both complete inventories, including the generated TensorDict
version authority. The common auditor still accepts four rows and rejects none.
Raw measurements retain their original source SHA; the integration does not
create new numerical or whole-library coverage.

The clean integration commit `67e166018cd5ec1c05413d2daf1c7302b2461970`
passes 260 targeted host tests, CPU core 322 passed / 42 skipped / one existing
xfail, and the expanded structural gate 1,628 passed / six skipped. Layout,
generated manifests, Ruff and diff checks pass; all 74 pre-existing nox function
ASTs remain unchanged. The independent integration seal contains 28 files and
52,728,692 bytes, SHA256
`7ffc1a2416cbcab5707d8aac6cf7b3710bccd397e213c74e929e1c1076e440df`.
A precommit source test correctly rejected the new helper's absence from the
previous Git HEAD; the unchanged test passes against the actual committed source.
