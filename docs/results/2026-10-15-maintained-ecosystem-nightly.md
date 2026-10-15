# Maintained PEFT and Swift CPU nightly gates

- Status: workflow wiring plus local execution; no GitHub run was triggered or observed.
- Date: 2026-10-15
- Measured source: `08a3ee3ca981855db55e3d7a38d7ba6503b73e7e`
- Upstream baseline integrated before the final helper change: `0fb3920d4a44f37fa6cea7fa856374f0bacf73b4`
- Owner: Jittor compatibility maintainers
- Review when: the workflow image/Python, package pins, provisioning, installed-source contract or either maintained gate changes.

## Entry and scope

`.github/workflows/ecosystem.yml` retains the existing generic `ecosystem` job and adds
independent `maintained-cpu` matrix jobs for `peft` and `swift`. Both scheduled and manual
workflow events use the same jobs. Each invokes the existing complete maintained
`ecosystem_peft` or `ecosystem_swift` session; missing provenance, libraries, oracle or
required execution is an error. No new adapter or model/optimizer mathematics is added.

On a healthy Linux Python 3.11 installation with a native C++ toolchain:

```bash
python3.11 tools/provision_ecosystem_nightly.py --library swift \
  --state-root "$JITTOR_LAB_ROOT/_state/ecosystem-nightly/<run>/swift" --run
python3.11 tools/provision_ecosystem_nightly.py --library peft \
  --state-root "$JITTOR_LAB_ROOT/_state/ecosystem-nightly/<run>/peft" --run
```

Use new state roots. CI uses `RUNNER_TEMP`; checkouts, wheels, environments, HOME,
TMPDIR, model caches, compilation caches and logs stay outside the source checkout.
The optional read-only source/wheel cache arguments retain exact source checks and
full dependency resolution. They do not copy installed environments or JIT binaries.
The source checkout must be clean before formal execution.

## Environment and installation proof

Each library gets three fresh environments: the controller uses canonical nox/build
requirements; candidate and oracle each install the complete real runtime dependency
closure. Runtime setuptools 80.9.0 respects the official Torch wheel's `<82` requirement;
canonical setuptools 83 remains in the separate controller. All three run `pip check`.
The oracle must load genuine Torch `2.11.0+cpu` and its binary extension. Only the
candidate activates the repository's official editable Torch shim. CUDA is explicitly
hidden and both canonical and legacy CUDA flags are zero.
Both actual CPU Torch wheels have SHA256
`8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485`.

The maintained sources remain PEFT 0.17.1 and Transformers 4.56.2, with the exact official
Git SHAs and wheel SHA256 values in `_peft_scorecard.PINS`. Swift 4.5.2 is built from clean
`ff5128777dd3dc5538f1a95fbcc29442c327d7d2`, not described as a PyPI wheel. NumPy 1.26.4,
Accelerate 1.10.1 and the existing numerical budgets remain fixed. Swift additionally
pins PyArrow 19.0.1 and actually checks an Arrow/NumPy roundtrip and Dataset construction
before first JIT. Wheel files, resolved locks, clean upstream inventories and installed
payload/METADATA/RECORD identities remain available in the unversioned evidence.

Independent installation prefixes exposed an existing Swift validator false negative:
absolute package roots and dependency origins were compared across runtimes. The fix
retains the raw paths, validates each actual package root and full installed payload on
its own side, and excludes only `package` and each dependency's `origin` from the
cross-runtime equality comparison. Versions, every content hash and all other identity
fields remain exact. The complete four-dependency set is mandatory. Real independent
prefix fixtures and missing-origin/content/version/wheel/metadata/root/key-set negatives
cover this change. No numeric threshold is relaxed.

## Local execution

Swift fresh provisioning used `af34c07aef9725423ba08b4b1aef647e4f8a3645`. Its workflow and
provisioner bytes are identical at the final measurement source `08a3ee3...`; the final
helper validation runs in those task-private candidate/oracle/controller environments.
The new output records both SHAs and normal reuse of that task's fingerprint cache;
it does not claim a second cold build or relabel the failed earlier run.

| Check | Actual result |
| --- | --- |
| Swift maintained FULL SFT CPU node | 1 passed / 0 skipped, 97.47 s |
| Swift three-environment `pip check` | all passed |
| Provisioner and Swift contract host tests | 77 passed / 0 skipped |
| PEFT maintained complete CPU gate | passed; 177 array pairs; maximum absolute error `1.4543533325195312e-05`, scaled relative `3.135175185365948e-07` |
| Layout | passed |
| Complete `tests/structure` + `compat/tests/structure` | 1621 passed / 4 skipped / 3 failed: missing unit ASV plus pending central toctree/MANIFEST registration |
| Original ASV failure after private unit-only dependency provisioning | 1 passed / 0 skipped |

The Swift node exercises the registered tiny Llama loader/template and real
`SwiftSft.main()`/`Seq2SeqTrainer.train()`, three SGD updates, all twelve trainable
parameter gradients and an embedding intermediate gradient, with three actual input
variants and an independent binary Torch oracle. Integer token inputs do not request
input gradients. This is a bounded numerical training case, not whole-library coverage.

The PEFT session runs six real workers: each runtime executes base, seed 7 input
perturbation and three-step training. Base/perturb request all 29 parameter gradients;
training preserves eight trainable LoRA parameters and 21 frozen parameters, checks
all three updates and leaves frozen parameters byte-identical. Its original strong
collector accepts all 177 array pairs. This is the existing complete tiny LoRA scope.

The complete structure run retained its original three failures. ASV was absent from
the new runtime because it is not a downstream runtime dependency; installing canonical
`asv==0.6.6` and its real closure into a **separate unit-only target**, with both measured
environments unchanged, made the exact failed node pass. The remaining nodes are
`test_docs_structure.py::TestDocsStructure::test_every_page_is_reachable_and_every_toctree_entry_exists`
and `test_packaging_structure.py::TestPackagingStructure::test_manifest_covers_runtime_trees_without_cache_payloads`.
Central results toctree/MANIFEST registration and final integrated structure validation
remain with the integrating maintainer; the local full run is not claimed green.

## Retained failed attempts

| Attempt | Actual terminal observation |
| --- | --- |
| Swift v1 | official Git HTTPS transfer failed; no numerical result |
| Swift v2/v4 | dependency transfers intentionally interrupted before JIT while preparing verified private replay inputs |
| Swift v3 | pip rejected canonical setuptools 83 versus official Torch's `<82` runtime dependency |
| Swift v5 | offline installation incorrectly requested two fsspec versions as roots; fixed by resolving the original declared roots |
| Swift v6 | real PyArrow 26 binary required NumPy 2 despite pip-compatible metadata; the original NumPy 1 protocol was preserved |
| Swift v7 | real independent-prefix training reached the metadata comparison, which rejected different installation paths; no successful numeric result claimed for this failed node |

All attempts retain their original source/log/exit identities. Bootstrap repairs and
verified download reuse were confined to task-private inputs; shared installation sites
were read-only. The v7 final environment's first CPU JIT was performed serially from an
empty cache. An unused copied cache was withdrawn before any v7 JIT and is recorded as
such; no copied binary was used to claim a new cold build.

## Evidence and limits

Unversioned evidence is under
`$JITTOR_LAB_ROOT/_state/ecosystem-nightly/20261015/`. It includes `swift-v7/provision.json`,
`swift-final-08a/{measurement,commands,completed}.json`, the maintained nox log,
`peft-v1/` (including its maintained six-worker arrays and validation records), original attempts, genuine package-prefix reproduction records and
`source-08a-complete.json` (2,867 Git blobs, including symlink target text). The final
manifest binds selected raw logs/proofs and source snapshots; environments, wheels,
compiled binaries and caches are excluded from that manifest and the repository.
`manifest.json` binds **313 files / 29,123,587 bytes**, SHA256
`14f583115e57e2b616173731ae62c6126acfbc762093cc7aabb9e51ee693c4bf`.

This change wires nightly execution and locally verifies only the stated Linux CPU
sessions. It does not prove a completed GitHub scheduled run, CUDA/NPU/ROCm/Windows,
performance, distributed/full-checkpoint workflows or support for the whole upstream
libraries. Existing versioned CPU/CUDA scorecards retain their original measured SHAs
and breadth. Dedicated VERL/Transformers/MM nightly integration remains a separate
follow-up; this two-library change does not close all of R32.

## Main integration verification

The integrating maintainer synchronized published
`cd7d45ab99e086f2b027be35bffa66d3b33efdd7` and retained the original measured
source identities above. Clean integrated source
`c4935e9061964f17fa7d8ac4fb9930e1b84f5bfd` adds the maintained gates, source
contracts, discovery and central results/MANIFEST registration. All 73 existing
nox function ASTs are unchanged; only `ecosystem_openmmlab_installed` is added.

On this integrated source, host contracts for provisioning, Swift, distribution
names, installed source/version proofs, OpenMMLab, shared coverage, VERL, PEFT and
Transformers pass **228 / 0 skipped**. Canonical core passes **322 / 42 skipped /
1 existing xfail**, with both native and Torch mode exits zero. Complete
`tests/structure` plus `compat/tests/structure` passes **1622 / 6 skipped** in
235.97 seconds. The six skips retain their accelerator/declared/uninstalled
native-distribution boundaries; no installation or hardware claim is inferred.
Layout, generated manifest, Ruff and whitespace checks pass.

The maintainer independently rehashed all 313 nightly evidence files and all
1033 OpenMMLab evidence files: no mismatches. Original failed attempts and
measurement/source identities are preserved. New isolated maintained numerical
runs at c4935 are in progress; these integration checks do not relabel the old
numerical results as c4935 measurements. Unversioned integration records are in
`$JITTOR_LAB_ROOT/_state/maintained-gates-main-integration-cd7/`.
