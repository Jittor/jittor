# OpenMMLab: fresh maintained CPU nightly validation

- Status: complete local CPU profile, independently audited; no hosted-schedule claim.
- Reviewed: 2026-10-16.
- Owner: Torch compatibility and downstream verification maintainers.
- Measurement source: `7bb144e984097b8cf14242818cb7fd4c9a58faa6`.
- Review when: upstream pins, builds, installation identity, source capture, numerical budgets or training owners change.

The complete `ecosystem_openmmlab_installed` CPU session passes both original nodes with zero skips in 163.15 seconds. The nox execution takes seven minutes including its first private JIT compilation. Every node runs seed 0, fixed-weight input perturbations at seeds 7 and 19, and three real MMEngine training steps. No preflight, case filter or smaller input replaces that session.

## Actual scope and result

| Original case | Input / parameters | Array pairs | Largest absolute error | Largest normalized error |
| --- | --- | --- | --- | --- |
| MMEngine BaseModel | FP32 sample `(4,6)`; six trainable parameters | 99 | 2.39e-7 | 5.22e-7 |
| MMCV ConvModule | FP32 sample `(2,3,8,8)`; eight trainable parameters and three BatchNorm buffers | 132 | 2.39e-7 | 1.55e-6 |

Independent NumPy recomputation verifies all **231 array pairs**, exact field/shape/dtype sets, finite values, all requested parameter and sample gradients, changed outputs and gradients under both perturbations, and complete identical initial parameters/buffers across all phases. The unchanged per-field normalized budgets are forward .002 and backward .01, with the original gradient-block floor. Fixed inputs, targets and integer counters agree exactly.

Training uses unmodified upstream `BaseModel.train_step`, `OPTIM_WRAPPERS.build` and `OptimWrapper`: SGD at learning rate .01, momentum .9 and gradient clipping at max norm 1. The saved gradient stage remains after clipping and before the optimizer step. The independent audit checks all 84 parameter updates and 84 momentum updates against each runtime's actual saved gradients and previous values, within four float32 ULPs of the update scale. All parameters change at every step. Training contains 75 MMEngine and 102 MMCV fields, with no missing gradient or shortened trajectory.

All 16 actual worker PIDs are distinct, and all 478 snapshot observations are CPU-resident. Both runtimes use two runtime/OMP/MKL/OpenBLAS threads, FP32 computation, disabled TF32/cuDNN benchmark and highest matmul precision. The controller explicitly hides CUDA; candidate metadata records `use_cuda=False`, `use_acl=False`, fallback policy error and count zero. The oracle loads genuine Torch 2.11.0+cpu from its own prefix. The interpreters use independent installation prefixes. The session has an external HOME/temp/cache boundary and a private candidate JIT cache; it does not claim separately captured per-runtime HOME/temp paths. Benchmark and unit-test compilation caches remain separate, and first compilation is serialized. Individual MM worker packets do not provide the newer shared full platform header or a separate CUDA-availability readback; neither is backfilled.

## Actual source and installations

The pinned clean upstreams remain MMEngine 0.10.7 at `390ba2fbb272816adfd2883642326d0fd0ca6049` and MMCV 2.2.0 at `3a5fb4b96ab1b241e7379adf7c326ef6edef6515`. The provisioner builds actual wheels from these checkouts, using the compatible pinned runtime build tools and preserving upstream symlinks. MMCV is built as `mmcv-lite`, without compiled ops; no `mmcv.ops` support follows.

| Built input | Full package files per prefix | Actual wheel SHA256 |
| --- | --- | --- |
| MMEngine | 157 | `d9691e3a6bc32141d9c6e13d9f8dce1234b4be3716024bf3f6b9dbdd32064da8` |
| MMCV-lite | 531 | `0a9658f06bd79a10343621834cbfad7a4dbf57f7bd41d8e7e483468939ea09a8` |

Both independently installed prefixes match every source-built wheel package byte and its original METADATA. Their actual pip RECORD hashes/sizes bind every package member; those receipts differ from the archive RECORD and were never replaced with it. The post-run four installation proofs exactly equal the premeasurement proofs. The official CPU Torch wheel SHA256 `8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485` separately binds all 11,945 original installed archive files except self-unhashed RECORD, including `functorch/`, `torchgen/` and dist-info payload. Its actual pip RECORD rows match every original file. All three private environments pass post-run `pip check`.

Provisioning originally ran on dirty `2f8895f1a` while the new profile was being prepared. Its manifest honestly records `source_clean=False` and provisioner SHA256 `bce3d824bd7c5cb82ce52baaf9441a70bda455198c4f15ea9d6adf0cdd57db55`; those bytes match the committed profile. The measured run instead uses the clean `7bb144e98` checkout. The controller's complete 2,905 tracked file hashes, current bytes and original Git blobs agree, including symlink target text. A complete source archive is retained. Each worker also binds its selected original case/runner/measurement files; this selected map is not an exhaustive import inventory.

## Reproduced failure and repair

The original clean `33e69e84b` run genuinely failed both nodes at the base comparison: two valid independent installations had different absolute `origin` and `package` paths, which the comparator incorrectly treated as unequal identities. Its raw outputs and failure logs remain unchanged. The fix first checks each real origin against that prefix's package initializer, current complete package hashes, version and original source/wheel proof. Only the two absolute location fields are excluded from cross-runtime identity comparison; every other declared field must agree. The legacy training helper receives verified dependency identities in copied reports, and original evidence packets remain unchanged. Origin, payload, resource, version, missing/extra identity and training integration negatives cover this behavior.

The repair passes 54 focused host tests, CPU core 322 passed / 42 skipped / one existing xfail, expanded structure 1,628 passed / six skipped, layout, generated manifest, Ruff and diff checks. All 74 pre-existing nox function ASTs remain unchanged. Later Metrics/TensorDict shared-proof integration also reruns the MM host contracts within its 260 passing tests; original `7bb144e98` numerical measurements are not relabelled to that integration.

## Replay and retained evidence

```bash
python tools/provision_ecosystem_nightly.py --library openmmlab \
  --python <healthy-python-3.11> \
  --state-root "$JITTOR_LAB_ROOT/_state/openmmlab-nightly/<new-run>" --run
```

The CPU workflow installs its Ubuntu/OpenCV runtime libraries. A local rootless-container preflight failed at user-namespace setup, before running the profile; no CI-container pass is claimed.

Unversioned evidence is under `$JITTOR_LAB_ROOT/_state/openmmlab-nightly/20261016/`: `formal-controller-v2.json`, `formal-terminal-v2.json`, `fresh-v3/logs/038.log`, both preserved `maintained-output` attempts, `complete-independent-audit-v2.json`, `complete-source-audit-v2.json`, actual installation receipts and complete Torch proof. The twice-rehashed `complete-evidence-seal-v2.json` binds **275 files / 412,397,540 bytes**, SHA256 `62a8d264feafcf836712b5c238ccd53d3e56d3857cf92a6a807d1b47f37d463c`; installed environments, upstream clones and compilation caches are excluded.

These are fresh bounded CPU training results for the maintained profile. They do not add a new CUDA run or paired common scorecard, prove GitHub's schedule executed, establish performance or imply whole-library support. The [earlier CPU/CUDA source-bound measurements](2026-10-15-openmmlab-installed-source-training.en.md) retain their own source identities and scope.
