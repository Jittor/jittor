# Source-bound Accelerate checkpoint protocols

- Status: validated bounded CPU/CUDA numerical and fresh-process recovery protocols; world-two CPU-paired admission remains blocked
- Date: 2026-10-16
- Baseline: `c7c2d37a085dad5e3ba9eb333a34c3cb570bfb09`
- Production change: `815be5daca4aa8cc1a9f569b66b00f78bf77bbaa`
- Owner: Torch compatibility maintainers
- Review when: Accelerate version/source/wheel, protocol instrumentation, installation-proof rules, checkpoint math, launcher or device provider changes

The previous protocol auditor accepted copied real CPU records after both sides added an unbound member to the existing `installed_files_sha256` map. It also ignored an injected `distribution_version=9.9.9` field, but that field did not exist in the old observations: this second probe does **not** prove that an actually installed 9.9.9 distribution passed the old gate. Original metadata had only module version, origin, installed-file hashes and source hashes, without actual distribution METADATA/RECORD. The original eight phase observations were untouched. The demonstrated gap is incomplete validation of installation claims and missing actual distribution proof.

The maintained worker now binds its real imported Accelerate package to the official 1.10.1 wheel, the complete package file map, actual same-prefix METADATA and RECORD bytes, and the upstream source. The auditor rechecks those bytes, complete payload and paths before calculating numerical results. Installed RECORD uses CRLF in these fresh environments; capture preserves the exact UTF-8 bytes. The new helper and existing source-proof/dependency-report helpers are included in every phase's measured-source hash map. The gate requires the actual official wheel; no old observation receives reconstructed metadata. Historical rows missing these actually measured fields remain historical numerical evidence and cannot receive current strict installed-source admission.

Accelerate has no model-registry denominator. These are explicit `record_kind=protocol` rows with empty axes and registry references. A successful checkpoint protocol does not establish complete Accelerate support.

## Frozen inputs and actual environments

Accelerate 1.10.1 is the clean official upstream `623cc0ba588d192c77758939a6e340adf07edd4f`; wheel SHA256 is `3621cff60b9a27ce798857ece05e2b9f56fcc71631cfb31ccf71f0359c311f11`. Its complete package closure is derived from that checkout and official wheel, and validated separately in each actual producer and consumer process.

CPU used three fresh environments: controller, candidate and independent binary Torch 2.11.0+cpu oracle. CUDA used two additional fresh runtime environments: the native candidate's complete pinned CUDA 12.2 wheel stack and the independently resolved official Torch 2.11.0+cu128 oracle. All installations used real dependency resolution and passed `pip check`. Candidate Torch distribution metadata came from the real official CPU wheel; actual numerical execution explicitly used the shim and native backend. The oracle asserted its binary extension and absence of the shim.

All four actual runtime prefixes have complete source/binary RECORD hash-and-size audits. Generated Python bytecode is explicitly excluded: an earlier audit retained a NumPy generated-pyc mismatch rather than altering that installed prefix. In addition, the new oracle's official Torch and 19 actual CUDA dependency wheels match 12,743 installed package source/binary files. The CPU Torch payload in all three prefixes containing it, and the candidate's ten CUDA dependency wheels, separately match their official wheel payloads. Packaging metadata and relocated scripts remain separately RECORD-bound. These checks do not assert that every installed optional library was exercised.

The frozen framework's 2,887 Git blobs, including the `python/jittor/compat` symlink target text, match both the actual source tree and setup snapshot. A live observer recorded 165 selected CUDA-library mapping snapshots and rehashed 20 unchanged mapped libraries. Actual candidate/oracle NCCL, CUDA runtime and cuBLAS paths point into their respective new private prefixes; the old shared provider was not substituted. Actual binary Torch allocation, matrix multiplication, synchronization and driver UUID probes passed on both assigned cards before the CUDA run.

## Numerical and recovery scope

The original protocol, model, optimizer/scheduler, fixed initial weights, seed-0/7 inputs, full gradients and budgets remain unchanged: elementwise `atol=2e-6`, `rtol=2e-5`; three updates before save and two subsequent updates. Each producer terminates before a new consumer constructs fresh model/optimizer objects and loads the checkpoint. World one additionally uses StepLR and restores a fresh scheduler; the maintained world-two FULL protocol has no scheduler. The reference uses actual torchrun; candidate multi-rank execution uses public `python -m jittor.distributed.launch`, the jtrun owner.

| Device / protocol | Array pairs | Parameter gradients | Input gradients | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| CPU, world 1, FP32 | 438 | 28 / 28 | 14 / 14 | 5.960465e-8 |
| CUDA, world 1, FP32 | 438 | 28 / 28 | 14 / 14 | 2.980232e-8 |
| CUDA, world 2 FULL, FP32 | 820 | 56 / 56 | 28 / 28 | 1.490116e-8 |
| CUDA, world 2 FULL, BF16 | 820 | 56 / 56 | 28 / 28 | 1.490116e-8 |

All 2,516 cross-runtime array comparisons passed, including all 168 parameter-gradient and 84 input-gradient fields. The own-runtime restored model/optimizer, Python/NumPy/CPU RNG, every visible CUDA RNG stream on the CUDA rows, the world-one scheduler and continuation trajectories are exact. BF16 rows retain the maintained checks of actual compute dtype/device and FP32 master parameters/moments. Every required CUDA phase executed, with actual device assertions and fallback error/count-zero checks.

A fresh CPU oracle torchrun/Gloo probe successfully initialized both ranks, but public Accelerate did not select FSDP2 on CPU. Therefore CPU world 1 only pairs with CUDA world 1. The two world-2 CUDA protocols retain their independent numerical/recovery facts while remaining blocked for CPU-paired common admission; no DDP substitution or direct native CPU FSDP is counted as that protocol.

Actual shared `audit_coverage.scorecard` admission: **2 accepted / 2 blocked / 0 rejected**. CPU and CUDA world-one rows have verified L3; the two CUDA FULL rows have independent verified L3 but admitted `level=null` because their exact CPU protocol counterpart is missing. No registry numerator is created. The existing registry validator was unchanged and its generic negative contracts passed.

The common auditor also accepted the actual new CPU control and rejected four disposable copies after all eight phase records were consistently rebound: actual distribution version 1.10.1 changed to 9.9.9, an extra unbound installed payload, changed/rehashed METADATA text, and removal of the official wheel artifact. The unchanged historical v6 CPU row was separately rejected for missing the wheel artifact; no new metadata was supplied to that historical row. Results are in `common-negative-results.json`.

## Reproduction and evidence

All raw observations, installed-source proofs, scripts and logs are unversioned below `$JITTOR_LAB_ROOT/_state/ecosystem-evidence-next/20261016/`. CPU output is `formal-cpu/`; final CUDA output is `formal-cuda-v2/`; the original `formal-cuda/` attempt remains immutable. The first CUDA attempt failed because the external launch script supplied an NCCL file where the existing owner requires a directory. After its natural termination, the corrected script reran both interpreters into new output, using only its own normal benchmark cache. No JIT cache was copied. A collector attempt in the controller-only interpreter failed for missing NumPy; its output is retained, and actual recomputation used the prepared independent CPU interpreter.

Use the maintained commands in the repo runbook with matching prepared interpreters and `JITTOR_ACCELERATE_WHEEL`/`--wheel`. For new hardware runs choose fresh output and separate benchmark caches, explicitly hide CUDA for CPU, allocate two healthy cards for CUDA, and retain all environment/source proofs. This result establishes neither multi-host execution, sharded checkpoint formats, performance levels nor other Accelerate APIs.

Maintenance checks at the frozen baseline: `tools/run_test_suite.py --tier core` completed with 322 passed, 42 skipped and 1 xfailed; the dedicated protocol and generic downstream scorecard host tests completed with 134 passed and no skips; full `tests/structure` plus `compat/tests/structure` completed with 1,624 passed and four skips; repository layout passed. New unit homes/caches were separate from benchmark homes/caches, and first compilations were serial. For future framework-free host-only checks, use `JITTOR_TORCH_SHIM=0`; this actual run used mode 1 and retained its additional cold unit compilation. After adding this report, the documentation/private-path check had 18 passed and one expected integration prerequisite: `TestDocsStructure::test_every_page_is_reachable_and_every_toctree_entry_exists` reports this new page missing from the central results toctree. The integration owner must register it and regenerate the central manifest; neither the test nor the index was bypassed.

The immutable preliminary measurement seal is `measurement-seal-v2.json`: 564 files / 105,020,755 bytes, SHA256 `637cf98d1a7e52c8f5aefbf064dfbd5510864c9ad3b2cec2d74c7a9b014b3b0c`. It excludes then-live unit logs; final maintenance/common-scorecard artifacts are sealed separately. The preliminary v1 self-log mismatch is retained as a failed sealing attempt. An independent maintainer audit also verified all 564 files, all 2,887 source blobs, 2,516 pairs, 168/84 gradients, 48 distinct formal workers and 1,520 own-runtime exact checks. Its 2,336 compute-device observations distinguish the 280 scalar CPU optimizer step counters from actual compute tensors; they do not require every optimizer control scalar to be CUDA-resident.

The independent audit is unversioned at `$JITTOR_LAB_ROOT/_state/accelerate-main-integration-4ce/20261016/independent-c7-audit-v3.json`, SHA256 `4c5e8757397ad574d18a37e12c66e1cc1e803def7f748184da5aec8f2c0a8ecf`.

The central integration independently rehashed the final seal's complete 682 files
(117907054 bytes), whose manifest SHA256 is
`173107537e5a2567ede0975c65abedcc14cd38d50ac7ba2140aa71031fd8bf94`.
This supplements the independent original 2516-array, 48-worker and complete
installation/source audit above; neither the original measurement baseline nor
its numerical budgets are changed by integrating the report.

Final central checks: 182 targeted host tests, core 322 passed / 42 skipped /
1 existing xfail, expanded structure 1628 passed / 6 skipped; layout, manifest
and Ruff passed. Current common admission independently reproduces 2 accepted L3,
2 CPU-unpaired blocked independent L3, and 0 rejected on the unchanged original
records. Only the FlashAttention and Accelerate nox function bodies changed;
the other 72 remain identical to the synchronized integration baseline.
