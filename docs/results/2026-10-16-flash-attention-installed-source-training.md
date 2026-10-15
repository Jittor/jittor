# FlashAttention training with distinct installed-source identities

- Status: Complete fresh twelve-case maintained CUDA gate, independent raw-array audit and independent source/common recheck passed. All twelve independently verified CUDA L3 records remain blocked by the missing CPU counterpart; shared CUDA breadth receives no contribution.
- Measured source: clean `c33627faaf99f1cc9f422421f90c3896d4cbc326`.
- Owner: native CUDA and Torch compatibility maintainers. Reviewed: 2026-10-16.
- Recheck when upstream/CUTLASS pins, Torch ABI, facade/bridge, native autograd, provider selection, comparison budgets or source-proof tools change.

## Result and preserved scope

The maintained `ecosystem_flash_attention` nox entry executes all original twelve CUDA cases, with six public dense/packed/varlen APIs × FP16/BF16. Each runs base and controlled noise perturbation `0.125` in independent Torch and Jittor processes, with fixed seed `20261015`, three SGD updates at learning rate `.01`, FP32 parameters and parameter/input gradients, noncausal attention, head dimension 32 and dropout zero. Dense shape is `(2,11,2,32)`; varlen shape is `(11,2,32)` with sequence lengths `[4,7]`. All 48 workers and all requested gradients remain in scope.

The 24 runtime pairs retain 720 actual array comparisons and 216 separate update comparisons. Initial parameters require byte equality. The unchanged normalization is per tensor maximum absolute reference, floored at `1e-8`: FP16 forward `.003`, gradients/updates `.006`; BF16 `.03`. Input hashes verify fixed Q/K/V/target and changed noise; outputs must actually change on both runtimes. The twelve nodes passed without skips in 1063.92 s. An independent direct NPZ audit passed all 936 checks and 24 actual input-perturbation checks; its maximum normalized errors are FP16 `0.002433830240340736` and BF16 `3.90727286094882e-7`. FP16 parameter/input gradients are byte-identical for 140/144 arrays, BF16 for 144/144; all others satisfy their unchanged individual budgets. The complete maintained nox, including automatic source/numeric/common validation, succeeded in 27 minutes. These are correctness-gate times, not speed measurements.

This is tiny component training correctness. KV cache, other dimensions/causal/dropout settings, production model sizes, other hardware and speed are outside this fresh protocol. The official seven-API inventory is retained separately; the protocol has `axes=[]` and does not invent a registry denominator.

## Two actual implementations

The fixed official upstream is FlashAttention **2.7.4.post1**, commit `5231d95fe13733fb534c01895f7ea88c6a6c7793`, with actual official CUTLASS gitlink `c506e16788cb08416a4a57e11a9067beeee29420`. The oracle and candidate are distinct implementations:

| Identity | Independent reference | Candidate |
| --- | --- | --- |
| Python | All 58 tracked upstream package Python files, unchanged | Five recursively checked Jittor-owned facade files |
| Installation | Local wheel, complete 77-member RECORD/payload | Real `jittor-torch==2.0.0` carrier METADATA/RECORD/payload, private deployed facade |
| API metadata | Real local wheel distribution | Explicit owned compatibility metadata; not an upstream wheel installation |
| Native code | Previously genuinely source-built official extension | Actual official-source native bridge and packed extension, generated C++/CUDA and loaded binaries |
| Torch identity | Complete official binary Torch `2.11.0+cu128`, 11842 installed wheel members | Jittor Torch compatibility frontend from measured source |

The local reference wheel is **not a published PyPI binary**, and this task did **not rebuild its native extension**. Unchanged upstream `setup.py build_py` in a legitimate Git source archive packages the actual pinned Python and original verified native binary. All wheel Python members match tracked upstream source. Two tracked non-Python package files excluded by unchanged upstream build_py are explicitly recorded. The original native build manifest is checked against all 85 official CUDAExtension source declarations, upstream CUDA flags, original Torch/CUDA ABI and binary bytes.

Reference wheel SHA256: `c0685ff498af6cde5376f9b24353241957f1ecc534fe8e8976cd8f49a5d65a74`.
Original source-built reference extension SHA256: `3c83031779c640c6179800acf4828609df61794b768ff938a42fe49b6eeeaa8a`.
Official binary Torch wheel SHA256: `c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`.

Candidate extensions were cold-compiled serially in a private task cache, including the original head-32/64 × FP16/BF16 build configuration. The first real bridge build belongs to clean `cf4560f90`; later synchronized source and directory fix are measured at `c33627faa`, with the actually loaded native core/bridge bytes captured. Existing private cache entries were handled by the normal builder; no JIT cache was copied into a new runtime. Superseded mutable preflight bytes are archived separately as evidence and never used as a compilation cache.

Each tensor process rechecks complete imported wheel/carrier bytes, source descriptor, all 1654 measured framework/tool files, actual generated bridge sources and loaded binaries. The actual compiler configuration also binds 2343 Python/CUDA/pybind headers and real `pybind11==2.13.6` METADATA/RECORD/payload. Actual CUDA pointer residency is observed for all operands, outputs, losses, parameters and gradients, along with runtime process IDs, physical-device UUID, selected NVIDIA metadata and loaded library hashes.

Platform: Linux, CPython 3.11.16, NVIDIA RTX 4090 D / sm89, driver 580.178.04. Candidate NVCC is 12.2.140 / GCC 11.5, with actual runtime API 12020; reference uses genuine Torch Cu128 and actual runtime API 12080. This is not a claim that every candidate dependency comes from a uniform Cu122 stack. Each actual library/provider path is preserved. Some historical read-only NVIDIA dependency metadata lack RECORD; that absence is explicitly recorded rather than repaired or called complete NVIDIA installation proof. Complete Torch, reference Flash wheel and candidate carrier RECORD/payload checks remain mandatory.

## CPU boundary and common admission

A fresh independent genuine Torch process, using the actual installed local wheel with CUDA hidden, allocated CPU FP16 Q/K/V and called the public dense function. It raised upstream `NotImplementedError` for `flash_attn::_flash_attn_forward` on CPU. The pinned source also declares dense/varlen forward and backward custom operations with `device_types="cuda"`. This is an unsupported CPU boundary, not a numerical CPU counterpart.

New rows use the exact `flash-attention-training` protocol and distinct implementation proofs. The ordinary registry/installed-source rules are unchanged. The maintained common auditor returned **0 accepted, 12 blocked, 0 rejected**, with all twelve `independent_verified_level=3` and the explicit reason “Missing verified CPU counterpart for this exact protocol”. Independently verified CUDA L3 remains blocked without a matching genuine CPU counterpart; CPU fallback is not fabricated and CUDA breadth receives no contribution. Historical checkout-only raw and old schema conclusions are not patched or relabeled. Their original numerical evidence remains separate; the current installed-source audit rejected all twelve old records for missing installation/source bindings.

## Maintained entry and immutable evidence

Use the [Flash skill setup](../../agent/skills/flash-attention-torch-compat/SKILL.md), including explicit `JITTOR_FLASH_ATTN_HEAD_DIMS=32,64` and `JITTOR_FLASH_ATTN_DTYPES=fp16,bf16`. Use private pre-provisioned candidate and independent binary-Torch interpreters, an explicitly allocated GPU, the actual pinned upstream checkout, local reference wheel/native build manifest and full official Torch wheel proof. Set `JITTOR_FLASH_ATTN_SOURCE_PROOF` to a JSON containing the four real paths (`upstream`, `wheel`, `oracle_build`, `torch_wheel_proof`), then:

```bash
JITTOR_TEST_DEVICES=cuda JITTOR_FLASH_ATTN_EVIDENCE_DIR=<new-external-case-root> \
  <controller-python> -m nox -s ecosystem_flash_attention
```

The session checks real oracle CUDA, installs only the private candidate compatibility package, deploys/verifies the facade through public `shim.deploy --target/--check`, runs all twelve existing ecosystem pytest nodes, then automatically runs the full numeric collector/common scorecard. Existing case directories are rejected; failed explicit outputs remain. This opt-in entry is separate from a claim of new nightly CI wiring.

Independent pure-host recheck uses the exact measured source and new output:

```bash
<oracle-python> agent/skills/flash-attention-torch-compat/scripts/collect_scorecard.py \
  --root <measured-worktree> --upstream <pinned-official-checkout> \
  --source-proof <external-source-configuration.json> \
  --evidence-dir <immutable-formal-case-root> --output-dir <new-external-recheck> \
  --baseline c33627faaf99f1cc9f422421f90c3896d4cbc326
```

Separate raw-array audit independently recomputes the original 936 checks and input perturbations. Full Git/source, actual external installation/native/compiler bytes, setup failures, all worker logs/JSON/NPZ and rechecks are sealed outside Git. Unversioned original evidence is under `$JITTOR_LAB_ROOT/_state/flash-source-bound/20261016/`: `formal-v4/cuda`, automatic `formal-v4/cuda-scorecard`, `independent-recheck-v4`, and `formal-v4-independent-array-audit.json`. The complete `evidence-manifest-v4.json` binds 2888 Git source items and 17666 external/raw files, SHA256 **`41434968792cf06967514d8b9aaba9932f39498285795d74e7b0fac48d514304`**. The post-run Git snapshot is byte-identical to the initial snapshot (SHA256 `12574dc22a3682d5f191c4e85a8dcc1c64486bd4e5ac48a75019d98f4906d13b`). Both collectors agree on all semantic, numeric and platform fields: zero accepted, twelve blocked, zero rejected, `level`/`verified_level` null and `independent_verified_level=3`. The original frozen worktree and evidence remain read-only.

Setup failures are preserved: an initial source-only Git packaging attempt hit upstream's unconditional ROCm submodule network step; legitimate archive-layout packaging succeeded without edits. Candidate preflight first failed before math due to undeployed facade, then missing real pybind headers; both have controller-only launch identity and no backfilled runtime proof. After their actual fixes the independent minimal 39-check pair passed. Formal-v3 then failed all twelve nodes before starting tensor workers because nox had not created its oracle state directory; a focused host regression reproduced it, the directory initialization was fixed, and formal-v4 uses fresh output and a new clean source. Earlier unit environment packaging/xdist setup failures are also retained.

Verification before final measurement: 77 bounded provenance/comparison/nox host checks; required `tests/structure` 1511 passed / 4 skipped in 200.12 s; Ruff, layout and diff checks passed. Full Git snapshot checks 2888 tracked files against their actual Git blobs, treating symlinks as target text. Raw failures and superseded source identities remain separate from the final measured cohort.

## Separate artifact-admission follow-up

After sealing the original cohort, a separate clean worktree synchronized remote `4cefbacdca771f7d25512beccc541d902836b894` at merge `9d6fb0a703cd7b4df0d00c965494d7ef4f9993bf`. A genuine binary-Torch CPU host-only probe created a coherent counterfactual wheel/prefix/RECORD: it added an untracked non-Python JSON resource and correctly rehashed every RECORD member. The old descriptor and actual imported-package capture accepted all 78 members. This is preserved as a failing admission test, not as tensor or GPU evidence.

The narrow follow-up fixes the **actual local source-built artifact SHA256**, `c0685ff498af6cde5376f9b24353241957f1ecc534fe8e8976cd8f49a5d65a74`, in this bounded protocol. It is not a PyPI Flash wheel claim. Complete coherent add/remove payload fixtures now fail even when their RECORD and installation match perfectly. A real positive check reconstructs the original artifact descriptor and proves it is identical to the descriptor captured by the sealed `c33627faa` workers; the real coherent variant is rejected at the fixed-artifact guard. Any different legitimate local build needs a reviewed artifact pin and fresh source-bound evidence. Candidate facade package closure already requires the exact recursively checked five-file owned payload. The ordinary registry/installation rules and CPU prerequisite are unchanged.

Postprocessor commit: **`be4827ef3e80a8f2c0c48ed67f2f6298c39ddee7`**. Its 58 pure host tool checks passed; required `tests/structure` passed 1516 / 4 skipped in 501.82 s after separate serial CPU bootstrap in its own cache. Ruff, layout and diff checks passed. Unversioned follow-up evidence is under `$JITTOR_LAB_ROOT/_state/flash-artifact-closure/20261016/`: `coherent-variant-red-v1.json`, clean-processor `coherent-variant-green-v2.json` and `postprocessor-manifest-v1.json`. The follow-up manifest binds 2895 postprocessor Git items and 96 evidence files, SHA256 **`42d02a6b12d59fe3f16571bf4c9b586de8e87854e1a7b98a6e45ff386faa9da7`**. The guard is admission-only: the original 48 workers, 936 numeric checks, measured source and original manifest were not rerun or relabeled.

The central integration independently rehashed all 17666 measurement/external files
(9467317936 bytes), all 96 postprocessor artifacts (606105132 bytes), and both
complete Git trees against their original blobs. It independently repeated all
720 array comparisons, 216 update checks and 24 perturbation checks under the
original budgets. These checks preserve measurement `c33627faaf99f1cc9f422421f90c3896d4cbc326`
and the separate artifact-guard baseline; they do not relabel either as the integration commit.

The integration also reproduced a collector defect: executing the current collector
with `--root` pointing at the frozen measured checkout incorrectly checked the
collector's own framework tree and imported the historical validator. The repair
threads the measured root through common protocol validation while loading the
current collector, comparator and fixed-artifact guard from the auditor tree.
`--root` must identify a clean tracked checkout at the exact full `--baseline` SHA;
its original probe hash comes from that Git commit. Other protocol validators keep
their existing call contract. A historical checkout cannot select an older guard.
Root-owned audit scripts, failed attempts and rechecks are retained unversioned at
`$JITTOR_LAB_ROOT/_state/flash-main-integration-4ce/20261016/`.

Final integration verification: 182 targeted host checks, CPU core 322 passed /
42 skipped / 1 existing xfail, and expanded structure 1628 passed / 6 skipped.
Layout, manifest and Ruff checks passed; 72 unrelated nox function bodies are
unchanged. The current-guard collector completed the full twelve-case recheck:
0 accepted, 12 blocked, 0 rejected, with every numerical/protocol/platform field
matching the frozen positive control. The coherent 78-member counterfeit wheel
and its valid installed RECORD were rejected by the current fixed-artifact guard.
The postprocessor ran before the integration commit; its exact owner bytes and
rechecks are recorded separately from the original clean measurement baseline.
