# Ascend 310P: CPU-prefix recomputation and vision device-graph remeasurement

- Status: same-commit control reduces Swin latency 32.64%, while native Torch remains faster; complete classification/depth outputs and shared-image effects pass; RT-DETR original order still fails
- Reviewed: 2026-11-02 (America/Los_Angeles)
- Implementation: `013bc35bdeb3d460c269c85dc9781e841731764e`; all 42 final forward reports were checked against this clean SHA, harness fingerprints, no packing and zero Jittor fallback increments
- Synchronization: starting `d5bd7f551d1864f58e751e3faf5f9dc6fa4e64e6` already integrated upstream `4543b0f2b1ebda3382c67670dd3f12c105ad3f7e`; this round's 45-second fetch timed out, so no newer remote state is claimed
- Scope: stock ConvNeXtV2, SwinV2, Depth Anything V2 inference and public replay on Ascend 310P3/CANN 9.1.0; excludes complete-model training and labeled-dataset accuracy
- Owners: runtime, ACL backend and Torch compatibility maintainers
- Review when replay/placement/liveness, CANN/driver, dtype, model snapshots or Transformers changes

## Final speed under default scheduling

The main table uses identical **default CPU scheduling, four threads, parent affinity allowing CPUs 0–19 and no taskset restriction**. Each process has 15 complete warmups, 15 synchronized samples and two distinct input slots; round two reverses model/backend order. Values are pooled medians of all 30 samples. Both sides use identical pinned weights, FP16, batch=1 and no packing. JIT, loading, preprocessing and output postprocessing are excluded; every result is retained and synchronized.

| Model/random input | Native Torch | Jittor | Complete original-order outputs |
| --- | ---: | ---: | --- |
| ConvNeXtV2 Tiny / 224², eager on both | 20.013794 ms | 22.496115 ms | PASS |
| SwinV2 Tiny / 256², eager on both | 11.954828 ms | 22.398027 ms | PASS |
| Depth Anything V2 Small / 518² | eager 44.742496 ms | explicit SDPA 27.203338 ms | PASS; measured configuration 1.644743× faster |

Conv remains about 12.4% slower; Swin takes about 1.87× native Torch latency. Depth retains its previously validated explicit-SDPA configuration advantage, remeasured here rather than attributed to new code. Its main comparison explicitly permits attention metadata differences while checking all other specifications and complete outputs; slower native-SDPA image timing is not used for ranking.

Conv round medians are native 20.167/19.906 and Jittor 22.469/22.549 ms; Swin 12.161/11.565 and 22.871/22.137 ms; Depth 44.435/44.823 and 27.216/27.167 ms. October 30's Swin 33.242 ms remains a [historical result](2026-10-30-vision-ascend310p-optimization.en.md); this round's causal attribution uses same-commit toggles. Earlier generation-model scope remains in the [four-model report](2026-10-29-stock-vision-ascend310p.en.md) and its references, without remeasurement here.

## Same-commit policy control

On the same clean implementation, default scheduling and timing rules, only the common ne method and CPU-prefix graph admission change. Stock models, weights, inputs and every output remain intact. Historical ne is extracted from pinned `d5bd7f551`, avoiding a false disabled arm pointing to current optimized code. `operator_ab` records function/wrapper hashes, policies and hit counts. Shared correctness fixes stay active: `old_all` is a policy-disabled arm on current source, not a claim to rerun the entire historical revision.

| Default-scheduling Swin policy | Round 0 | Round 1 | 30-sample pooled median |
| --- | ---: | ---: | ---: |
| Historical ne, prefix recording disabled | 33.455691 ms | 33.241483 ms | 33.443680 ms |
| Current ne, prefix recording enabled | 22.791812 ms | 22.114030 ms | 22.529030 ms |

Latency falls **32.63591%**, with the same direction in both rounds. Both slots' complete logits are byte identical across policies and pass against independent native NPU. The wrapper's 22.529030 ms and the standard unwrapped main table's 22.398027 ms are different protocols and must not share denominators. Linear-query selections are zero: no new Linear dispatch was retained or credited. Current ne includes self-ne simplification, scalar placement and result-frontend corrections; the complete method-group effect is not attributed to one line.

The following separate resource budget fixes affinity to four distinct physical P cores, `0,2,4,6`. It slows native Swin, so it is not the sole performance denominator and is not divided into default-scheduling rows.

| Fixed four P cores, otherwise identical protocol | Native Torch | Jittor |
| --- | ---: | ---: |
| ConvNeXtV2, eager | 21.555060 ms | 22.362533 ms |
| SwinV2, eager | 15.035254 ms | 24.215964 ms |
| Depth, native eager/Jittor SDPA | 46.926215 ms | 27.023503 ms |

Fixed-core same-commit Swin control measures historical ne/prefix disabled 33.430506 ms, method corrections only 24.238347 ms, and current production 23.746110 ms: overall latency falls 28.96874%, with complete outputs passing. An earlier unpinned 23.6285→22.1355 ms prefix-only experiment remains stage evidence, not a replacement for final controls.

## Three retained public-layer corrections

1. **Reduce comparison graphs while preserving placement.** Operator/method self-ne on the same real Tensor reuses canonical `isnan`: true for NaNs, false for infinities and signed zeros, avoiding subtraction, absolute value and duplicate special-value graphs. Results retain the input frontend type/placement. New Python/NumPy scalars are constructed in the Tensor input's scope with existing dtype inference. Previously stock Swin's CPU-mask `!= 0` built an int32 scalar under outer ACL construction and performed D2H; the comparison now stays CPU. Existing Tensor operands are not relocated; complex values, different holders and the separate function signature do not enter the self shortcut.
2. **Recompute CPU prefixes and record device suffixes.** Admission requires a closed, explicitly placed CPU prefix entering the device suffix through explicit H2D copies. CPU math, waits and H2D run every call and remain in complete-forward latency. Mask results are not cached and `_is_constant` is not a purity proof. Random draws, foreign code and device-readback dependencies remain rejected, with ordinary executor replay available. Final Swin reports confirm an active prefix device graph and 29 device-graph replays per main-table process.
3. **Keep replay state/storage valid.** Private CPU copies preserve construction placement. Kept fused graphs also mark internal conditions kept, preventing later physical writes from saving old conditions. Previous asynchronous consumption finishes before CPU staging is overwritten; addresses are rechecked after prefix recomputation, releasing stale graphs for ordinary execution on migration, including through low-level launch. CPU intermediates/H2D destinations count toward budgets; native pins participate in `graph_check` root accounting without a second holder overwriting the original representative. Dynamic branches, queuing, migration, release and graph-check regressions exercise these contracts.

Jittor selects these routes by public operators, placement and replay conditions, without modifying Wan2.2 or vision-model inference code; Torch programs require the active shim. Existing no_grad, AMP=0, HF32 opt-in and 4 MiB input/256 MiB retention budgets remain. Swin reports a **prefix-retention budget upper bound of 18,394,432 bytes**, including CPU intermediates and H2D device destinations, potentially counting aliases repeatedly. This admission count is not measured CPU usage or a measured peak; the separately reported `estimated_retained_bytes` is not relabeled as a measured peak either. Recording omits none of the prefix work. October 30's p=2 norm/arg-reduction accuracy fixes remain baseline, covered by the [earlier report](2026-10-30-vision-ascend310p-optimization.en.md). Real hardware scope is 310P3, without inferred other-SoC, CUDA, ROCm, NPU BF16 or complete-training acceptance.

## Effects on the same real image

Both sides read identical immutable cat-image inputs and hash-bound official preprocessing metadata, using independent official CPU postprocessing. Complete classification/depth arrays retain relative-L2≤0.02, without reordering or relaxed thresholds. Cat-image SHA256: `f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda`.

| Real image | Complete-output relative L2 | Effect |
| --- | ---: | --- |
| ConvNeXtV2 logits | 0.002758931 | Egyptian cat on both; identical top-five order, overlap 5/5 |
| SwinV2 logits | 0.003420380 | Egyptian cat on both; identical top-five order, overlap 5/5 |
| Depth predicted depth `[1,420,518]` | 0.001378379 | max abs 0.0625; resized 432×532 relative L2 0.001338019 |

Both Depth image runs explicitly use SDPA for identical effect specifications; image timing is excluded from the main ranking. Depth figures share a relative-depth color range, not meters. All three figures were visually reviewed: source/native/Jittor panels, captions and classification top-five labels are complete without clipping; see unversioned `final/visual-qa.json`. Single-image agreement is not ImageNet accuracy, COCO mAP or ground-truth depth quality.

New RT-DETR diagnostics still fail: original-order logits relative L2 is 0.054352806/0.048059742 across slots, boxes 0.202647855/0.164068814. Complete arrays remain unchanged in order, with no ID alignment or relaxed thresholds, and no qualified speed ranking.

## Gates and limits

| Verification | Actual result |
| --- | --- |
| CPU core | 327 pass / 45 skip / 1 existing xfail |
| Final Torch structure session | 1646 pass / 4 skip; separate native recorder/package identity 7 pass |
| CPU replay/rematerialization | 49 pass / 44 skip; 26 pass / 13 skip |
| Separate Half CPU selection | 26 pass / 32 deselected |
| Native ACL/CPU mixed | 515 pass / 0 skip, including CPU, not 515 NPU tests |
| Compat mixed | 318 pass / 0 skip, including the 190-case mixed ne matrix |
| Public self-ne probe | Independent CPU reference; Jittor CPU/NPU each 10 fields exact; native NPU `passed=false`, `--require-exact` exits 1 |
| Public Linear probe | Jittor CPU/NPU and native NPU each 28 cases pass against independent CPU; no new Linear route |

Probes synchronize lazy outputs before residency checks; unexecuted `location=none` is not treated as a device failure. Native NPU's infinity self-comparison failure remains diagnostic evidence, not CPU mathematical acceptance, and skips are not hardware validation.

New Linear/Addmm dispatch was removed, retaining only its reusable probe. Raw native CPU scalar-ne and eq+not each disagree with CPU Torch in 14/192 fields. Existing `result_type` and an explicitly typed scalar produce 210/210 exact CPU results, including under ambient ACL, but complete Swin timing changes are +2.89%/−8.38%, pooled about −1.12% with opposite directions. That route was not adopted and establishes no NPU BF16 computation.

Five public CPU scalar eq/ne boundary cases across six spellings still expose real differences, recorded as [KI-DTYPE-005](../../agent/manuals/known-issues.md#ki-dtype-005-cpu-scalar-comparisons-differ-at-half-and-int64-boundaries). Self-ne and placement corrections do not fix every scalar signature. Existing KI-BACKEND-025/026/027 special-index, integer-value and norm limitations remain open. Swin still pays for CPU FP16 mask math, H2D and device execution. Conv convolution arithmetic is unchanged; scheduling differences are not convolution optimizations.

## Reproduction and evidence

Environment: 310P3 44 GB, CANN 9.1.0, driver 26.1.0.b050, independent Torch 2.12.0+cpu/torch_npu 2.12.0, Transformers 4.56.2 and Diffusers 0.35.1. Background services were retained; physical exclusivity is not claimed. Pinned snapshots are Conv `f4db009e63145e02b3c075aa64d90ce41bcca4b1`, Swin `40213dad8563a5a916d434a1291443c0fa6358f0` (losslessly converted 245-tensor safetensors), and Depth `5426e4f0f36572d16453bbda7a8389317b1bef99`; model IDs remain in the [October 30](2026-10-30-vision-ascend310p-optimization.en.md) and [four-model](2026-10-29-stock-vision-ascend310p.en.md) reports.

Configure independent interpreters, CANN and external lab through the [performance skill](../../agent/skills/jittor-transformers-perf/SKILL.md). Use standard `benchmark_generative_vision_ascend.py` on both sides with identical `--case`, `--model`, `--source-root` and `--dtype float16 --seed 20261027 --warmups 15 --samples 15 --slots 2`. Retain default affinity/four threads for the main table. Conv/Swin both use `--attention eager`; Depth uses native eager/Jittor sdpa, with `--allow-attention-difference` only for that comparison. The comparator retains `--max-relative-l2 0.02`. Reverse execution order in round two and save distinct reports. Image runs pass identical `--input-npz`; both Depth image runs use sdpa.

Unversioned evidence root: `$JITTOR_LAB_ROOT/_state/generative-vision-310p/iteration-2026-11-02/`. `final/summary.json` combines 42 forwards, default/fixed-core protocols, arrays and controls; `final/default-{baseline,commands}.json` and `final/{baseline,commands}.json` record resource budgets; `final/effects-*` and `visual-qa.json` preserve three-image effects/QA. `final-gates/`, `final-probes/` and `final-source-freeze.json` retain gates, probes, commands and source fingerprints. `final-probes/verification.json` explicitly indexes final 28-case parity, the initial lazy-residency guard failure, successful reruns and expected negatives. The original `commands.json` exit 1 remains; not every attempt is claimed successful. `cpu-ne-scalar/` preserves rejected candidates; preliminary diagnostics do not replace the production table. Raw models, caches, images and logs remain outside the repository.
