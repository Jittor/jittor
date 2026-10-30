# Ascend 310P: vision remeasurement and FP16 numerical corrections

- Status: no production speed breakthrough; FP16 extrema and single-axis norm accuracy corrected; complete ConvNeXt/Swin outputs and real images pass; RT-DETR original-order acceptance remains failed.
- Reviewed: 2026-10-30 (America/Los_Angeles).
- Implementation: `02b021cba3c22f7907ae283610d82fe6dd6fb4de`. All 16 final forward reports were verified against this SHA, an empty dirty diff and no packing; Jittor fallback increments are zero.
- Synchronization: starting baseline `f283ba92d` already integrated upstream `4543b0f2b1ebda3382c67670dd3f12c105ad3f7e`. Two fetch attempts timed out after 45 seconds each; no newer remote synchronization is claimed.
- Scope: public operator correctness, two complete classification-model remeasurements and RT-DETR diagnostics on Ascend 310P3/CANN 9.1.0; excludes complete-model training and labeled-dataset accuracy.
- Owners: ACL backend and Torch compatibility maintainers.
- Review when CANN/drivers, dtype/dispatch, checkpoints, Transformers or replay policy changes.

## Final performance and images

ConvNeXtV2 and SwinV2 remain slower than independent native Torch. Controlled enabling/disabling of the standard production norm query establishes no stable speed improvement. The route is retained for genuine finite-FP16 accuracy corrections. Preliminary LAB candidate gains do not count as final production acceleration.

| Model/input, batch=1 | Torch | Jittor | Complete outputs |
| --- | ---: | ---: | --- |
| ConvNeXtV2 Tiny / 224² | 20.002 ms | 22.010 ms | PASS |
| SwinV2 Tiny / 256² | 11.803 ms | 33.242 ms | PASS |
| Depth Anything V2 Small / 518², October 29 reference | eager 44.671 ms | explicit SDPA 27.050 ms | Previous PASS, 1.65× faster; not remeasured |
| RT-DETRv2 R18vd / 640² | Excluded | Excluded | Original-order FAIL |

The first two rows use raw FP16/eager/no packing: **14-sample pooled medians**, seven samples in each of two rounds, not averages of round medians; each process has five warmups and two distinct random slots. Conv round medians are native 20.129/19.979 ms and Jittor 22.124/21.846 ms; Swin native 11.880/11.796 ms and Jittor 33.421/33.197 ms. The second round reverses both model/backend order. Initial JIT, loading, preprocessing and output postprocessing are excluded from steady state.

Only the affected Conv/Swin models are remeasured. Depth's complete configuration and earlier generation results remain in the [October 29 four-model report](2026-10-29-stock-vision-ascend310p.en.md) and its references. These retained values are not new gains, and slower native Depth SDPA is not used to inflate the ratio.

Both image comparisons use the same shared cat-image archive and independent CPU official postprocessing. The two figures were visually checked: source/native/Jittor panels and all top-five labels are visible without missing content or clipping.

| Real image | Logits relative L2 | Top-1 | Top-5 |
| --- | ---: | --- | --- |
| ConvNeXtV2 | 0.002758931 | Egyptian cat on both sides | Identical order, overlap 5/5 |
| SwinV2 | 0.003420380 | Egyptian cat on both sides | Identical order, overlap 5/5 |

All ten random/image complete-array comparisons meet the existing relative-L2≤0.02, shape/finite/all-field checks, including four production-A/B comparisons below. Outputs are not reordered and thresholds are unchanged. Cat-image SHA256 is `f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda`. Single-image agreement is not ImageNet accuracy, COCO mAP or ground-truth depth quality. Depth/RT image effects were not remeasured.

## Standard production A/B establishes no speed gain

On the same implementation commit, toggle only the canonical norm query, retaining stock models and original outputs. `operator_ab` metadata records the policy, hit count and diagnostic-wrapper hash. Disabled uses the old expression; enabled uses production, with zero/72 hits respectively.

| Swin production query | Round 0 | Round 1 | 14-sample pooled median |
| --- | ---: | ---: | ---: |
| Disabled, old expression | 33.566 ms | 33.333 ms | 33.431574 ms |
| Enabled, production | 33.615 ms | 33.164 ms | 33.471810 ms |

Production is approximately **0.12% slower**, within this run's variability, so no stable improvement is claimed; round directions also differ. All original-order logits pass against native. Earlier operator-substitution LAB prototype results were 33.451848→32.714397 ms, approximately 2.20%, followed by a single public stage at 33.008215 ms. Neither establishes final production improvement; the faster round is not selected in place of the complete comparison.

Conv device graphs are active. Swin retains executor replay but whole-device recording refuses the stock forward's CPU `ternary[4,64,64]` mask. No unsafe CPU constant folding or forced recording is used. Zero fallback does not mean the model performs no explicit CPU work.

## The norm route is retained for accuracy

One canonical `_try_vector_norm` query is shared by native `linalg.vector_norm`, `nn.normalize(p=2)` and the Torch norm signature adapter. The standard ACL runner admits only **Ascend310P/no_grad/real FP16 or FP32/positive shapes/p=2/a canonical single final axis**. Strided inputs get an explicit contiguous copy included in measurements. Other SoCs, dtypes, orders, axes and gradients retain their existing expressions; no model forward, checkpoint or adapter changes are made.

Independent Torch CPU and native NPU oracles verify corrections to these **genuine finite results**:

| FP16 input/operation | CPU/native NPU/new route | Old Jittor |
| --- | --- | --- |
| Norm of `[1000,1000,1000]` | 1732, representable in FP16 | 255.875, incorrect after intermediate-square overflow |
| Default normalize of the same input | Approximately 0.577 per element | Approximately 3.908 per element |
| Norm of `[1e-5,2e-5,3e-5]` | Approximately 3.743×10⁻⁵ | Zero |
| Default normalize of the tiny input | Approximately `[0.2676,0.535,0.801]` | Saturation value 65504 |

The feature is therefore retained as an accuracy correction while its lack of demonstrated production acceleration is reported. Native-NPU agreement alone is not CPU correctness: true norms outside the half range and NaN/Inf limitations remain explicit below.

Tested finite outputs of the original 14 Swin normalize/40 GRN SDK cases are byte identical to native NPU, unchanged in dtype and fallback0. Initial default-transpose inputs were materialized. Four genuine-view stages × FP16/FP32 were subsequently rerun under `transpose_storage_view=1` and remained native-exact. FP16 true-view operator medians including copy are old approximately 404–417 μs/new 398–423 μs, with no stable operator win.

Native `vector_norm(dim=None)` first flattens to rank one and reduces axis0, so it can select this final-axis implementation. The provider directly declines `None`; directly querying compat `torch.norm(dim=None)`/`F.normalize(dim=None)` retains the old fallback. Native `jt.norm`'s square-sum epsilon floor is unchanged. GRN's multi-axis `(1,2)` does not enter production.

## Independent FP16 MaxDim/MinDim failure and correction

The model-free FP16 input `x[0,r,c]=(r % 97)/16+c/128`, shape `[1,8400,80]`, has unique extrema per row. The raw SDK returns wrong values in **84 rows**, ranges 1056–1067, 2112–2123, …, 7392–7403. At row 1056, max's true value and the value at index 79 are 5.9921875, but the SDK returns 5.3671875.

Independent Torch CPU/NumPy agree; recreating fresh contiguous native-NPU tensors from the identical NumPy input still fails; direct native Jittor `ArgReduceOpRunner`→SDK produces the same error. This excludes model-hook snapshots, readback timing, storage mutation and TopK tie policy as explanations for this local failure.

The updated portable probe runs the same five inputs for max/min. CPU passes 5/5; native Torch NPU/raw Jittor SDK agree on shifted FP16 84 wrong values/zero wrong indices and random FP16 83 wrong values/28 wrong indices. Ascending, descending and FP32 controls pass. Public Jittor max/min each pass 5/5 with `passed=true` and `--require-exact` exit 0. Raw SDK `passed=false` remains diagnostic failure, not public acceptance.

The existing `arg_reduce_acl` provider computes **310P FP16 max/min** indices in FP32, then gathers values from the **original FP16 input**, reshaping through the selected input-axis geometry and restoring the original output shape. Every finite FP16 value is exactly representable in FP32. Rank-one, keepdims, value/index dtypes and gradients are preserved, without a no-grad or model-geometry special case. Other SoCs/dtypes are unchanged.

The FP32 SDK values are not simply cast back to FP16: that device-Cast experiment saturated Inf/NaN to 65504. Gathering the original input avoids that added error. Tested native/compat × max/min mixed-zero/multiple-NaN/Inf values, zero signs and indices remain unchanged. The existing first-index limitations are not corrected.

Complete-call microbenchmarks use two slots, five warmups and 15 alternating rounds per slot, retaining/synchronizing outputs and including cast/Gather. `[1,16,7]` changes **0.0856→0.1423 ms**; `[1,8400,80]` changes **0.1038→0.2570 ms**. This is a correctness cost, not a speedup or a prediction of RT whole-model overhead.

## RT-DETR remains unaccepted

The old native FP16 trace's encoder scores differ from its observed max/topk inputs at 84/8400 rows per slot, maximum absolute difference 2.361328125/2.861328125. This local SDK error is now independently isolated. Jittor compat `.max(...).values` already used a separate correct ReduceMax. After the index correction, complete stock RT-DETR encoder scores, max inputs, TopK IDs, original logits/boxes remain **byte identical** to the previous Jittor trace, fallback0.

Against the preserved native FP16 trace, the two slots' original-order logits relative L2 is **0.0543528/0.0480597**, boxes **0.2026479/0.1640688**, still failing 2%. The local half-reference SDK error is not established as the sole cause: earlier FP32 cat-image logits/boxes also fail 0.050188/0.12894, with candidate-set/order sensitivity to tied/nearby scores remaining. No ID alignment, output reordering or tolerance relaxation is used; no qualified RT speed is reported.

## Rejected performance candidates

| Candidate | Auditable result | Decision |
| --- | --- | --- |
| Dedicated ConvDepthwise2d | Byte-identical outputs; initial whole-model improvement approximately 3.1%, only 0.28% with reversed order; no stable operator signal | No new dispatch |
| FP16 Im2col+Baddbmm | Ordinary stage3/stage4 approximately 0.56/0.27 ms; generic columns differ, and the same NaN input changes 791 old Conv values, max abs 79984 | Reject |
| GatherV2+where replacing Im2col | Full indices/mask/transfer/Gather/where/contraction/sync approximately 25.86 ms, old stage3 around 1.1 ms | Reject |
| FP32 Im2col+multiply/sum | Stage3 17.79 ms versus old around 1.17 ms; 766/3687936 ordinary columns differ from CPU and zero-init does not repair them | Reject; cause unisolated, no definitive SDK-defect claim |
| Multi-axis GRN `(1,2)` norm | Whole-model LAB 21.920203→22.440176 ms, slower | Retain old multi-axis expression |

The old GRN extreme-finite precision gap remains: tested FP16 large 100×/tiny 1e-4× norm errors against native are 10.2035%/69.3944%. The SDK can improve them, but this does not establish production multi-axis or complete-model support.

## Gates and remaining limits

| Session | Actual result |
| --- | --- |
| CPU core | Native 98 pass / 44 skip / 1 xfail; Torch 229 pass / 1 skip; exit 0 |
| Complete Torch structure + 10 compat CPU | 1591 pass / 4 skip |
| Native recorder/package identity | 7 pass |
| Separate HalfCPU selection | 26 pass / 32 deselected |
| Native ACL/half mixed | 501 pass / 0 skip, including 26 CPU; not 501 NPU tests |
| Compat mixed | 128 pass / 0 skip, including CPU |
| VectorNorm targets, overlapping final sessions | 30 actual NPU / 3 compat CPU / 69 host; typed owner/registration/launcher ABI canary pass |
| Static | Source-stage layout 497 Markdown, manifest/diff and 18 Python AST pass |

The initial native-structure combination used `-m not npu`, deselecting all 58 Half tests while seven other tests passed. The execution gate correctly rejected missing required execution; a separate `-k cpu` invocation ran 26 successfully. The original attempt is retained. This was selection, not a mathematical failure. Core xfail is the existing swap condition; capability/opt-in skips are not hardware acceptance.

The following limitations remain open; see [KI-BACKEND-025/026/027](../../agent/manuals/known-issues.md):

- Multiple-NaN/mixed-signed-zero arg indices do not follow CPU first-index semantics, unchanged before/after. Raw CANN MaxDim/MinDim and torch_npu itself are not repaired by this patch.
- The existing int32→int64 arg-reduction→int32 path returns `-2147483648` for `2147483647`, with correct indices. The new half guard does not handle integers; an independent original-branch control also fails. Input cast/SDK value output/restoration cast are not separately isolated. This existing test is outside the final explicitly selected 501 tests. Its initial 40 pass / 1 fail and independent 1 fail control remain preserved; the test was not relaxed.
- FP16 norms with NaN, ±Inf or true result above 65504 produce CPU NaN/Inf but SDK 65504; old Jittor was already incorrect on these examples. Zero FP16 normalize with default eps 1e-12 produces CPU NaN and native/old/new Jittor 65504; explicit eps 1e-4 gives zero on all measured sides.
- FP32 vector-norm NaN/Inf agrees with CPU, but normalize `[NaN,1,2]` remains old/new Jittor `[NaN,1/eps,2/eps]` versus all-NaN CPU/native. The individual responsible owner is unisolated; no definite clamp cause or hardware-wide inability to support special values is asserted. The 80-field diagnostic retains `reference_matched=false`, not a CPU full-domain pass.

The public routes apply automatically when operator/dtype/axis/grad/SoC conditions hold, without editing Wan2.2 or these model inference implementations; Torch programs require the installed, active shim. Attention, packing and replay remain explicit configurations. The norm fast path is inference-only; ArgReduce retains its gradients. No complete training, BF16, 910B, CUDA or ROCm acceptance is inferred.

## Environment, reproduction and evidence

The previous pinned snapshots are retained: Conv `facebook/convnextv2-tiny-1k-224` revision `f4db009e63145e02b3c075aa64d90ce41bcca4b1`; Swin `microsoft/swinv2-tiny-patch4-window8-256` revision `40213dad8563a5a916d434a1291443c0fa6358f0`. Both use identical losslessly converted Swin safetensors, 245 tensors unchanged. Environment: 310P3 44 GB / CANN 9.1.0 / driver 26.1.0.b050, independent Torch 2.12.0+cpu / torch_npu 2.12.0, Transformers 4.56.2, Diffusers 0.35.1. Background services were retained; physical exclusivity is not claimed.

Native jit_compile=False and matmul/conv HF32=True; Jittor no_grad=1, AMP off, acl_allow_hf32=1, replay input 4 MiB / retention 256 MiB. Seed 20261027, isolated interpreters/compile caches and serial single-device runs; no other task-heavy jobs during accepted timing. Native fallback counter is inapplicable and is not described as passing Jittor's counter.

Configure independent interpreters, CANN and external lab paths using the [performance skill](../../agent/skills/jittor-transformers-perf/SKILL.md). Main-table reproduction:

```bash
perf_scripts="$JITTOR_SOURCE_ROOT/agent/skills/jittor-transformers-perf/scripts"
vision_run="$JITTOR_LAB_ROOT/_state/vision-follow-up"
# case_name: convnextv2 / swinv2; checkpoint is its pinned local snapshot
"$TORCH_NPU_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend torch --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/torch.json"
JITTOR_TORCH_SHIM=1 "$JITTOR_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend jittor --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/jittor.json"
"$TORCH_CPU_PYTHON" "$perf_scripts/compare_generative_vision_ascend.py" \
  "$vision_run/torch.json" "$vision_run/jittor.json" --max-relative-l2 0.02 \
  --output "$vision_run/parity.json"
```

Reverse backend/model order for a second round and retain different filenames. Prepare shared image archives/sidecars through the previous report's preprocessing entry; pass identical `--input-npz` on both sides and use independent CPU `compare_stock_vision_effects.py` postprocessing. Separate `probe_acl_arg_reduce.py` public/SDK routes, CPU-first, with identical max/min inputs: `--require-exact` should fail the raw SDK's known cases. Save a CPU reference before NPU `probe_acl_vector_norm.py`; ordinary diagnostic exit records differences, while `--require-match` makes mismatches fail.

Unversioned evidence root: `$JITTOR_LAB_ROOT/_state/generative-vision-310p/iteration-2026-10-30/`:

- `final/`: `baseline.json`, `commands.json`, `summary.json`, 16 forward reports/arrays, 10 parity reports, two effects/PNGs and `visual-qa.json`. A/B reports include `operator_ab`; the external lab wrapper does not replace models.
- `final-gates/`: commands/logs and the retained incorrect-selection attempt.
- `root-proofs/`: five-input max/min CPU/native-NPU/raw-SDK/public controls and arrays.
- `rtdetr/`: direct SDK, before/after special values, cost, existing integer failure, same-output trace and `rtdetr-fixed-comparison.json`.
- `vector-norm/`: finite accuracy/special values, true views, LAB A/B, public stage and source freeze; `convnext/`: all rejected candidates, `IM2COL_HANDOFF.md` and failed FP32 zero-init logs.

Raw models, caches, images and experiment logs stay outside the repository. LAB prototypes and agreement on SDK errors are not final production gains or mathematical correctness.
