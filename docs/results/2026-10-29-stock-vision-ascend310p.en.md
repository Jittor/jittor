# Ascend 310P: four complete vision models, compatibility and performance

- Status: Three models pass full-output and real-image comparisons; RT-DETRv2 executes but fails ordered outputs and is excluded from qualified rankings
- Reviewed: 2026-10-29 (America/Los_Angeles)
- Implementation: `56ee4e59e3ed634af27fe328c9a19194b802847e`; all 28 final forward reports were measured from a clean worktree
- Sync scope: upstream `4543b0f2b1ebda3382c67670dd3f12c105ad3f7e` integrated; three fetch attempts after resuming failed due to networking, so newer remote state is not claimed
- Scope: One Ascend 310P3, complete pretrained classification, depth, and detection forwards; excludes training and annotated-dataset accuracy
- Maintainer: ACL backend and Torch compatibility maintainers
- Review when: CANN / driver, dispatch, dtype rules, checkpoints, Transformers, or replay policy changes

## Final performance

Explicit Depth Anything V2 SDPA is **1.65× faster** than native eager, the faster native configuration measured here.
ConvNeXtV2 and SwinV2 still trail native Torch. All four retain stock forwards and task heads.

| Model / input (batch=1) | Torch eager | Jittor eager (tool default) | Jittor SDPA (explicit) | All-array comparison |
| --- | --- | --- | --- | --- |
| ConvNeXtV2 Tiny / 224² | 20.050 ms | 22.391 ms | No corresponding branch | PASS |
| SwinV2 Tiny / 256² | 11.712 ms | 33.333 ms | No corresponding branch | PASS |
| Depth Anything V2 Small / 518² | 44.671 ms | 67.931 ms | **27.050 ms** | Eager / SDPA PASS |
| RT-DETRv2 R18vd / 640² | Excluded | Excluded | No corresponding branch | Ordered FAIL in both rounds |

Times are **pooled medians of 14 samples**, seven per round, not averages of two medians; FP16, no packing.
Native Depth SDPA is 77.273 ms. Matched-attention comparisons also pass, but its slower denominator must not be used to claim 2.86×.
RT-DETR diagnostic medians are native 41.553 / Jittor 75.751 ms; outputs fail, so `qualified_speedup=null`.
The comparison covers this round's eager / SDPA and separate packing experiments, not a global optimum over Torch compile or custom kernels.

## Automatic optimizations and model-code boundaries

Covered Wan2.2 and vision forwards need no changes, and no model adapter is added. Torch programs still need
an installed and activated `jittor.compat.torch` shim, NPU selection, supported dtype, and no-grad inference.
Native Jittor and the shim share applicable backend paths; arbitrary expressions are not guaranteed to become fused ops.

| Mechanism | Automatic or explicit |
| --- | --- |
| GELU / RMSNorm / attention / unit division, public semantics fixes, and new operators | Automatic dispatch when op, shape, dtype, grad, and SoC conditions hold |
| Executor / device graph replay | Attempted automatically; budgets or resources can refuse device recording while retaining applicable execution paths |
| `freeze_linear_layers_` / `--pack` | Explicitly freezes and prepacks FP16 Linear; not transparent by default and unsuitable for trainable weights |
| Previous Wan 8 MiB input threshold / SDXL 1 MiB recording budget | Explicit settings, without changing production defaults or model forwards; unused by these four models |
| Depth `--attention sdpa` | Explicit public model configuration; this tool forces eager by default, which is not HF's automatically selected default |

Merge integration retains upstream 910B implementations. The 310P raw PFA, small-width RMSNorm rejection,
and new ConvTranspose2d, bilinear resize, grid sampling, GatherV2, and TopK admission have SoC / inference guards.
Frontend argument / dtype, normalize, argsort construction, and canonical module dispatch are shared changes.
The ConvTranspose2d module's new dispatch query can also reach an existing CUDA handler: CPU contracts pass,
but actual CUDA execution is unverified. 910B and ROCm hardware are unverified; the new SDK inference
routes do not admit gradients. 310P results are not acceptance evidence for other hardware. See [deferred hardware validation](../../agent/manuals/deferred-hardware.md).

## Verified fixes and bottlenecks

- Correct all / any dim / keepdim; retain low-precision frontend exp / pow output dtypes; use scalar clamp in normalize
  so Python epsilon does not promote FP16 norms. Native exp / pow whitelist policy is unchanged; gradients / active epsilon are checked.
- Add 310P no-grad ConvTranspose2d and bilinear resize SDK routes so the complete Depth head executes.
  Previously ignored `interpolate(antialias=True)` is now explicitly rejected; antialias remains unimplemented.
- Forward stable through the argsort constructor; ACL `stable=True` remains unsupported. Grid sampling covers
  FP16 / FP32, 4-D, bilinear / zeros / align_corners=False. No raw-SDK zero-copy claim is made.
- Swin uses GatherV2 for one nonempty 1-D integer selector with full slices on other axes instead of full-coordinate
  advanced Index. No input-value or model-table cache is used. Gradients, 910B, bool / multiple indices / partial slices retain the old path.
- Generic TopK uses partial selection instead of full sort+slice for 310P no-grad, contiguous 2-D FP16 / FP32,
  last-axis positive k, largest / sorted=True. Values / indices are bitwise equal to native NPU for 22 same-input combinations.
  Old compat indices were already equal on those inputs: this is not a tie fix.

Controlled dirty-stage whole-Swin A/B is **46.033 → 33.564 ms (1.37×)**, pooled across 30 samples,
with bitwise-equal outputs for two inputs. Final clean latency is 33.333 ms, still behind Torch.
The stock model explicitly creates CPU attention masks. Whole device recording refuses the host ternary,
while executor replay remains active; zero fallback does not mean every operation executes on NPU.
ConvNeXt profiling identifies convolution as its main cost; per-op synchronized profile wall time is not model latency.
Public TopK microbenchmarks improve 1.36–1.70× over the old route, median ratio 1.58× across 22 cases; no model-wide gain is inferred.
Separate packing experiments show no stable gain; native packed ConvNeXt fails on an unsupported SDK layout and is excluded from the main table.

RT-DETR intermediates show many tied scores and sensitivity in candidate sets / order; the full difference is not completely isolated.
Random logits relative L2 is 0.048–0.054 and boxes 0.164–0.203. After TopK, cat FP32 still fails (0.050188 / 0.12894).
Candidate-ID alignment is diagnostic only; an unisolated native intermediate difference does not establish a vendor bug.
Invalid indices are documented in [KI-BACKEND-024](../../agent/manuals/known-issues.md#ki-backend-024-acl-advanced-indexing-does-not-reject-out-of-bounds-integer-indices):
old Index and GatherV2 return incorrect data where CPU raises. Invalid cases do not establish support.

## Pinned weights and environment

| Official model ID | Pinned revision |
| --- | --- |
| `facebook/convnextv2-tiny-1k-224` | `f4db009e63145e02b3c075aa64d90ce41bcca4b1` |
| `microsoft/swinv2-tiny-patch4-window8-256` | `40213dad8563a5a916d434a1291443c0fa6358f0` |
| `depth-anything/Depth-Anything-V2-Small-hf` | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| `PekingU/rtdetr_v2_r18vd` | `5650961749fa93567c0d46fc7f43ea4f9e914107` |

External `model_paths.json` retains complete hashes of three safetensors files and the official Swin bin.
Independent Torch loads Swin with `weights_only=True` and saves safetensors with all 245 tensors unchanged;
both runtimes use the same converted weights. Hardware is one 44 GB Ascend 310P3, CANN 9.1.0, driver 26.1.0.b050;
native PyTorch 2.12.0+cpu / torch_npu 2.12.0, Transformers 4.56.2, Diffusers 0.35.1.
A background service retains about 2.7 GB; physical exclusivity is not claimed. Native jit_compile=False and
matmul / conv HF32=True; Jittor acl_allow_hf32=1, AMP off, automatic replay input 4 MiB / retention 256 MiB.

Separate interpreters / caches use the same card sequentially. Initial JIT, profiles, and oracles are separate;
no task-owned compiler or NPU job runs concurrently with accepted samples. Seed=20261027, two random-input slots,
separately recorded cold inference, then 5 warmups / 7 individually synchronized samples per round.
Round one is ConvNeXt → Swin → Depth → RT-DETR, native then Jittor; round two reverses models / runtimes,
including additional Depth configurations. Load, packing, cold, preprocessing, and output CPU postprocessing are excluded.
ConvNeXt / Depth and RT-DETR activate device graphs in both rounds; Swin refuses as described above.
All 28 reports have the implementation SHA and empty-diff SHA256. Jittor forbids fallback, checks device residency,
and records zero fallback deltas. Native counters are inapplicable (null), not a Jittor-counter pass.
Full random outputs require relative L2≤0.02, matching shape / finite / all fields; tiny CPU uses≤1e-4 and cannot replace full-model acceptance.

## Real-image outputs

Both sides use one 432×532 cat image, SHA256 `f311fbd22e8a0562a237f0985abe185b2dba82435acebd9b452f1dbdd4387dda`.
Independent CPU preprocessing creates shared NPZ inputs; both sides read the same archives, and all four main archives remain unchanged.
Sidecar / archive / spec / report bind image and input hashes, shapes, and checkpoint. Renderer negative controls
reject all three wrong-archive / checkpoint / shape substitutions. Both sides use matching attention (Depth SDPA)
and independent native CPU FP32 official postprocessing. All four visualizations pass checks for missing / cropped content.

| Real-image result | Numerical and visual agreement |
| --- | --- |
| ConvNeXtV2 | Logits relative L2 0.00275893; both top-1 Egyptian cat; identical top-5 order, overlap 5/5 |
| SwinV2 | Logits relative L2 0.00377682; both top-1 Egyptian cat; identical top-5 order, overlap 5/5 |
| Depth | Aspect-preserving 420×518, raw relative L2 0.00137838; official resize to 432×532 gives 0.00133802; shared color scale looks close |
| RT-DETRv2 | Ordered logits / boxes relative L2 **0.07662588 / 0.21533035, FAIL**; at threshold 0.3 both return one cat, no unmatched detections, IoU 0.99899117, score difference 0.00102407 |

Close detection postprocessing does not replace full ordered-array acceptance. Single-image implementation agreement
is not ImageNet accuracy, COCO mAP, ground-truth depth accuracy, or metric distance. Non-square Depth real-image
latency is not mixed with the random 518² table.

## Validation and failures

| Session / evidence | Actual result |
| --- | --- |
| Four tiny CPU models | All pass; initial RT-DETR fixture has 84 tied scores, so the same public config uses initializer_range 0.01→0.1 for a nondegenerate fixture; pretrained weights and output ordering are unchanged |
| All / any | Independent CPU 177 contracts: 115 mismatches→zero; targeted CPU 41 / NPU 41 passed |
| Dtype / normalize | CPU 42 passed; mixed 71=42 CPU+29 real-device; eight independent Torch / JT×CPU / NPU×FP16 / FP32 processes complete |
| ConvTranspose2d / bilinear | SDK 14 / 32 combinations; public session 84=81 ACL-suite+3 CPU; antialias 4=2 CPU+2 ACL; rejection tests are not numerical support |
| Grid sampling / TopK | Grid 16 SDK, 26 NPU / 7 CPU / 39 host; TopK SDK / public each22 same-input combinations, native NPU50 / CPU17, compat CPU3 / NPU-session5, host39; all zero skipped, overlapping counts not summed |
| CPU core | Native 98 passed / 44 skipped / one xfailed; Torch 229 passed / one skipped; both exit zero |
| Final structure and additional CPU | Full Torch structure+7 compat CPU: 1586 passed / 4 skipped; native-only recorder / graph / capture / dispatch: 80 passed / 66 GPU-capability skipped, not CUDA validation |
| Final NPU / compat | Native real NPU 411 passed / zero skipped; mixed compat125 passed / zero skipped includes CPU, not125 NPU cases |
| Tool boundaries / static | Four strict-FP32 CPU guards, four CLI boundaries, three renderer negative controls pass; producer arrays unchanged; source layout495 active Markdown / diff pass |

The core xfail is existing KI-TEST-001 swap; core skips retain capability / opt-in / device-count conditions,
and the four structure skips retain declared conditions. First structure results of 5 failed / 1574 passed / 4 skipped
are preserved: import cycles, source-owner lists, and duplicate test names are fixed without relaxing the ratchet.
Initial NPU409 passed / one failed incorrectly assumed NumPy readback migrates its source. Repaired tests verify
unchanged residency / pointers and restoration after explicit CPU migration; production behavior is not altered to satisfy that test.
Original Swin dtype, Depth transpose / resize, and RT-DETR argsort / grid failures are archived; old dirty timings are not relabeled clean.
[KI-BACKEND-019](../../agent/manuals/known-issues.md#ki-backend-019-acl-argsort-constructor-repair-still-needs-the-original-910b-and-dtype-acceptance)
still requires the original 910B / float64 gate.

## Reproduction and external evidence

Follow the [performance skill](../../agent/skills/jittor-transformers-perf/SKILL.md) and [310P guide](../guides/ascend-310p.en.md)
to install independent core / compat, load CANN, and externally select the same device. `TORCH_NPU_PYTHON` is independent
native Torch; `TORCH_CPU_PYTHON` is the CPU oracle without torch_npu imports; `JITTOR_PYTHON` has compat installed.
PYTHONPATH alone does not install the independent package. Variables below identify a local checkout, external lab,
and pinned checkpoint. Run the matrix serially and keep test caches separate.

```bash
perf_scripts="$JITTOR_SOURCE_ROOT/agent/skills/jittor-transformers-perf/scripts"
vision_run="$JITTOR_LAB_ROOT/_state/vision-reproduction"
mkdir -p "$vision_run"
# case_name: convnextv2 / swinv2 / depth_anything_v2 / rtdetr_v2
"$TORCH_NPU_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend torch --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/torch.json"
JITTOR_TORCH_SHIM=1 JITTOR_HOME="$vision_run/benchmark-cache" backend_fallback=error \
  "$JITTOR_PYTHON" "$perf_scripts/benchmark_generative_vision_ascend.py" \
  --backend jittor --device npu --case "$case_name" --model "$checkpoint" \
  --dtype float16 --attention eager --seed 20261027 --warmups 5 --samples 7 --slots 2 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/jittor.json"
"$TORCH_CPU_PYTHON" "$perf_scripts/compare_generative_vision_ascend.py" \
  "$vision_run/torch.json" "$vision_run/jittor.json" --max-relative-l2 0.02 \
  --output "$vision_run/parity.json"
```

Reverse model / runtime order for round two, retaining distinct files. Repeat Depth with `--attention sdpa` on each side.
Native eager versus Jittor SDPA additionally requires `--allow-attention-difference`, allowing only attention metadata to differ.
Prepare an image archive / sidecar below, then add `--input-npz "$shared_input"` to both forward commands with
**matching attention**, saving `torch-image.json` / `jittor-image.json` and retaining other timing options.

```bash
"$TORCH_CPU_PYTHON" - "$perf_scripts" "$case_name" "$checkpoint" "$source_image" "$shared_input" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from _generative_vision_cases import preprocess_image
metadata = preprocess_image(*sys.argv[2:])
Path(sys.argv[5]).with_suffix('.json').write_text(json.dumps(metadata, indent=2))
PY
"$TORCH_CPU_PYTHON" "$perf_scripts/compare_stock_vision_effects.py" \
  --left "$vision_run/torch-image.json" --right "$vision_run/jittor-image.json" \
  --image "$source_image" --input-metadata "${shared_input%.npz}.json" --threshold 0.3 \
  --source-root "$JITTOR_SOURCE_ROOT" --output "$vision_run/effects.json"
```

Unversioned evidence root: `$JITTOR_LAB_ROOT/_state/generative-vision-310p/iteration-2026-10-27/`:
`model_paths.json`, `inputs/`, `final-oct29/{baseline,commands,summary,visual-qa}.json`,
`r{0,1}-*.json` / `.json.npz`, `image-*.json` / `.json.npz`, four `effects-<case>.json.png`, and logs.
Additional SDK evidence is under adjacent `iteration-2026-10-29/`. Raw data, images, models, and caches remain outside Git;
regenerate missing artifacts with the public entry points above.
