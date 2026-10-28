# Qwen3-0.6B: 310P3 FP16 inference performance

- Status: Single-card, batch=1, dynamic KV cache device performance and parity passed; scope below
- Last reviewed: 2026-10-12
- Source baseline: `aad488adca252aeec92edd2edf21548cf3c321f9`; optimization implementation based on `3cfaa92e9` (including remote `7a18abf29`)
- Owner: ACL backend and Torch compatibility maintainers
- Review when: CANN, driver, Transformers, Torch, weight layout, SDPA or graph replay mechanisms change

## Results and measurement definition

On the same checkpoint, precision and device, Jittor's 502-token decode is **6.4%** faster than native torch_npu, and 1998-token decode **25.6%** faster. Full 32-token generation time drops by **5.6%** and **32.0%** respectively. Both sides explicitly pre-convert the Linear weights, synchronize before and after each generation round, and all warmup, timing, and validation outputs participate in the token consistency check.

| Input tokens | Phase | Jittor | native torch_npu |
| --- | --- | --- | --- |
| 502 | prefill: generate(1) | 0.07014 s | 0.07051 s |
| 502 | decode | **42.70 token/s** | 40.13 token/s |
| 502 | generate(32) total | **0.79607 s** | 0.84300 s |
| 1998 | prefill: generate(1) | **0.31810 s** | 0.64120 s |
| 1998 | decode | **38.45 token/s** | 30.61 token/s |
| 1998 | generate(32) total | **1.12434 s** | 1.65381 s |

The main table is the median of 5 warmups and 7 timed runs of full generation. decode is defined as `31 / (median(generate(32)) - median(generate(1)))` and must not be mixed with `32 / total_time`, which includes prefill. The tool separately stores a direct `model(...)` prefill, which includes logits at all positions and is not used in the table above. First JIT, model loading, and weight conversion are excluded from steady state.

Both sides ran sequentially; the main table uses the formal re-measurement after the RMSNorm auxiliary-statistics fix. Short-input prefill is essentially level, and a difference under 1% is not treated as a stable gain.

The native side chose binary op mode `jit_compile=False`, eager attention and NZ weights. Native SDPA with the same pre-converted weights was also measured: 502/1998-token decode was 39.48/16.17 token/s and full generation 0.902/3.060 s, so the faster eager was used as the baseline. Jittor uses SDPA and explicit `freeze_linear_layers_`; both caches are dynamic with explicit `compile=none`. Jittor keeps the default `auto_graph_replay=1`; this switch or the `compile` parameter is not used to replace real replay statistics.

## Environment and correctness

| Item | Configuration |
| --- | --- |
| NPU | Ascend 310P3, 44 GB, single card |
| Driver / CANN | 26.1.0.b050 / 9.1.0 |
| Python / NumPy | 3.12.3 / 1.26.4 |
| Jittor | this tree 2.0.0, standalone Torch frontend |
| Native reference | PyTorch 2.12.0+cpu + torch_npu 2.12.0, separate interpreter |
| Transformers | 4.56.2 on both sides |
| Weights | Qwen/Qwen3-0.6B, snapshot `c1899de289a04d12100db370d81485cdf75e47ca` |
| Inference | FP16, batch=1, greedy, forced 32 new tokens |
| RoPE FP32 matmul | both sides explicitly allow HF32; Jittor uses `acl_allow_hf32=1` |

Each long prompt plus the two short prompts (arithmetic, capital of France) produce outputs token-for-token identical to the independent native reference; repeated validation is also identical. Last-position logits relative L2 error at 502/1998 tokens is **0.003048 / 0.003917**, maximum absolute error **0.05859 / 0.07422**, argmax identical, all finite.

All Jittor computation ran under `backend_fallback=error` and a fallback-count guard; after validation and profiling the count was still **0**. Native Torch has no corresponding counter, and the JSON records `fallback_count_available=false` explicitly.

## Why the changes help

In the CANN trace of unpacked inference, TransData accounts for about 44% and 51% of accumulated device op time for Jittor and Torch respectively, and the large-vocabulary output weight is also converted repeatedly per token. The explicit snapshot arranges the FP16 weights of 197 Linear layers into NZ layout on the device once, used directly as matmul inputs. Jittor conversion takes about 0.38–0.39 s and the native reference about 0.63 s, timed separately.

The source model has 596,049,920 Parameter elements. After Jittor conversion the linear weights are registered as buffers, and the Parameter count becomes 155,648,000; the values remain consistent with the model's mathematical weights. Both sides create separate snapshots for the lm_head that originally shared storage with the Embedding, so actual storage increases. Usage constraints and the checkpoint format are in the [310P guide](../guides/ascend-310p.en.md).

310P prefill is wired to PromptFlashAttentionV3, with padding, KV head expansion, and output trimming done on the device side; decode uses IncreFlashAttentionV4. The former is a different interface from the FlashAttentionScore missing on this device; its support conditions are limited by the [CANN PFA V3 documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnPromptFlashAttentionV3.md) and real-device probes.

Packed Linear ND shape handling, no-grad Concat and FP16 RMSNorm were changed to native nodes; basic slicing, cat and RMSNorm frontend dispatch reduce repeated Python validation while retaining registry override, AMP, view writeback and training branching. cProfile was used to locate these host dispatch costs; profile time is not used in the performance tables.

The native RMSNorm entry accepts only FP16 input and FP16 weights and returns the normalized result; the auxiliary output CANN requires is kept in a private buffer. FP32, BF16 and training use the Function path; this FP16 inference speedup is not extrapolated to those paths.

The following correctness fixes are also acceptance conditions:

- When a large graph triggers memory-pool trim, the ACL raw free interface immediately releases memory still used by queued tasks; 1998-token prefill once reproduced an illegal address. It now waits for the device before actually returning the SDK allocation; normal in-pool reuse adds no synchronization. Pinned allocation waits for the device that has used it. The [CANN aclrtFree documentation](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0100.html) states explicitly that the API does not sync implicitly.
- Graph replay adds a Var-change check for registered buffers. After loading a new packed buffer it recaptures, avoiding continuing to output old-weight results; shared holders are deduplicated, and update-type state can still replay.
- On 310P, repeated CANN RMSNorm calls sometimes produced partially zero auxiliary `rstd` for small tensors; reading device memory directly confirmed the main output correct and the auxiliary value wrong. 310P training computes FP32 `rstd` independently on the device for backward; inference does not expose CANN's auxiliary result to callers.

## Reproduction

First install two separate interpreters and the same Transformers version per the [310P environment guide](../guides/ascend-310p.en.md), and prepare the same local checkpoint. Allocate separate caches for tests, benchmarks and different backends; run single-card tasks sequentially. Raw data lives in the lab; the following variables are provided by the run environment:

```bash
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python backend_fallback=error \
  "$JITTOR_PYTHON" "$perf_script" --backend jittor --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation sdpa \
  --freeze-linear --validate-prompts \
  --logits-output "$RUN/jittor-512-logits.npy" --output "$RUN/jittor-512.json"

"$TORCH_PYTHON" "$perf_script" --backend torch --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation eager \
  --torch-jit-compile off --torch-pack-linear --validate-prompts \
  --logits-output "$RUN/torch-512-logits.npy" --output "$RUN/torch-512.json"

"$NUMPY_PYTHON" agent/skills/jittor-transformers-perf/scripts/compare_qwen3_ascend.py \
  "$RUN/jittor-512.json" "$RUN/torch-512.json" --output "$RUN/parity-512.json"
```

Change `512` to `2048` and use different output names to reproduce the long input. The actual prompt length is given by the JSON's `prompt_tokens`; on this checkpoint they are 502 and 1998.

Unversioned evidence for this run is in `$JITTOR_LAB_ROOT/_state/310p-perf/`: `jittor-accepted-*`, `torch-final-*` and `parity-accepted-*` JSON/NumPy files under `bench/`; raw CANN traces and CSVs under `profile-static-api/`, `profile-torch-api/`. The full command entry is in the [performance skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-transformers-perf/SKILL.md).

## Verification scope

Targeted acceptance covered real NPU packed layout/numerics/standalone snapshot, ND and empty inputs, full PFA output, Concat/RMSNorm, private buffer lifetime, async device/pinned release, and graph replay after buffer updates. Shared frontend fast-path CPU regression: 53 passed, 11 skipped on accelerator-backend conditions. Final RMSNorm NPU regression: 48 passed, covering AMP, shape changes, replay, buffer release, post-inference training and analytical gradient checks for previously failing sequences.

Commit gate: CPU `core` native session 98 passed / 41 skipped / 1 xfailed, Torch session 224 passed / 1 skipped. Full `tests/structure` split per `process_modes.py`: Torch session 1409 passed / 4 skipped, native dtype recorder 5 passed; both sessions exited 0 under strict coverage checks. The 4 skips come from accelerator conditions and explicit declarations and are not counted as verified functionality. Repository layout and generated manifest checks passed.

This report is limited to this model, device, precision, length and generation mode; it does not extrapolate to sampling, multiple requests, training or other models. CPU and 310P were actually executed; no CUDA or multiple ACL cards were available for hardware verification in this run.
