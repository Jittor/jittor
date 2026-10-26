# Qwen3-8B: Ascend 310P3 FP16 inference comparison

- Status: real-NPU steady-state performance at two input lengths and cross-framework parity passed
- Date: 2026-10-13
- Jittor baseline: `5b0d2fd31f810ef892b05b29120a01ef0217dd3c`
- Baseline composition: earlier 310P optimization `aad488adc`, integrating remote `3495d443d`; only added a lazy import for the SoC query to satisfy the new structure gate
- Maintainer: Jittor core maintainers
- Scope: single card, FP16, batch=1, dynamic KV cache, greedy, fixed 32-token generation
- Review when: model snapshot, framework/CANN/driver versions, device, precision or generation configuration change

## Conclusion

Qwen3-8B runs fully on both sides. At 502-token input Jittor decode is **1.8%** slower than native Torch and full generation about **0.7%** slower, roughly level this round; at 1998-token input Jittor decode is **20.1%** faster and full generation time **25.4%** lower. 8B short input did not reproduce the earlier 0.6B speed lead, and the small-model conclusion cannot be extrapolated to all models.

| Actual input tokens | Metric | Jittor | native torch_npu |
| --- | --- | ---: | ---: |
| 502 | generate(1), seconds | 0.37607 | 0.41166 |
| 502 | decode, token/s | 9.20 | 9.37 |
| 502 | generate(32), seconds | 3.74550 | 3.72041 |
| 1998 | generate(1), seconds | 1.44814 | 2.43054 |
| 1998 | decode, token/s | 8.81 | 7.34 |
| 1998 | generate(32), seconds | 4.96680 | 6.65605 |

Each case has 5 warmups and 7 timed runs, with the median in the table, synchronizing the device before and after each sample. decode is defined as `31 / (median(generate(32)) - median(generate(1)))`. `generate(1)` is first-token latency including the generation pipeline; the tool separately stores a direct `model(...)` forward computing logits at all positions, not mixed with this. Model loading, weight conversion and first JIT are excluded from steady state. This is one sequential round; small differences are not treated as stable cross-environment advantages.

## Environment and baseline choice

| Item | Configuration |
| --- | --- |
| NPU | single Ascend 310P3, device reports 44213 MB total memory |
| Driver / CANN | 26.1.0.b050 / 9.1.0 |
| Python / NumPy | 3.12.3 / 1.26.4 |
| Jittor | 2.0.0, standalone Torch frontend, source commit above |
| Native reference | PyTorch 2.12.0+cpu / torch_npu 2.12.0, separate interpreter |
| Transformers | 4.56.2 on both sides |
| Model | Qwen/Qwen3-8B, snapshot `b968826d9c46dd6066d109eabc6255188de91218` |
| Structure | 36 layers, hidden=4096, intermediate=12288, H=32, KV=8, head_dim=128 |
| Original parameter count | 8,190,735,360; same local checkpoint on both sides, loaded as FP16 |
| Generation | thinking disabled, greedy, forced 32 new tokens |
| FP32 RoPE matmul | both sides explicitly allow HF32 |

Jittor uses SDPA and `--freeze-linear`, native uses eager, `--torch-jit-compile off --torch-pack-linear`. Both pre-convert 253 Linear layers, dynamic cache, explicit `compile=none`; Jittor keeps the default `auto_graph_replay=1`. Native eager and SDPA were measured separately; eager was faster in both cases, so it is the main-table baseline. Native SDPA results under the same protocol follow:

| Input tokens | decode, token/s | generate(32), seconds |
| --- | ---: | ---: |
| 502 | 8.38 | 4.22386 |
| 1998 | 4.77 | 10.16345 |

Both sides ran in separate processes sequentially; no other NPU tests or CPU compilation for this task ran during timing. The device retained one existing service process whose utilization was 0 before the test; it was not stopped. This run does not involve training, quantization, concurrent serving or other inference engines.

## Correctness and startup cost

The full token sequence for both main prompts across all warmups, timing and validation is bit-identical across frameworks; two more prompts (arithmetic and capital of France) each reproduce generation identical to the dynamic-cache reference and are bit-checked against the native reference. Last-position logits are all finite, argmax identical, relative L2 below the preset 1% threshold. Jittor ran under `backend_fallback=error` and a count guard; the final count was **0**. Native Torch has no corresponding fallback counter, and the JSON records `fallback_count_available=false`.

| Input tokens | logits relative L2 | max absolute error | argmax |
| --- | ---: | ---: | --- |
| 502 | 0.004847 | 0.08203125 | identical |
| 1998 | 0.003125 | 0.06445312 | identical |

Jittor weight packing takes about 1.01/1.01 seconds; native eager about 1.18/1.00 seconds. Jittor's converted Parameter element count is 622,638,080, the remaining linear weights stored as buffers; the model was not pruned or quantized; native still stores NZ weights as Parameters. 8B's embedding and lm_head do not share weights originally.

Startup cost is separate: in the short-input process, Jittor model load/migrate/convert totals 33.90 seconds versus 5.53 for native; Jittor had also earlier compiled the core in a separate empty cache. This round did not do strict peak memory sampling, so the steady-state speed conclusion cannot be taken as a startup or memory advantage.

## Reproduction

Prepare two separate interpreters, the same Transformers and model snapshot per the [310P guide](../guides/ascend-310p.en.md), and load the CANN environment. The run directory and interpreters below are provided by the environment; the two frameworks run sequentially.

```bash
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python backend_fallback=error \
  "$JITTOR_PYTHON" "$perf_script" --backend jittor --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation sdpa \
  --freeze-linear --validate-prompts \
  --logits-output "$RUN/jittor-sdpa-512-logits.npy" \
  --output "$RUN/jittor-sdpa-512.json"

"$TORCH_PYTHON" "$perf_script" --backend torch --model "$QWEN_MODEL" \
  --dtype float16 --allow-hf32 --context-tokens 512 --new-tokens 32 \
  --fixed-length --warmups 5 --samples 7 --attn-implementation eager \
  --torch-jit-compile off --torch-pack-linear --validate-prompts \
  --logits-output "$RUN/torch-eager-512-logits.npy" \
  --output "$RUN/torch-eager-512.json"

"$NUMPY_PYTHON" agent/skills/jittor-transformers-perf/scripts/compare_qwen3_ascend.py \
  "$RUN/jittor-sdpa-512.json" "$RUN/torch-eager-512.json" \
  --output "$RUN/parity-512.json"
```

Replace `512` with `2048` for long input; change native to `sdpa` and use different output names to retest the other configuration. The tool does not force the prompt to exactly the requested length; on this snapshot they are 502/1998 tokens.

Raw data is unversioned, in `$JITTOR_LAB_ROOT/_state/qwen3-8b-310p/`: six benchmark JSONs in `bench/`, two `parity-*` JSONs and logits; `logs/` holds raw logs; `checkpoint_manifest.json` records the same snapshot's file sizes, cache blob identities, and config/tokenizer hashes. The run entry is in the lab's `qwen3-8b-310p/`. The worktree was still being integrated during the native test, and the benchmark script used matches the baseline above; during the Jittor test the worktree was committed and clean.

CPU core on the integrated baseline: 322 passed / 42 skipped / 1 xfailed; full structure gate 1429 passed / 4 skipped, native dtype recorder 5 passed. The first structure failure came from an import cycle in the SoC query module; re-verified after the lazy import, without relaxing gate thresholds.
