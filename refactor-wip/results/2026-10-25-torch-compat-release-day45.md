# Torch compatibility release acceptance: 2026-10-25 Day2–5 continuation

## Status

**未完成；Day2–3 仍未通过，Day4–5 只有共享 GPU 诊断结果。** 本记录不改变原有门槛、seed、比较字段或失败证据。源码与候选 HEAD 在本轮运行期间冻结；主树仍保留 dirty/untracked 工作。

## Baseline and live-state audit

- origin/cgq_transformers was fetched at e5cffe1ae60c418f53c5bb1cdca9182577fd2b81; no origin/2.0-refactor integration was attempted.
- Main worktree remains integrate/cgq-transformers-native-multicard-target at 2690b03e3ff489134c093eaf5c4a8fd2af53fd5c. The two acceptance worktrees remain at release/torch-compat-day23-20261020 and release/torch-compat-day23-fix-20261021, both at 9e0e187bb013e71d85111acfd668e46aeff9f9d2; dirty/untracked files were preserved.
- T5 r36 completed its identity-checked continuous and fresh-process resume captures under t5-resume-diagnostic-r36, using CUDA UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5 and a new Jittor cache. No T5 process remains active; both receipts and trace artifacts were rechecked before the diagnostic comparison.
- Root had about 113 GiB free and /data6 about 6.0 TiB free. No cache or checkpoint was deleted; active holders and all evidence remain intact.

## Day2–3 evidence

### T5 BF16 step-51 diagnostic r36

r36 is the new isolated cohort intended to capture functional dropout input and encoder block 0 FFN wi/activation boundaries. Both continuous and fresh-process resume captures completed at step 51 with sample ids [14,136], actual cuda:0, fallback=0, and unchanged frozen sources. The strict comparator remains FAIL because the new pre-forward parameter signatures do not match the retained r6 reference signature; this fail-closed check is preserved. The retained paired diagnostic is paired-diagnostic-r36.json under the r36 artifact directory. It records continuous top-level and encoder block 0 as eval while resume is train. The first paired trace divergence was encoder block 0 FFN wi input (1449 elements, max abs 1.430511474609375e-06), after earlier dropout and attention events were exact; later wi inputs diverged as well. This is a mode/trajectory diagnostic, not a release pass. The retained strict gate remains FAIL (771/2088 mismatches for the historical seeds), and r35's incomplete epoch-1 run is retained.

### Qwen BF16 reduction r24

PT and JT replays ran serially on the same A800 UUID GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068, with the frozen r22 input (input_sha256 f2df7b930b0f2227dbea25b0a95bd3bd6bc2dfe811a328d0fe7389c0e089cd13) and manifest (manifest_sha256 ce6872e7e6fc2c21008d488c058af7f610a303d1a9b74a77ec5a62a2e3759ea0). Both report actual_device=cuda:0 and fallback_count=0; PT/JT array hashes are f50b79258c22ac0931759ed5eb41ddc47779de13bf89a8d3a8ec83101aaf2f84 and 14f8667360b75b9a1248a6d06d136cc2be20d922281a2310ef7529fb37893208. pow2 is exact, while native mean, sum(last_dim)/N, CUDA-scalar division, and epsilon variants all retain 1046/3072 differing values with maximum absolute difference 1.4901161193847656e-08. This does not close the full BF16 logits gate; LoRA full/resume remains blocked.

### Vision and BERT

The r18 wrapper failure due to legal missing torch.__file__ remains preserved; r19's getattr measurement fix and r21/r22b cohorts remain separate evidence. Vision remains FAIL: same-runtime/cross-runtime failures are retained. BERT BF16 remains FAIL because the frozen two-seed task-loss gap is 15.425953905%, above the unchanged 15% threshold. No seed, tolerance, or checkpoint audit was changed.

## Day4–5 performance and memory

The independent day45-bert-release-r3 cache and output directory contain real AdamW updates (parameter_changed=true), matched FP32 weights/inputs/device/optimizer, synchronized timing, 10 samples after warmup, three PT/JT alternating rounds for representative batch=8, seq=256, and two rounds for legal near-capacity batch=64, seq=512. CUDA device is 0, Jittor fallback_count=0; allocator allocated/reserved peaks and NVML process peaks are recorded per run. Representative JT/PT median ratios are 1.409, 1.510, and 1.341 (p95 ratios 1.398, 1.478, and 1.427). Near-capacity median ratios are 0.746 and 1.132, with p95 ratios 1.147 and 1.136. The invalid batch=64, seq=1024 attempt is retained as a fail because BERT has 512 maximum positions.

All these runs saw strong shared GPU load (summary status DIAGNOSTIC_SHARED_GPU), so they are diagnostic only and do not satisfy the release speed/memory gate. Existing pilot/protocol outputs remain diagnostic and were not overwritten. A clean isolated performance window and any common-path regression repair still require a later run.

## Reproduction and review

Raw artifacts are under /data6/cgq/jittor-lab/_state/release-day23/20261020-r3/: t5-resume-diagnostic-r36/, qwen-rmsnorm-reduction-r24/, day45-bert-release-r3/, and the retained Vision cohorts. Recheck all process identities and receipts before continuing. The release gate is not closed until T5, Qwen, Vision, BERT, and an uncontended Day4–5 measurement satisfy their original criteria.
