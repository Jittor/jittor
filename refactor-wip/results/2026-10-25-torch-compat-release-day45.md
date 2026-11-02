# Torch compatibility release acceptance: 2026-10-25 Day2–5 continuation

## Status

**未完成；Day2–3 仍未通过，Day4–5 只有共享 GPU 诊断结果。** 本记录不改变原有门槛、seed、比较字段或失败证据。源码与候选 HEAD 在本轮运行期间冻结；主树仍保留 dirty/untracked 工作。

## Baseline and live-state audit

- origin/cgq_transformers was fetched at e5cffe1ae60c418f53c5bb1cdca9182577fd2b81; no origin/2.0-refactor integration was attempted.
- Main worktree remains integrate/cgq-transformers-native-multicard-target at 95d0600ba7d4888082b3a0f71bac9f2df8acd5dc. The two acceptance worktrees remain at release/torch-compat-day23-20261020 and release/torch-compat-day23-fix-20261021, both at 9e0e187bb013e71d85111acfd668e46aeff9f9d2; dirty/untracked files were preserved. origin/cgq_transformers was fetched at e5cffe1ae60c418f53c5bb1cdca9182577fd2b81; origin/2.0-refactor was not integrated.
- T5 r38 completed its identity-checked continuous and fresh-process resume captures under t5-resume-diagnostic-r38, using CUDA UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5 and a new Jittor cache. No T5 process remains active; both status receipts, logs, and trace archives were rechecked before the diagnostic comparison.
- The current host had about 108 GiB free on / and 6.0 TiB on /data6; all GPUs were under unrelated shared load during the audit. No cache, checkpoint, failed evidence, or worktree modification was deleted; no stale PID was signaled.

## Day2–3 evidence

### T5 BF16 step-51 dynamic trace r38 (2026-10-25)

r38 is a fresh isolated cohort/cache on candidate HEAD '9e0e187bb013e71d85111acfd668e46aeff9f9d2' and GPU UUID 'GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5'. Continuous and fresh-process resume both completed the step-51 capture with sample ids '[14,136]', actual 'cuda:0', fallback_count=0, and unchanged 97 frozen sources. Both module traces contain 404 events. At hook time, training/is_train/is_training and functional-dropout training=true agree for both sides, and event order, kind, shape, and device agree. The first 21 archived tensors, including encoder block 0 dropout and FFN wi input/output, are exact; the first archived tensor difference is event 21, encoder.block.1.layer.1.DenseReluDense.wi input, with 1,377 differing elements and maximum absolute difference '3.5762786865234375e-07'. The static 543-module snapshot reports continuous=eval and resume=train because it was taken after capture; it is not evidence that the forward hooks used different modes. The formal comparator still fails because the new pre-forward parameter signatures differ from the retained r6 reference, and the historical strict recovery failure 771/2088 remains open. The resume log also retains the missing tied-weight keys warning (encoder.embed_tokens.weight, decoder.embed_tokens.weight, lm_head.weight). This is a bounded localization result, not a release pass.

### T5 BF16 step-51 diagnostic r36

r36 is the new isolated cohort intended to capture functional dropout input and encoder block 0 FFN wi/activation boundaries. Both continuous and fresh-process resume captures completed at step 51 with sample ids [14,136], actual cuda:0, fallback=0, and unchanged frozen sources. The strict comparator remains FAIL because the new pre-forward parameter signatures do not match the retained r6 reference signature; this fail-closed check is preserved. The retained paired diagnostic is paired-diagnostic-r36.json under the r36 artifact directory. It records continuous top-level and encoder block 0 as eval while resume is train. The first paired trace divergence was encoder block 0 FFN wi input (1449 elements, max abs 1.430511474609375e-06), after earlier dropout and attention events were exact; later wi inputs diverged as well. This is a mode/trajectory diagnostic, not a release pass. The retained strict gate remains FAIL (771/2088 mismatches for the historical seeds), and r35's incomplete epoch-1 run is retained.

### Qwen BF16 reduction r24

PT and JT replays ran serially on the same A800 UUID GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068, with the frozen r22 input (input_sha256 f2df7b930b0f2227dbea25b0a95bd3bd6bc2dfe811a328d0fe7389c0e089cd13) and manifest (manifest_sha256 ce6872e7e6fc2c21008d488c058af7f610a303d1a9b74a77ec5a62a2e3759ea0). Both report actual_device=cuda:0 and fallback_count=0; PT/JT array hashes are f50b79258c22ac0931759ed5eb41ddc47779de13bf89a8d3a8ec83101aaf2f84 and 14f8667360b75b9a1248a6d06d136cc2be20d922281a2310ef7529fb37893208. pow2 is exact, while native mean, sum(last_dim)/N, CUDA-scalar division, and epsilon variants all retain 1046/3072 differing values with maximum absolute difference 1.4901161193847656e-08. This does not close the full BF16 logits gate; LoRA full/resume remains blocked.

### Vision and BERT

The r18 wrapper failure due to legal missing torch.__file__ remains preserved. r19 contains the getattr measurement fix, but its controller stopped before any launch with STOPPED_PREFLIGHT_NO_NEW_LAUNCH: the prepared plan gate identity contains two extra frozen paths (MANIFEST.in and tests/structure/core/test_error_categories.py) compared with the focused-gate summary, so r19 is PREPARED_NOT_RUN, not a model result. r21/r22b remain separate evidence and Vision remains FAIL: same-runtime/cross-runtime failures are retained.

The new r25 cohort re-froze the five original Vision cases with matching focused-gate identity and a separate Jittor cache. Its first cnn-fp32-821-smoke CUDA run completed with execution receipt PASS and return code 0; the artifact audit passed all checks, including actual cuda:0 parameters, zero fallback, exact save/reload logits, two completed updates, and a nonzero update to all 62 trainable parameters. This is only a bounded smoke result. The r25 stage all controller subsequently completed all five frozen cases with retained comparison failures, as summarized below. The Jittor cache was isolated while the wrapper HOME/TMPDIR/XDG paths pointed at retained r19 warm-unit directories; this environment detail remains part of the audit record.

The r25 stage-all controller then completed all five cases with status COMPLETED_WITH_FAILURES_RETAINED. Every smoke/full/resume receipt and artifact audit completed with actual cuda:0 and fallback_count=0. The retained comparison outcomes are: cnn-fp32-821 same-runtime exact PASS but cross-runtime strict FAIL with 160 failures; vit-fp32-821 same-runtime exact FAIL with 100 failures; cnn-bf16-821 same-runtime exact FAIL with 245 failures; vit-bf16-821 same-runtime exact FAIL with 143 failures; cnn-fp32-823 same-runtime exact PASS but cross-runtime strict FAIL with 177 failures. The full and resume runs therefore establish executable CUDA/update artifacts, not Vision release acceptance. All receipts, comparisons, checkpoints, and logs remain under the r25 directory; no retry or overwrite was performed. Vision remains FAIL.

BERT BF16 remains FAIL because the frozen two-seed task-loss gap is 15.425953905%, above the unchanged 15% threshold. No seed, tolerance, or checkpoint audit was changed.

## Day4–5 performance and memory

The independent day45-bert-release-r3 cache and output directory contain real AdamW updates (parameter_changed=true), matched FP32 weights/inputs/device/optimizer, synchronized timing, 10 samples after warmup, three PT/JT alternating rounds for representative batch=8, seq=256, and two rounds for legal near-capacity batch=64, seq=512. CUDA device is 0, Jittor fallback_count=0; allocator allocated/reserved peaks and NVML process peaks are recorded per run. Representative JT/PT median ratios are 1.409, 1.510, and 1.341 (p95 ratios 1.398, 1.478, and 1.427). Near-capacity median ratios are 0.746 and 1.132, with p95 ratios 1.147 and 1.136. The invalid batch=64, seq=1024 attempt is retained as a fail because BERT has 512 maximum positions.

All these runs saw strong shared GPU load (summary status DIAGNOSTIC_SHARED_GPU), so they are diagnostic only and do not satisfy the release speed/memory gate. Existing pilot/protocol outputs remain diagnostic and were not overwritten. A clean isolated performance window and any common-path regression repair still require a later run. The r3 runs do meet the mechanical protocol shape (real AdamW parameter updates, independent performance cache, synchronized PT/JT timing, allocator and NVML peak fields, actual CUDA device, and fallback_count=0), but no common-path code change was made from shared-load measurements.

## Reproduction and review

Raw artifacts are under /data6/cgq/jittor-lab/_state/release-day23/20261020-r3/: t5-resume-diagnostic-r38/ (with r36 retained), qwen-rmsnorm-reduction-r24/, day45-bert-release-r3/, and the retained Vision cohorts. Recheck all process identities and receipts before continuing. The release gate is not closed until T5, Qwen, Vision, BERT, and an uncontended Day4–5 measurement satisfy their original criteria.


## 2026-10-25 r4 bounded wait

A new independent Day4-5 performance directory is prepared at /data6/cgq/jittor-lab/_state/release-day23/20261020-r4/day45-bert-release-r4/. It binds repair worktree HEAD 9e0e187bb013e71d85111acfd668e46aeff9f9d2, dirty SHA 4f28ab7c26e2ede3239cbad0ae0336c92d492dcfcf87d792c342850314625cee, weights SHA 86fa7113367fd0d231a76571ca1b73a1b24166af33558fee960636ce56337ed7, and GPU UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5. The bounded waiter requires utilization <=10 percent and free memory >=60000 MiB for five consecutive five-second samples, with a 1800 second timeout. It is currently waiting with free memory 70498 MiB but utilization 100 percent and zero qualifying streak; no PT/JT child has started.


At latest live poll 2026-10-24 21:43 UTC, waiter PID 1746490 remained alive with zero qualifying samples; target GPU utilization was 100 percent and free memory 41199 MiB. No launch or measurement receipt exists, so r4 remains unstarted.


## 2026-10-25 r4 wait terminal state

r4 bounded waiter terminated at 2026-10-24T21:56:34.266419+00:00 UTC. The receipt `/data6/cgq/jittor-lab/_state/release-day23/20261020-r4/day45-bert-release-r4/wait_receipt.json` records `WAIT_TIMEOUT_NO_RELEASE_WINDOW`. No 1800-second window met utilization <=10 percent, free >=60000 MiB, and five consecutive five-second samples; utilization stayed at 100 percent and free memory fell from 70498 MiB to 41199 MiB. Final source HEAD and dirty SHA remained `9e0e187bb013e71d85111acfd668e46aeff9f9d2` and `4f28ab7c26e2ede3239cbad0ae0336c92d492dcfcf87d792c342850314625cee`. No launch receipt, completion receipt, run JSON, or measurement log was generated, confirming formal r4 PT/JT measurement did not start. Day4-5 remains incomplete; r3 shared-load diagnostics are not release performance or memory evidence.

## 2026-10-25 Qwen RMSNorm reduction r26 diagnostic

After r4 timed out, a bounded Qwen correctness probe reused frozen r22 input and manifest under a new r26 output/cache; no LoRA training was launched. PT/JT both ran on CUDA 0 with fallback_count=0 and unchanged candidate HEAD/status. On frozen shape `[2,96,16,128]`, pow2 remained exact. Native mean, direct sum/N, CUDA tensor division, reshape sum, epsilon, and reverse sum retained 1038–1046 mismatches (max abs `1.4901161193847656e-08`); four chunk sums reduced this to 662 mismatches but did not close the full logits gate. Qwen and the full no-hook BF16 logits check remain FAIL; LoRA full/resume remains prohibited. Raw output is `/data6/cgq/jittor-lab/_state/release-day23/20261020-r3/qwen-rmsnorm-reduction-r26/`.
