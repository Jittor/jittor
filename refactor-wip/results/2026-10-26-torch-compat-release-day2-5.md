# Torch Compatibility Release Acceptance: 2026-10-26 Day 2–5 Continuation

## Status

**Not complete.** Day2–3 remains failed or incomplete at its original gates. Day4–5 has shared-GPU diagnostic data and a new bounded waiter, but no release-valid stable-window result yet. Thresholds, seeds, comparison fields, failure evidence, checkpoints, caches, and worktree changes remain intact.

- Date: 2026-10-26
- Main baseline: integrate/cgq-transformers-native-multicard-target at 95d0600ba7d4888082b3a0f71bac9f2df8acd5dc
- Candidate: release/torch-compat-day23-fix-20261021 at 9e0e187bb013e71d85111acfd668e46aeff9f9d2
- Remote sync: origin/cgq_transformers fetched at e5cffe1ae60c418f53c5bb1cdca9182577fd2b81; origin/2.0-refactor not integrated
- Owner: cgq / Jittor Torch compatibility maintainers
- Review when: candidate source, dependencies, fixtures, device, or acceptance protocol changes

## Workspace and Live State

The main worktree retains 95 dirty or untracked status entries. The original Day2–3 worktree retains 51 and the repair worktree retains 83. All three worktrees were inspected before continuing; no stash, reset, clean, source overlay, or remote integration was performed. The 77 dirty source files bound to Qwen r44 were rehashed in the repair worktree and all 77 matched the run identity.

No Day2–5 training or benchmark process is active. The r54 Jittor capture and comparison finished; its r52 cache lock has no holder. The sole task process is the Day4–5 r5 waiter, PID 2616242, started 2026-10-26 02:09:02 +0800. At 04:35 +0800 it remained WAITING; the state file's latest sample (04:35:08) was 49,409 MiB free at 100% utilization on UUID GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5, with zero qualifying samples. No launch, measurement, or completion receipt existed. The waiter remains active; recheck its live identity and state before a later continuation.

Filesystem space was about 100 GiB available on / and 6.0 TiB on /data6. No files were removed. Checkpoints, failure evidence, logs, caches, and worktree edits remain preserved.

## Day2–3

### T5 BF16

The existing r36 and r38 step-51 captures remain the newest localization evidence. r36 captured the requested dropout and encoder block 0 FFN boundaries: earlier dropout/attention events were exact, then encoder block 0 DenseReluDense.wi input differed at 1,449 elements with maximum absolute difference 1.430511474609375e-06. Its formal comparator remains FAIL because the pre-forward signature does not match retained r6. r38's dynamic trace found its first archived difference later, at encoder block 1 DenseReluDense.wi input (1,377 elements, max abs 3.5762786865234375e-07). The historical strict recovery failures remain 771/2088. No full100 was repeated.

### Qwen BF16

Qwen r44 completed the original unhooked BF16 CUDA full-logits comparison from a fresh PT/JT run using the frozen input and source manifest. Both worker commands exited 0; the comparison receipt is FAIL_RETAINED and records full_training_launched=false. Both sides used actual cuda:0 with fallback_count=0. Inputs and loss met their unchanged comparisons, but logits failed: 7,015,339 of 29,171,712 values exceeded atol=0.005, rtol=0.03; maximum absolute difference was 2.25. On valid positions, 2,756,985 of 22,030,720 values exceeded budget and maximum absolute difference was 0.3125. The loss difference was 0.0038883686 and remained within budget. Only the logits key failed. The comparison SHA256 is d604c30238a3a6c27d2adf5a283ed38af05d0396599088dd3019d46cf60eb770; execution receipt SHA256 is eadd2e696d142817bc357b8249319f7c109f5de39ee7224e947fd3f97a34deee. No LoRA full/resume was launched.

r46 continued from r23/r26 on the frozen r22 tensor `[2,96,16,128]`, reducing the earliest mismatch to the CUDA mean accumulation order. In PyTorch 2.6's `Reduce.cuh`, this case uses 32 lanes, each accumulating four values at offsets `lane + {0,32,64,96}`; `block_x_reduce` then combines warp offsets 1, 2, 4, 8, and 16 before `MeanOps` divides by 128. A new explicit expression using that order matched native PyTorch exactly on all 3,072 outputs in both the independent PT process and Jittor shim process. Jittor's native `mean(-1)` retained 1,046 mismatches with max absolute difference `1.4901161193847656e-8`. Both processes used actual `cuda:0`, fallback_count=0, the same frozen input/manifest, and candidate HEAD/status identity `9e0e187bb013e71d85111acfd668e46aeff9f9d2` / `a061f1cd85aa91037f50720ce2a50a5211843f16feeed7809871ae724398c41e`. This is a reduction diagnostic only; it does not change the public implementation, prove full-model logits acceptance, or authorize LoRA. The r44 full-logits FAIL remains the release state.

r50 is the latest unhooked full-logits gate. With the unchanged `atol=0.005, rtol=0.03`, it remains FAIL: 6,690,070 of 29,171,712 logits are outside budget and maximum absolute difference is 1.953125. The original failure and comparison fields remain intact; no LoRA full/resume was launched.

r54 then compared the public RoPE module and a direct replay of the frozen Transformers source formula. Both captures reproduced their own frozen same-runtime input and logits exactly, used the same A800 UUID `GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068`, actual `cuda:0`, fallback_count=0, BF16, seed 821, and candidate HEAD/fingerprint `9e0e187bb013e71d85111acfd668e46aeff9f9d2` / `d8b57cb5da41cbbf1a59c1310cc1204d79a0b9394aa66b4f1774cdec0bd927dd`. The 113-capture comparison had no input mismatches and first differed at `rotary_emb/output/1` (sin): 162/12,288 values, max absolute difference `3.725290298461914e-9`. Direct replay found exact `position_ids`, one differing `inv_freq` value at index 37 (`0.0003398208355065435` PyTorch vs `0.00033982080640271306` Jittor), 81/6,144 differing `freqs`, and 162/12,288 differing `sin`; `cos` was exact. Layer 2's first BF16 main-path mismatch remained `self_attn.o_proj`: 59/196,608 values, max absolute difference `0.0009765625`.

A separate 64-value CUDA probe decomposed the default `1 / (base ** exponent)` initialization with `base=1,000,000`. Jittor's float32 `pow` and reciprocal matched host `math.pow` rounding for all 64 values; independent PyTorch differed from that host result at 4 power values and 3 reciprocals. This identifies a backend floating-point difference in the RoPE frequency initialization, not a proven Jittor correctness defect or a source repair. Replacing the expression with `exp(-x * log(base))` was not a general match. r54 applies no release thresholds and is diagnostic only: r50 remains FAIL, no source changed, and LoRA remains blocked until the original no-hook gate passes.

### Vision and BERT

The r18 missing torch.__file__ wrapper failure remains historical entry-point evidence. After the measurement entry was corrected, the new r25 cohort completed all five smoke/full/resume cases and artifact audits on real CUDA with fallback_count=0. cnn-fp32-821 passed same-runtime exact comparison but failed cross-runtime strict with 160 items; vit-fp32-821, cnn-bf16-821, and vit-bf16-821 failed same-runtime exact with 100, 245, and 143 items; cnn-fp32-823 passed same-runtime exact but failed cross-runtime strict with 177 items. Vision remains FAIL; all receipts, comparisons, checkpoints, and logs are retained.

BERT BF16 remains FAIL: the frozen two-seed task-loss gap is 15.425953905%, above the unchanged 15% limit. The checkpoint audit, seeds, and comparison fields were not changed.

## Day4–5

The r3 protocol measured real AdamW updates with matched FP32 weights, inputs, device, optimizer, and synchronization. It recorded 10 hot samples after warmup, model-ready/first-step/steady timing, allocator allocated/reserved peaks, NVML process peaks, actual cuda:0, and fallback_count=0. Representative batch 8, sequence 256 had three alternating rounds with JT/PT median ratios 1.409, 1.510, and 1.341; p95 ratios were 1.398, 1.478, and 1.427. Near-capacity batch 64, sequence 512 had only two rounds, with median ratios 0.746 and 1.132 and p95 ratios 1.147 and 1.136. Representative allocator allocated peaks were 3,006,131,712 bytes for Jittor and 3,041,185,280 for PyTorch; reserved peaks were 3,322,937,344 and 3,282,042,880. Near-capacity allocated peaks were 21,715,963,392 and 21,845,145,088; reserved peaks were 22,012,755,968 and 22,368,223,232. These measurements are retained in day45-bert-release-r3.

r3 summary status is DIAGNOSTIC_SHARED_GPU. The repeated representative slowdown is a common-path regression signal, but shared load and the incomplete near-capacity round count prevent a release conclusion or a code-level attribution. No source change was made from these measurements. r4's bounded 1,800-second wait ended WAIT_TIMEOUT_NO_RELEASE_WINDOW without a measurement launch.

A new queue was prepared at /data6/cgq/jittor-lab/_state/release-day23/20261026-r45/day45-bert-release-r5. It has a separate performance cache and output root, binds repair HEAD 9e0e187bb013e71d85111acfd668e46aeff9f9d2, dirty source SHA 4f28ab7c26e2ede3239cbad0ae0336c92d492dcfcf87d792c342850314625cee, and weights SHA 86fa7113367fd0d231a76571ca1b73a1b24166af33558fee960636ce56337ed7. The waiter requires utilization <=10%, free memory >=60000 MiB, and five consecutive five-second samples, with a six-hour timeout. If admitted, it runs 10 samples after four warmups for both representative and legal near-capacity grids, with three serial PT/JT alternating rounds and first JIT serialized. At 02:34 +0800 the waiter remained WAITING after 120 observations; the latest target sample was 50,422 MiB free and 99% utilization, with no launch or completion receipt. The queue does not claim acceptance until measurements, summary, and memory/device audits complete; shared-load output remains diagnostic.

At the 04:35 +0800 live check, the same waiter process and start time were confirmed. Its latest recorded sample had 49,409 MiB free, 100% utilization, and zero qualifying samples. No launch receipt or measurement child existed, so no new Day4–5 performance data were produced. The waiter remains active to preserve its bounded queue; its process identity and receipts must be rechecked before continuation.

The common-path performance issue is not marked fixed. After a stable-window result confirms a representative slowdown above 20%, inspect the affected training path, make the narrow repair in the repair worktree, then rerun its impacted correctness gates and matched performance cell with a new frozen source identity.

## Artifacts

- Qwen r44: /data6/cgq/jittor-lab/_state/release-day23/20261026-r44/qwen-full-nohook-r1/
- Qwen r46 reduction tree: /data6/cgq/jittor-lab/_state/release-day23/20261026-r46/qwen-reduction-tree-r1/
- Qwen r50 full-logits gate and r54 RoPE diagnostic: /data6/cgq/jittor-lab/_state/release-day23/20261026-r50/qwen-full-nohook-r2/ and /data6/cgq/jittor-lab/_state/release-day23/20261026-r54/qwen-rope-replay-r1/; r54 comparison SHA256 `022b806e18f8bb0bbc9c70758b72ca228fd3eddfbb128a6e779281f328f08ff6`
- Day4–5 r3 diagnostics: /data6/cgq/jittor-lab/_state/release-day23/20261020-r3/day45-bert-release-r3/
- Day4–5 r4 timed-out wait: /data6/cgq/jittor-lab/_state/release-day23/20261020-r4/day45-bert-release-r4/
- Day4–5 r5 wait and future measurement: /data6/cgq/jittor-lab/_state/release-day23/20261026-r45/day45-bert-release-r5/
- T5 and Vision evidence: see the 2026-10-25 report and original Day2–3 report
