# Qwen2.5 NPU prefill hidden-state control — 2026-10-16

Native vLLM-Ascend and Jittor core vLLM 0.24 each ran Qwen2.5-3B BF16 TP1 on the same NPU6, 128 input + 16 greedy output, at batch 4 and batch 1. The batch-4 prompt-token SHA256 was `642b80747d8ec7d15b291ef59e952d3c5d1aa5b2a3bbe49a3677c92bf3afd13b` for both backends. Native and Jittor returned the same 16 token IDs; each worker exited gracefully. An opt-in `model.forward` call-0 snapshot recorded the last request's final prefill-position hidden vector (2048 BF16 elements); it did not snapshot every request.

| Comparison | Mean absolute difference | RMSE | Exactly equal / 2048 |
| --- | ---: | ---: | ---: |
| Batch 4, Jittor versus native | 0.034562 | 0.049065 | 144 |
| Batch 1, Jittor versus native | 0.030971 | 0.049581 | 192 |
| Native batch 1 versus batch 4 | 0.028732 | 0.045292 | 222 |
| Jittor batch 1 versus batch 4 | 0.022057 | 0.031033 | 280 |

Prefill numerical variation is present within native across batch sizes and in single-request cross-backend comparison. These call-0 observations do not isolate a batch-4-only implementation error or establish the cause of the later output-index-47 divergence. They do show that hidden vectors are not bitwise equal before decode, while all first 16 greedy outputs remain equal. This is a correctness/numerical diagnostic, not a performance comparison.

Logs: `_state/qwen25-b{1,4}-128x16-prefill-hidden-{native6,jittor6}-20261016.log`. Compare with `docs/results/2026-10-15-npu-vllm-validation.md` for the existing batch-4 index-47 logprob and call-11/call-47 hidden-state probes.
