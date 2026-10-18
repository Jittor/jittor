# vLLM TP1 decode metadata follow-up (2026-10-11)

- Status: experimental, validated on one RTX 4090; source tree has pre-existing uncommitted changes.
- Baseline commit: `2b49749cff5bf5fb3a1a102e3465021cc25f4707` on `chk` (ahead of `origin/chk` by two commits when checked).
- Owner: Jittor vLLM adapter maintainers.
- Review when: vLLM `buffer_utils`/`block_table` changes, Jittor gains real pinned host memory, or CUDA Graph shape switching is changed.

## Scope and implementation

The vLLM 0.24.0 adapter now bypasses `pin_memory()` for contiguous NumPy metadata of at most 4096 bytes in `async_copy_to_gpu`. In this Jittor runtime, `pin_memory()` is an ordinary CPU copy rather than a page-locked allocation, so it cannot enable asynchronous DMA. Torch tensor inputs, larger or noncontiguous NumPy inputs, and the opt-out `JITTOR_VLLM_DIRECT_NUMPY_H2D=0` use the original method. The existing fused block-table/slot-mapping op now accepts unpadded batch=4 as well as batch=1, with the old path retained for other shapes, padding, multiple KV groups, or context parallelism. Its opt-out is `JITTOR_VLLM_FUSED_BLOCK_METADATA=0`.

The change is adapter-local: it does not alter generic Torch `pin_memory()` semantics or model arithmetic.
An additional real-CUDA ownership probe mutated the source NumPy array after
`torch.from_numpy` and after each CUDA copy was submitted; the host snapshot,
new CUDA tensor, and persistent `out` tensor all retained the pre-mutation values.

## Protocol and results

One physical RTX 4090 (GPU 5), Qwen3-0.6B, TP1, FP16, CUDA Graph capture sizes 1 and 4, 128 input tokens, 64 output tokens, two warmups and three measured repetitions per batch. Both interpreters used vLLM 0.24.0 and Transformers 5.5.3; the native oracle used PyTorch 2.11.0+cu130, and the runner asserted it was not the shim. Options, prompts, device name, and all 15 measured output-token sequences matched. Timings are offline `engine.step` token-delivery timings, not HTTP latency or isolated GPU kernel timings. No other process occupied GPU 5 during the formal runs; an earlier cold-cache attempt overlapped another user's CPU compilation and was abandoned without using its timings.

| Comparison | Batch | Off | On | Change |
| --- | ---: | ---: | ---: | ---: |
| Direct small-NumPy H2D, tok/s | 1 | 228.65 | 242.73 | +6.16% |
| Direct small-NumPy H2D, tok/s | 4 | 782.96 | 824.53 | +5.31% |
| Fused block metadata, tok/s | 4 | 803.65 | 841.82 | +4.75% |

The reverse H2D-off run returned 230.60/780.10 tok/s for batch=1/4. The fused batch=4 A/B had 12/12 exact sequences. Three 1↔4 switch cycles in one engine completed with 15/15 exact sequences against the independent oracle, but this bounded test does not establish production stability.

| Final same-day run | Batch=1 | Batch=4 |
| --- | ---: | ---: |
| Jittor, tok/s | 239.33 | 835.58 |
| PyTorch, tok/s | 405.48 | 1493.81 |
| Jittor/PyTorch | 59.0% | 55.9% |
| Jittor median ITL | 3.520 ms | 4.119 ms |
| PyTorch median ITL | 2.306 ms | 2.517 ms |

The component A/B gains and final-run rates are not added arithmetically: the former isolate one flag at a time, while the latter is a fresh combined run.

## Validation and boundaries

The host-only snapshot/metadata contract suite passed 25/25 tests, and the real-CUDA greedy sampling suite passed 4/4. `git diff --check` and Python syntax compilation passed. The new H2D path retains the destination buffer for `out=` calls, which CUDA Graph replay requires. Remaining throughput is still only 55.9–59.0% of native PyTorch on this protocol; further claims need a new host/GPU timeline with the latest code rather than attributing the gap to one kernel by inference.

Raw unversioned artifacts: `$JITTOR_LAB_ROOT/_state/vllm-tp/20261011/` (`h2d-{off,on,off2}-warm.json`, `fused4-{off,on,switch}.json`, `final-{jittor,oracle}.json` and matching logs). A warm TP1 JIT cache from the previous run was reused; initialization and capture were excluded from measurements. No commit or push was made because the adapter source file already contained unrelated uncommitted work.
