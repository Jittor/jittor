# Running vLLM on Ascend 910B3 through Jittor: bootstrap, correctness and single-request performance

- Status: the native vLLM-Ascend baseline is accepted; with Jittor + the Torch
  compatibility layer + an external NPU platform plugin, the public `vllm.LLM.generate` for
  Qwen3-0.6B is accepted for both correctness and performance within the restricted
  protocol; broader coverage and vendoring the plugin into the repository remain open
- Date: 2026-08-31 (through 2026-09-02)
- Baseline commit: `ed6ede29`
- Validation scope: one Ascend 910B3, CANN 9.0.0 + ATB; on the native side Python 3.10.20,
  PyTorch 2.10.0, torch-npu 2.10.0, vLLM 0.20.2 + vllm-ascend 0.20.2rc1; on the Jittor side
  vLLM 0.20.2; Qwen3-0.6B BF16, unquantised, TP=1, single request, `max_model_len=32`,
  block size 128, eager, async scheduling and prefix cache disabled. Not covered: multiple
  requests, long context, chunked prefill, quantisation, TP>1, repeated engine lifecycles
- Maintainers: Torch compatibility layer and vLLM adapter maintainers
- Review triggers: version changes in vLLM, Transformers, CANN, ATB or the external plugin;
  changes to the ACL routing of serving primitives (paged attention, RMSNorm, RoPE, SwiGLU,
  split)

## Conclusion

The 4-token greedy output for the prompt `The capital of France is` is identical on both
paths:

```text
token ids: [12095, 13, 576, 6722]
text:       " Paris. The capital"
```

The Jittor path completed model construction (226 parameter tensors), strict safetensors
loading, an independent attention cross-check, the full prefill, a manual decode reusing the
KV cache, and the worker/scheduler/KV allocation/sampling loop of the public
`vllm.LLM.generate`. Every request has `fallback_count=0` and `cpu_compile_count=0`, and the
process never loads `torch_npu` or `vllm_ascend`.

Performance (same prompt and engine configuration, 10 requests per process, excluding the
first graph-warm-up request):

| Path | Warm-request median | Samples |
| --- | ---: | --- |
| Native vLLM-Ascend | 0.389976 s | 18 combined from two rounds, 0.3708–0.4127 s |
| Jittor, initial generic path (block16) | 0.926 s | — |
| Jittor, baseline commit | 0.363298 s | 27 combined from three rounds, 0.3468–0.3760 s |

That is, within this restricted protocol Jittor is about 6.8% faster. To avoid a
double-free of the custom transformer op at process exit, the native side uses
`VLLM_BATCH_INVARIANT=1` and does not enable that custom-op library.

## The changes that made it work and made it fast

Compatibility layer (main repository):

- `torch.library.infer_schema` (the four schema groups match PyTorch 2.10.0 character for
  character), `Library.define/impl/_register_fake`, and the `.default` overload of
  `torch.ops` are no longer empty stubs; import boundaries such as `torch.Tag`,
  `torch.types.*` and the FakeTensor eager context are filled in;
- consistent logical `set_stream/current_stream/default_stream` state under a single
  physical stream (no claim of multi-stream concurrency); shim deployment also publishes
  the `flash-attn` dist-info;
- `torch.add(..., out=view)` and `torch.index_select(..., out=view)` return the original
  `out` and write back into the parent tensor (vLLM's input ID staging depends on this).

Jittor serving primitives (`python/jittor/nn/paged_attention.py`, `serving_ops.py` and
`backends/acl/kernels/`), all of which take over only on ACL, under no-grad, and with
supported dtypes/shapes; everything else takes the original path:

- paged attention / `reshape_and_cache` use Gather/Scatter for block-table reads and KV
  updates and no longer trigger the reindex CPU fallback; single-token decode with block
  size 128 hands the strided descriptor of the combined cache plus the block table directly
  to `aclnnIncreFlashAttentionV4`, and cache updates use same-stream small D2D copies;
  prefill of a new request does causal CANN SDPA directly on the current Q/K/V;
- RMSNorm → `aclnnRmsNorm`; NeoX full rotary → `aclnnApplyRotaryPosEmb`; `silu_and_mul` →
  `aclnnSwiGlu`; static split → `aclnnSplitWithSize`; RoPE position-table lookup →
  Embedding/Gather;
- following the batch-invariant rounding order, each layer's residual Add+RMSNorm, the
  exact BF16 RMSNorm+weight multiply for Q/K, and the two RoPE calls are each fused into one
  CodeOp. The total device operator time in the decode profile drops from about 118 ms to
  about 20 ms.

External plugin `vllm-jittor-npu` (not in the repository): it activates only after detecting
`torch.__jittor_version__`, refuses an already-loaded `torch_npu`, registers as an OOT
platform via `vllm.platform_plugins`, disables TorchInductor and CUDA graph; CUSTOM
attention uses the public paged attention described above; the worker reuses the V1
`GPUModelRunner` and explicitly synchronises the NumPy and tensor mirrors of
`CpuGpuBuffer`/`InputBatch` (Jittor's `.numpy()` does not share memory), failing closed on
conflict; only unquantised TP=1 is supported, everything else fails closed.

Withdrawn experiments (numerically correct but slower or unsafe): letting CANN consume the
block16 strided cache directly (about 6.95 s/token), switching host block IDs to SliceV2
(7.996 s/request), a static RoPE cache (the median got slower), and function-level RAII for
the CANN executor's TensorList (use-after-free in a later asynchronous Cast, whose lifetime
must outlive stream completion).

## Reproduction

```bash
source "$CANN_SET_ENV"; export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
python -m jittor.compat.shim.deploy --target "$RUN_ROOT/shim-site"
export PYTHONPATH="$RUN_ROOT/shim-site:$JITTOR_SOURCE/python"
pip install -e <vllm-jittor-npu checkout>      # 外置插件，提供 platform entry point
JITTOR_TORCH_SHIM=1 python <vllm-jittor-npu>/probes/qwen3_engine.py     # Jittor 路径
python <vllm-jittor-npu>/probes/qwen3_native_engine.py   # 在原生 vLLM-Ascend 环境里
```

The probes ship with the external plugin and are not in this repository. Both sides use the
same prompt, engine parameters and request count, and discard the first request; the first
request on a cold cache takes several minutes because it serially generates ACL JIT graphs
(332 s in the baseline re-check) and is excluded from performance. Maintenance regressions
on the main-repository side: `tests/backends/acl/test_acl.py` and
`test_acl_torch_compat.py` (NPU), plus the serving-primitive and vLLM layer patch cases on
CPU.

## Still open

- The external NPU platform/worker plugin is not versioned. The main repository's
  `adapters/jittor_adapters/vllm` later took in the CUDA/generic adapter, but it does **not**
  include the Ascend platform and worker (checked on `e3c369acb`), so the engine conclusions
  of this report currently have no reproducible entry point inside the repository;
- Multi-request throughput, first-token and per-token latency, long context, prefix cache,
  chunked prefill, abnormal exit and repeated engine lifecycles are all untested;
  quantisation and TP>1 remain fail-closed;
- The whole-block host mirror synchronisation and metadata materialisation of
  `CpuGpuBuffer`/`InputBatch` can still be reduced.

The known-issues ledger has no separate entry; this report is the only evidence for the NPU
vLLM path.
