# Ascend 910B3: cold-cache startup, ACL execution and the NPU gate

- Status: accepted within the maintained NPU gate scope (`397 passed, 9 skipped`); the
  capability boundaries behind the 9 explicit skips remain open, see "Still-open
  boundaries" below
- Date: 2026-08-28 (reviewed 2026-09-01)
- Baseline commit: `532c250b`
- Validation scope: one Ascend 910B3 (Linux aarch64), driver/`npu-smi` 25.5.1, CANN 9.0.0,
  Python 3.9.25, NumPy 1.26.4, pytest 7.4.4; cold start from an empty Jittor/ACL compile
  cache, serial warm-up first, then the full `nox -s npu`. Not covered: ACL float64, the
  full training pipeline, distributed NPU, ROCm
- Maintainers: Ascend backend maintainers
- Review triggers: a change in CANN or driver version, a change in the `nox -s npu` test
  set, or any skip below being removed or added

## Conclusion

The current source has been validated on a real 910B3 from a cold cache: probing CANN and
the ACL device, running the float32 matmul probe, and covering the ACL backend, extension
operators, indexing, 227 OpInfo entries, negative-integer floor-divide and NaN-comparison
regressions. The device evidence does not rely on "import succeeded" or CPU fallback: at
baseline the ACL matmul regression also asserted that the operator compiled on ACL and did
not fall back to CPU; the autouse fixture in `tests/backends/acl` now uses the fallback
counter from `forbid_backend_fallbacks()` instead of reading logs. All 9 skips are
registered capability boundaries; none of them is "the environment was not ready".

## Reproduction

```bash
export CANN_SET_ENV=<CANN 安装目录>/set_env.sh
export JITTOR_CI_PYTHON=<昇腾环境的 python>
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
python -m nox -s npu
```

The session first runs `npu-smi info` and a matmul probe with exact expected values, then
runs `NPU_TESTS` grouped by process mode (`noxfile.py`), with `JITTOR_TEST_REQUIRE_ACL=1`
and `JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`, so a missing card makes the whole group skip
itself rather than read as green. The first compile (core, ACL extensions, model operators)
must finish serially before the full gate runs; the cache lives outside the repository and
uses a dedicated `cache_name`.

## Gate results (baseline commit)

| Stage | Result |
| --- | ---: |
| ACL device and float32 matmul probe | passed |
| `tests/backends/acl/test_acl.py` | 41 passed |
| `tests/backends/acl/test_acl_torch_compat.py` | 16 passed |
| `tests/backends/acl/test_aclop.py` | 112 passed, 2 skipped |
| `tests/backends/acl/test_acl_indexing.py` | 4 passed |
| `tests/ops/test_ops.py` (OpInfo) | 220 passed, 7 skipped |
| NPU floor-divide fixed-value vectors and broadcasting | 2 passed |
| NPU float32 NaN/Inf predicates | 1 passed |
| NPU float32 fused/non-fused NaN comparison | 1 passed |
| Total | 397 passed, 9 skipped |

(At baseline these files lived under `tests/backends/npu/`; they have since moved to
`tests/backends/acl/`.) The floor-divide coverage spans uint8/int8/int16/int32/int64
fixed-value vectors and int64 broadcasting, compared against NumPy; the NaN-comparison
coverage spans six comparison forms across same/different Var and fused/non-fused.

## Defects reproduced first, then fixed

- **Cold start**: the CANN environment script failed under `bash -u` because variables were
  unset; the Torch pre-check handed CUDA-specific strict-math flags to `ccec`; Nox's 3.11
  config helper was inherited by the 3.9 process and produced a `cpython-311` core extension
  that 3.9 could not import. All three are fixed: optional variables are initialised before
  sourcing CANN, the Ascend environment is recognised before import and incompatible flags
  are dropped, and the config helper resolves against `JITTOR_CI_PYTHON`.
- **ACL operator semantics**: real-device OpInfo exposed missing or wrong paths in gather,
  permute, 1-D matmul, negative/fancy getitem, index_select, embedding max-norm,
  searchsorted, affine-grid, the NaN/Inf predicates, ceil-mode pooling and Torch `where`, as
  well as the dtype bridge depending on stale NanoString values and lacking complex64. The
  fix routes public spellings to the corresponding ACL implementations, normalises negative
  indices, and uses stable NanoString constants. The float32 error on asin/acos is covered by
  a single targeted `1e-4` tolerance; no other operator tolerance was relaxed.
- **kthvalue and max/min under lazy evaluation**: samples built a float64 Var first and then
  cast, triggering the unsupported ACL float64 fallback; `max(dim).values` used to be
  rebuilt from argmax+gather and would misalign on long sequences. Now samples enter in the
  target dtype, values go through CANN ReduceMax/ReduceMin, and argmax/argmin only supplies
  indices; `JT_SYNC=1` is no longer needed.
- **Boolean mask assignment**: `x[mask] = scalar` used to silently become a no-op because
  it counted with an unsupported boolean reduction; scalar masked-scatter no longer reduces
  the mask, and the tensor source keeps the exact length check.
- **Cross-device convolution comparison**: the old `test_conv` took a lazy random graph from
  an ACL scope straight into a CPU scope and reused it, so the two sides did not consume the
  same input. After building separate Vars from a fixed NumPy snapshot, the forward and both
  gradients pass on 910B3; no tolerance was relaxed and no fallback was added for
  convolution.

## Transformers inference probe (same machine)

Transformers 4.56.2 + the Torch compatibility layer, checkpoint deserialised on CPU and then
explicitly migrated to the NPU, SDPA, KV cache, batch 1. This is an inference-correctness
probe, not a throughput benchmark.

| Evidence | Qwen3-8B float32 |
| --- | ---: |
| Parameter count | 8,190,735,360 |
| Process device memory (after loading) | 32,376 MB |
| Steady-state prefill / single-token generation | 0.1144 s / 0.1282 s |
| Generated token | ID 19 (`4`) |
| ACL fused attention hits / misses | 216 / 0 |
| CPU fallbacks / CPU-compiled operators during generation | 0 / 0 |

Follow-up re-check on the same machine: Qwen3-0.6B full logits have a maximum absolute
error of `3.80e-05` against native `torch_npu` SDPA, with identical argmax and identical
top-20 sets; in bfloat16, 0.6B and 8B produce token-for-token identical repeated generation,
parameters stay bfloat16 and resident on the device; 0.6B BF16 SDPA (FlashAttentionScoreV2
for prefill, IncreFlashAttentionV4 for single-token decode) with 3 warm-ups and 10 samples
takes 0.53622 s versus 0.52247 s for native `torch_npu`. FP16 fused SDPA still fails closed.

## Still-open boundaries

Checked on the current tree (`e3c369acb`), the following skips are still present and the
corresponding entries are still open:

| Skip | Reason | Tracking |
| --- | --- | --- |
| Narrow-integer `sum`/`max`/`min`; native bool `all_`/`any_` | Missing ACL atomic overloads / logical reductions not validated on NPU | `KI-BACKEND-001` |
| float32 composite `atan2` | Can trigger a vector-core exception that terminates the process | `KI-BACKEND-002` |
| complex `irfft` | Does not return within 600 s | `KI-BACKEND-003` |
| The two native FlashAttention forward/backward cases | Skipped when `jt.nn.FlashAttention` is unavailable | — |

These skips cannot become in-process xfails: `atan2` terminates the process and `irfft`
hangs. At baseline there was also a float32 `prod` skip; `prod` now runs through CANN
`aclnnProd`/`aclnnProdDim`, and the current tree no longer has that skip. This report is
also the NPU-side evidence for `KI-OPS-002` (integer floor-divide, ROCm still to be
validated) and `KI-SEMANTICS-003` (float comparison, remaining NPU dtypes and ROCm still to
be validated).

Out of scope for this report: general ACL float64 (CPU fallback does not count as NPU
evidence), `arg_reduce` backward, BF16 training, distributed NPU and other downstream
projects, each of which has its own gate. For user-side installation, probes and
troubleshooting see the [Ascend 910B guide](../guides/ascend-910b.en.md).
