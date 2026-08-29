# Ascend 910B 环境、ACL 冷启动与 NPU 门禁验证

- Status: Accepted within the maintained NPU gate; explicit skips remain
- Last reviewed: 2026-08-29
- Source baseline: `f8e39607` plus the changes documented here
- Owner: Jittor core and ACL backend maintainers
- Review when: CANN/driver versions, ACL source transformation, NPU gate scope,
  or any listed skip changes

## 结论

Jittor 2.0 当前源码已在真实 Ascend 910B3 上完成冷缓存验证。最终门禁从空的
Jittor/ACL 编译缓存启动，检测 CANN 与 ACL 设备，执行 float32 matmul 探针，并覆盖
ACL 后端、扩展算子、索引、227 项 OpInfo、负整数 floor-divide 及 NaN 比较回归。

设备证据不依赖导入成功或 CPU fallback：ACL matmul 回归捕获到
`compile acl op`，同时断言日志中没有 `fallback cpu`。维护范围内最终共
`362 passed, 11 skipped`；skip 均对应本报告和 known-issues ledger 中的明确能力边界。

## 环境与隔离

- Device: Ascend 910B3
- Architecture: Linux aarch64
- Driver / `npu-smi`: 25.5.1
- CANN / `ccec`: 9.0.0
- Hardware Python: 3.9.25
- NumPy: 1.26.4
- SciPy: 1.13.1
- pytest / pytest-timeout: 7.4.4 / 2.3.1
- Nox tooling Python: 3.11
- Device selection: one allocated idle 910B3 through
  `ASCEND_RT_VISIBLE_DEVICES`; an existing workload on another device was not
  touched
- Isolation: `$JITTOR_LAB_ROOT/_state/nox/envs/npu/tmp/`,
  `cache_name=nox_npu`
- Raw logs: `$JITTOR_LAB_ROOT/_state/ascend-910b-guide/final/logs/`
- JIT policy: the first current-source core, ACL extension, and model operation
  compiles ran serially; the full maintained Nox gate ran afterward

The validated versions are a reproduced baseline, not universal minimum or
maximum requirements. Host-specific paths, device numbers, caches, and raw logs
remain outside the repository.

## Verify-then-fix findings

### Cold startup and source conversion

The original checkout did not cold-start reliably in the maintained CANN shell.
The CANN environment script could fail under `bash -u` when library variables
were unset, Torch preflight passed CUDA-only strict math flags to `ccec`, and the
ACL CUDA-source transformer could rewrite CUDA words inside comments and string
literals while mishandling repeated `cudaGetDeviceCount`, `cudaError_t`, and
`cudaSuccess` tokens.

The NPU session now initializes optional environment variables before sourcing
CANN. Torch preflight detects an Ascend environment before Jittor import and
removes the incompatible CUDA flags. The ACL transformer lexically excludes
comments and literals, preserves include rewriting, maps every device-count call
through a typed wrapper, and uses ACL error types and success constants. ACL
registration and compilation failures are no longer hidden by broad exception
fallbacks.

A clean Nox run also exposed an interpreter mismatch: the tooling Python 3.11
config helper was inherited by the external Ascend Python 3.9 process, which
built a `cpython-311` core extension that Python 3.9 could not import. The NPU
session now resolves and validates the config helper from `JITTOR_CI_PYTHON`;
the accepted cold run produced `cpython-39` extensions.

### ACL execution and operator semantics

Real-device OpInfo expansion reproduced missing or incorrect paths in gather,
permute, 1-D matmul, negative/fancy getitem, index_select, embedding max-norm,
searchsorted, affine-grid construction, NaN/Inf predicates, pooling ceil-mode,
and Torch `where`. The ACL dtype bridge also depended on stale NanoString numeric
values and lacked complex64.

The fixes route the public spellings to their ACL implementations, normalize
negative indices, implement ACL-safe reshape/broadcast forms, use stable
NanoString constants, and avoid unsupported intermediate operations. Ascend's
measured asin/acos float32 error is covered by a focused `1e-4` reference
tolerance without relaxing other operators.

### Lazy ordering and max/min values

The full asynchronous OpInfo sequence initially failed while materializing
`kthvalue`. Its ordering sample builder created float64 device Vars and cast them
afterward, invoking unsupported ACL float64 fallback. Samples now enter Jittor in
their target dtype and `kthvalue` yields them incrementally.

The ACL `kthvalue` path also uses Jittor's captured native `argsort`, whose
contract returns both indices and values. This avoids feeding Torch-compatible
indices-only `argsort` output into a second gather during lazy materialization.

A subsequent long-sequence failure returned argmax indices where Torch-compatible
`max(dim).values` was expected. On ACL, the compatibility wrapper had reconstructed
values through an argmax-plus-gather dependency. It now uses native ACL ReduceMax
or ReduceMin for values and retains argmax/argmin only for indices. The complete
default asynchronous suite then passed; `JT_SYNC=1` is not required by the final
gate.

Float16/float32 `arg_reduce` itself now dispatches to CANN MaxDim/MinDim and
returns both Jittor outputs in their native order. Focused max/min cases cover
multiple axes, keepdims, scalar output descriptors, and first-index tie behavior;
the test fails if the operation compiles for CPU or falls back. Its generic
backward still reaches an unsupported ACL index operation and remains explicitly
outside the forward support claim.

### Boolean-mask assignment

The cold sequence reproduced a run-order-dependent no-op for `x[mask] = scalar`.
The ACL setitem path counted selected elements with a boolean reduction, and the
unsupported reduction could incorrectly report zero and return the input
unchanged. Scalar masked-scatter now supplies a source up to the input element
count without reducing the mask. Tensor sources retain exact length validation
through a supported int32 reduction. The indexing gate covers non-empty scalar,
empty scalar, and tensor-source masks; all 27 internal indexing checks passed.

### Transformers SDPA and public `all`

Qwen3's Torch SDPA path now dispatches the verified FP32 no-grad subset to CANN
`aclnnFlashAttentionScoreV2` before compatibility GQA expansion. Focused real-NPU
references cover square causal prefill, rectangular decode, GQA, and arbitrary
float additive masks; all execute without CPU compilation or fallback. The
runner also releases its per-call CANN integer-array descriptors, which matters
for multi-layer and multi-token generation.

Transformers checks whether an attention mask can be ignored through
`padding_mask.all()`. The public ACL path now sends bool truth reductions to
CANN 9 `aclnnAll`/`aclnnAny`; numeric inputs first receive the required nonzero
comparison. This preserves the public bool result while avoiding unsupported
`reduce.logical_and` and the earlier multi-op composition. Full, dimension, and
negative-dimension reductions pass on the real NPU. The Jittor core bool
`all_`/`any_` OpInfo matrix remains a separate explicit skip and is not claimed
by this ACL wrapper path.

## Maintained gate

```bash
export JITTOR_LAB_ROOT=/path/to/jittor-lab
export CANN_SET_ENV=/path/to/Ascend/cann-9.0.0/set_env.sh
export JITTOR_CI_PYTHON=/path/to/ascend-python/bin/python
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
python -m nox -s npu
```

Full Nox results after the serial prewarm:

| Stage | Result |
| --- | ---: |
| ACL device and float32 matmul probe | passed |
| `tests/backends/npu/test_acl.py` | 26 passed |
| `tests/backends/npu/test_acl_torch_compat.py` | 2 passed |
| `tests/backends/npu/test_aclop.py` | 110 passed, 2 skipped |
| `tests/backends/npu/test_acl_indexing.py` | 2 passed |
| `tests/ops/test_ops.py` | 218 passed, 9 skipped |
| NPU floor-divide fixed vectors and broadcast | 2 passed |
| NPU float32 NaN/Inf predicates | 1 passed |
| NPU float32 fused/unfused NaN comparisons | 1 passed |
| Total | 362 passed, 11 skipped |

The complete maintained session finished successfully in 43 minutes, including
the Nox-managed core rebuild and all real-device tests.

The floor-divide regression covers uint8, int8, int16, int32, and int64 fixed
vectors plus int64 broadcasting against NumPy. The NaN comparison gate covers
six comparison forms with same and distinct Vars under fused and unfused
execution.

## Supporting validation

- Focused CPU semantic matrix: `49 passed, 16 skipped, 618 deselected`.
- Torch preflight and bootstrap regressions: `8 passed, 16 deselected`.
- Repository structure suite: `219 passed, 2 skipped`.
- Repository layout and documentation-governance gate: passed.
- Documentation link audit and built-API audit: passed; the fresh Ascend catalog
  contains 56 translated messages with no fuzzy or untranslated entries.
- Fresh English and Simplified Chinese HTML builds: passed. The exact strict
  `nox -s docs_zh` command still stops during gettext extraction on 11 existing
  warnings in older API docstrings and documentation links. None originates in
  `guides/ascend-910b`; the strict baseline warning debt remains outside this
  backend and guide change.

The intentional ACL additions increased `misc/tensor_ops.py` from 2,845 to
2,874 lines, so its structure budget was synchronized from 2,850 to 2,900.

## Transformers Qwen3-8B inference

A separate real-device probe validated the local Qwen3-8B checkpoint through
Transformers 4.56.2 and the Jittor Torch shim. The accepted path used CPU
checkpoint deserialization followed by an explicit migration to the visible NPU,
float32 weights, SDPA attention, KV cache, batch size 1, a 22-token prompt, and
one greedy output token. This is an inference correctness probe, not a throughput
benchmark.

| Evidence | Result |
| --- | ---: |
| Parameters | 8,190,735,360 |
| First parameter | `float32`, `is_cuda=true` |
| Jittor backend flags | `has_acl=1`, `use_acl=1`, `use_cuda=1` |
| Process device memory after load | 32,376 MB |
| Total HBM use on the selected card | 35,735 / 65,536 MB |
| Load and explicit migration | 107.10 s |
| First SDPA prefill including JIT | 8.49 s |
| Steady prefill / one-token generation | 0.1144 s / 0.1282 s |
| Generated token | ID 19, text `4` |
| ACL fused-attention hits / misses | 216 / 0 |
| ACL `fallback cpu` diagnostics | 0 |
| CPU-compiled operations during generation | 0 |

The preceding Qwen3-0.6B SDPA probe used the same fail-closed protocol for eight
tokens and generated `2 + 2 = 4.` in 0.5791 s median. It recorded 2,268 fused
attention hits, zero misses, zero fallback, and zero CPU compile. The 8B run then
exercised all 36 decoder layers and the language-model head with the full
checkpoint resident on one 910B3. Its model forward and final greedy selection
likewise completed through ACL, so float32 inference is fully device-resident
within this protocol.

A 2026-08-29 follow-up connected Qwen3's version-specific RoPE helper to the
generic Jittor ACLNN RoPE capability through an external module patch. The full
8B checkpoint again used 32,376 MB of process device memory and generated token
19 (`4`), with `fallback_count=0` and `cpu_compile_count=0`. This confirms the
RoPE optimization on the real NPU.

The same follow-up then connected Torch SDPA to CANN FlashAttentionScoreV2 and
removed the public `padding_mask.all()` fallback. Qwen3-0.6B complete logits
against native `torch_npu` SDPA have maximum absolute error `3.8035214e-05`,
identical argmax, and identical top-10/top-20 token sets. Qwen3-8B retained token
19 while exercising the fused attention path in all layers. FP16/BF16 fused
SDPA remains fail-closed; the accepted whole-model claim remains FP32.

Verify-then-fix exposed two Torch-compatibility defects before the accepted run.
Explicit construction of the zero-length CUDA tensor used by the Transformers KV
cache attempted an invalid zero-byte CANN copy, and `torch.get_default_device()`
reported CUDA whenever ACL was installed even when `use_cuda=0`. Zero-length
tensors now retain logical device residency without a storage copy, and the
default-device query follows the execution flag. Both focused regressions pass on
the real NPU.

A later bfloat16 follow-up resolved that rejected attempt. Torch-compatible
factories and Linear/Embedding now honor the default dtype used during
Transformers construction, ACL Add/Sub accepts bfloat16, scalar host reads wait
for `aclstream`, and fused scalar H2D copies retain their pinned host source.
Qwen3-0.6B repeated six identical 8-token generations; Qwen3-8B repeated five
identical generations, each `[19, 13, 151645]` (`4.`). Parameters remained
bfloat16 and accelerator-resident, with zero CPU compilation and fallback.

## Explicit limitations

The nine OpInfo skips are:

- native Jittor core boolean `all_` and `any_` reductions; the public CANN 9
  `all`/`any` wrapper path used by Transformers is separately verified;
- composed float32 `atan2`;
- complex `irfft`;
- float32 `prod`;
- integer `sum`, `prod`, `max`, and `min` matrices that include unsupported
  sub-32-bit atomics.

The two ACL extension skips are native FlashAttention forward/backward when
`jt.nn.FlashAttention` is unavailable. Focused reproductions showed that the
skipped operations cannot remain xfails inside one pytest process: `atan2` can
raise a vector-core exception, `irfft` can stall beyond 600 seconds, and product
reductions can abort the process. They therefore fail closed as precise skips
linked to `KI-BACKEND-001` through `KI-BACKEND-004`.

General ACL float64 support remains unavailable. A CPU fallback for float64 is
not counted as NPU evidence. Fused SDPA is restricted to the verified FP32
no-grad subset. `arg_reduce` backward, bfloat16 training/fused SDPA, full
training, distributed NPU behavior, and other optional downstream projects are
separate gates and are not claimed by this report.

## User documentation

The bilingual [Ascend 910B setup and validation guide](../../docs/guides/ascend-910b.md)
records the portable CANN initialization, source installation, cache isolation,
real ACL probe, maintained nox command, troubleshooting steps, and current
limitations. It does not embed the validation host's personal paths or device
allocation.
