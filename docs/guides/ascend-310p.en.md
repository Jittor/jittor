# Ascend 310P: Configuration, Capability Boundaries, and Verification

310P is an inference card: the cube units support only FP16/INT8, with no BF16, and the
operator set that CANN compiles for `ascend310p` is also smaller than the one for training
cards. This page records only the parts that differ from the
[Ascend 910B guide](ascend-910b.en.md); the driver check, CANN environment, JIT cache
isolation, editable install, and other steps are completely identical, so follow that
guide first.

## Verified baseline

| Component | Verified version |
| --- | --- |
| Device | Ascend 310P3 (`npu-smi` single card, `/dev/davinci0`) |
| Architecture | Linux x86_64 |
| CANN toolkit and `ccec` | 9.1.0 |
| Python | 3.12.3 |
| NumPy | 1.26.4 |

Initial operator acceptance: `tests/backends/acl/` 214 passed / 11 skipped / 0 failed,
zero CPU fallback under `backend_fallback=error`, and every skip classified by capability.
Model-level acceptance: Transformers 4.56.2 + Torch shim, Qwen3-0.6B FP16 greedy
generation with zero fallback and identical results across three repeats (requires
`acl_allow_hf32=1`; RoPE's FP32 `inv_freq @ position_ids` takes an explicit reduced
precision path), see `docs/results/2026-09-22-qwen3-ascend310p-fp16.md`.
For the current 2.0 code, the synchronous performance comparison, input-length and
snapshot-weight constraints are in the
[310P3 performance report](../results/2026-10-12-qwen3-ascend310p-performance.en.md).

## Capability differences from 910B

Jittor identifies the chip at runtime through `aclrtGetSocName`
(`backends/acl/kernels/_soc.py`). The following are measured 310P differences, not guesses:

| Capability | 310P status | Behavior |
| --- | --- | --- |
| BF16 | no `DT_BFLOAT16` kernel | construction/conversion errors immediately, with a clear reason |
| FP32 matmul/bmm/conv | cube does not support `KEEP_DTYPE` | errors by default; `acl_allow_hf32` explicitly reduces precision |
| `StridedSliceAssignV2`, `EmbeddingDenseBackward`, `RotaryPositionEmbeddingGrad`, `FlashAttentionScore`, `NanToNum`, fixed-window `Maxpool`/`Avgpool`, `IndexPutImplAccumulate` | CANN not compiled for 310P | automatically uses a device-side composite implementation |
| comparisons on non-finite values | does not follow IEEE (`x==+inf` matches `{nan,+inf}`) | `isnan`/`isinf`/`isfinite`/`isposinf`/`isneginf`/`nan_to_num` implemented with equivalent invariant combinations |
| `aclnnAdaptiveAvgPool2d` | available but internally accumulates at low precision (~4e-5) | precision matches torch_npu through the same CANN kernel |
| `RotaryPositionEmbedding` | `seq_len==1` returns garbage (tiling boundary) | that case automatically switches to the composite `x*cos+R(x)*sin` forward |
| `rfft` (DFT constant FP32 matmul) | no usable precision path | declared unsupported |

FP16 gradient-free inference can use the inference card's attention operators through the
standard SDPA entry: causal equal-length prefill without an explicit mask goes through
`PromptFlashAttentionV3`, padded device-side to a multiple of 128 and trimmed back to the
valid queries afterwards; single-query decode goes through `IncreFlashAttentionV4`. These
two interfaces are different operators from the missing `FlashAttentionScore`. Transformers
models select `attn_implementation="sdpa"` when loading; calls that do not meet the
verified conditions continue to use the original composite path.

## FP32 matrix operations: explicit reduced-precision switch

Corresponding to `torch.npu.matmul.allow_hf32`, Jittor provides `acl_allow_hf32`, and the
three ways to enable it are equivalent:

```python
import jittor as jt

jt.acl_allow_hf32 = 1                            # 模块属性
# 或
jt.flags.acl_allow_hf32 = 1                    # flags
# 或局部生效
with jt.flag_scope(acl_allow_hf32=1):
    y = a @ b
```

The environment variable `acl_allow_hf32=1` also works. Once enabled, FP32 matmul/bmm/conv
runs with `ALLOW_FP32_DOWN_PRECISION` (internal FP16 splitting; measured maximum error
about 1e-3). **When it is not enabled, FP32 matrix operations error immediately instead of
silently reducing precision** -- this is deliberate: full precision is the default
promise, and reduced precision must be explicitly agreed to. The switch also takes effect
on 910B (voluntary reduced precision), but the default FP32 path itself is usable there.

## FP16 inference weight pre-conversion

When linear-layer weights are no longer updated, they can be converted once on the device
to CANN's NZ layout, avoiding repeated conversion on every decode step. First load a normal
checkpoint, migrate it to ACL, and call `eval()`, then convert explicitly; models using the
standalone Torch frontend need their `Linear` type passed in:

```python
import torch
from jittor.backends.acl.inference import freeze_linear_layers_

model.eval()  # model 已加载权重并位于 ACL
with torch.no_grad():
    model = freeze_linear_layers_(model, linear_type=torch.nn.Linear)
    output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
```

Native Jittor models use `jt.no_grad()` and the default `linear_type=jt.nn.Linear`.
The root module returned by the conversion must be kept: when the root itself is a Linear
it is replaced too. Currently only FP16, ACL, and gradient-free inference are supported;
training calls error; plain Linear subclasses are kept as is.
Linear layers with forward/pre hooks or instance forward/execute overrides refuse
conversion, to avoid silently dropping custom computation; when conversion fails the
original module tree stays intact.

The converted `FrozenLinear` keeps an independent weight snapshot, so later modifications
to the source weights do not affect the snapshot. It registers only `packed_weight` and an
optional bias buffer, and keeps no trainable weight Parameter; `parameters()` therefore
reports fewer entries, but the number and precision of the weights used in computation are
unchanged. After conversion `state_dict()` is in packed format; to load a normal checkpoint,
rebuild a normal model, load, and convert again. Weight conversion time should be recorded
separately and not counted in steady-state token throughput.
Linear weights originally shared with an Embedding get an independent snapshot, so
conversion may increase device memory usage.

## Known limitations

- The backward pass of `x[i]` (pure integer indexing) may read uninitialized bytes on 310P:
  this is a scheduler-layer defect after the `setitem_inplace` graph optimization packs the
  gradient into a zero base, unrelated to ACL kernels. The public path (`GetItemACL`) has
  been worked around; the native op path is declared as an expected failure in tests, see
  `agent/manuals/known-issues.md` KI-BACKEND-022.
- bool-mask `where`/`masked_fill` silently have no effect on 310P; paths involving
  non-finite values have switched to `nonzero`+`IndexPutImpl`.
- 310P makes no training claims: missing backward operators are filled in with composite
  implementations and cross-checked one by one, but no end-to-end training validation has
  been done.

## Quick verification

```bash
source /path/to/cann-9.1.0/set_env.sh
JITTOR_TORCH_SHIM=0 PYTHONPATH=python \
  JITTOR_HOME="$ACL_TEST_STATE/native-cache" \
  python -m pytest -q tests/backends/acl/ \
  --ignore=tests/backends/acl/test_acl_torch_compat.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
  JITTOR_HOME="$ACL_TEST_STATE/torch-cache" \
  python -m pytest -q tests/backends/acl/test_acl_torch_compat.py
```

`ACL_TEST_STATE` should point to a separate run directory under the lab. The two frontends
run in separate processes and cannot be collected together. The current tree has added
operators and lifecycle regressions, so 214/11 is a historical count; skips must have a
capability classification, and the device tests that actually run are authoritative. If an
FP32 matmul test reports an `acl_allow_hf32`-related error, the test already adds the
opt-in automatically with `_soc.supports_fp32_cube()`; normally no manual setting is needed.
