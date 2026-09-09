# 昇腾 ACL：整改期迁移记录

本文是 `2.0-refactor` 期间 ACL 后端迁移的过程记录：共享 launcher 迁移的收口状态、
无 CANN 环境下的语法检查方法、异机上的 Python dispatch 迁移与自定义 ACL 源码迁移、
以及属性数据通道的 owner。面向用户的昇腾安装与使用说明见
[`docs/guides/ascend-910b.md`](../../docs/guides/ascend-910b.md)。

## Shared launcher migration is closed for the standard owners

The last four standard owners that still drove the aclnn execute call
themselves now use `BaseOpRunner::launch`: SWhere, Sigmoid backward, BatchNorm
forward, and BatchNorm backward. Each kept its synchronous execution policy,
BatchNorm kept its training/momentum/epsilon attributes and three outputs, and
BatchNorm backward still frees its output mask after the launch has
synchronised.

This matters beyond tidiness. Those four were the last places where a failed
`aclnn` execute only printed a line and returned, leaving the output var
undefined while the graph continued. Routing them through `launch` makes the
failure raise with the operator name and decoded ACL status.

No hand-rolled tail remains: all 71 `executeOp` owners are tail-free. The last
holdout was reduce prod, whose three paths -- whole tensor, one axis, and a
staged multi-axis reduction over intermediate tensors -- now each call `launch`,
the staged one asynchronously so its unconditional barrier still lands before
the intermediates are freed. Two owners keep a `syncRun()` of their own for a
reason the shared tail cannot express: the AdamW loop synchronises once after
its last step rather than once per tensor, and the staged product path after
freeing its intermediates. KVCacheMemcpy never had a tail; it is a per-token
`aclrtMemcpyAsync` path with no aclnn workspace executor.

The prod conversion also closed a silent failure: `ret = aclnnProd(...)` was
assigned and never read, so a failed product returned the untouched output
buffer as the reduction with nothing logged. The same shape applied to a failed
`aclnnMaxDim`/`aclnnMinDim` workspace query, which printed a line and returned
with both the values and the indices output left uninitialised. Both raise now.

Validate on an Ascend 910B3 after sourcing CANN and confirming the device:

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/acl/test_acl.py \
  tests/backends/acl/test_acl_dtype_preservation.py \
  2>&1 | tee "$TMPDIR/acl-launcher-close.log"
```

The run counts only if all four owners actually executed on the NPU with
independent result/gradient checks. Each owner must synchronize inside
`forbid_backend_fallbacks()` and record a zero `backend_fallback_count()` delta.
Fallback attempts are NOT NPU validation, even if the request was rejected and
caught by the test. Preserve `execute launcher failed` and
`aclrtSynchronizeStream failed` diagnostics to attribute failures; log matching
is not the fallback acceptance gate.

Then repeat with `sync_run=0` to confirm the asynchronous path still launches
on ACL, applying the same runtime scope/counter checks.

Until that run exists, this is a source-only change. Hosts without CANN and an
Ascend device must not report hardware validation for it.

## Syntax-checking ACL sources without CANN

A host without CANN can still parse these sources against a generated stub SDK;
see `agent/skills/acl-host-syntax-check`. That check catches parse errors,
unknown identifiers, and a workspace query passed where a launcher belongs. It
cannot check the arguments of an `aclnnXxxGetWorkspaceSize` call, because those
signatures are not knowable without the SDK. It is not hardware validation.

## Python dispatch migration on another machine

The `2.0-refactor` Python dispatcher now takes its target from
`jt.core.dispatch_context(inputs)`. On a configured Ascend runtime this must
report `("acl_legacy", device_id)` while accelerator mode is enabled. A pending
or host-staged tensor is not evidence of a CPU execution target. This migration
has CPU and CUDA evidence only; the ACL registrations still need a real CANN
build and execution on the target machine.

After the CANN and device preflight above, run the existing RMSNorm, rotary,
SiLU and attention nodes with NPU selection, for example:

```bash
PYTHONPATH=python JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
python -m pytest -q -s tests/backends/acl/test_acl_torch_compat.py \
  -k 'rms_norm or rotary or silu or sdpa'
```

Record actual executed node ids, numerical/gradient comparisons, physical NPU
residency and zero CPU fallback using the checks above. A collected or skipped
node is not validation. Do not replace missing hardware evidence with the
host-only registry tests. Rebuild extensions against the current core.

Legacy `nn.backends.hooks.*`, `jt.cudnn` and `compile_extern.*` library attributes
are now read-only views. Backend integrations publish through
`_runtime.dispatch.register_kernel` and `_runtime.backend_libraries`; temporary
test overrides use `override_kernel`, which also restores absent registrations.
ACL implementations now register without `change_function` or public Module
replacement. The existing ACL source conversion remains until the separate
native SDK migration is complete.

### Custom ACL source migration

Custom SDK snippets must declare their source backend explicitly:

```python
result = jt.code(shape, dtype, inputs, cuda_header=acl_header,
                 cuda_src=acl_source, backend="acl")
```

The former `// aclop` marker is no longer a dispatch mechanism. The same
`backend="acl"` argument is required for preallocated `outputs=` snippets;
gradient CodeOps inherit it automatically. This does not select an NPU or make
an arbitrary CUDA kernel valid ACL code. Wrong accelerator targets fail before
execution; an optional `cpu_src` remains the CPU implementation.

The canonical Python SDK builders are in `jittor.backends.acl.kernels.ops`;
legacy `jittor.extern.acl.aclops` imports resolve as same-object aliases.
Rebuild native extensions against BackendOps ABI 2 and the updated CodeOp
constructor. On the target CANN machine, rerun indexing forward/backward and
writeback, normalization, convolution, explicit CodeOp outputs/gradients, and
HCCL nodes under the strict fallback policy above. Host syntax checks and
CPU/CUDA regressions are not evidence that these SDK calls passed on NPU.

Native basic Getitem/Setitem now use checked ACL device-copy plans for integer
indices, positive-step slices, new axes, ellipses, empty selections and broadcast
assignment. Validate chained forms such as `x[0][1] = value`, fp16/bf16 indexing
gradients with float32 upstream weights, and `return_x` on the target machine.
Copies must follow `aclstream` producers; a final synchronization alone cannot
repair a missing producer dependency. Non-equivalent storage overlap is rejected
before writes. Advanced/string/negative-step indexing and reduction assignment
are not implemented by this native copy entry; existing Python ACL paths may
handle variants before reaching it. Scalar broadcast currently uses repeated
device copies and has not been optimized or performance-validated.

## ACL attribute data-channel owner

The host-only attribute channel is defined by
`backends/acl/include/aclops/acl_data_channel.h`. It is intentionally
compiled before any CANN probe: the header owns the versioned record, typed
scalar/vector values, schema defaults, and deterministic cache key, but it does
not call ACL or construct an `aclTensor`. The corresponding source contract is
`tests/structure/backends/acl/test_acl_data_channel_contract.py` and can be run on a CPU
host:

```bash
python -m pytest -q \
  tests/structure/backends/acl/test_acl_data_channel_contract.py \
  tests/structure/backends/acl/test_acl_data_schema_normalizer.py
```

For the future generated-attribute consumer, `AclDataOwner::consume` supplies
a bounded `AclDataView` with typed scalar/vector accessors. This view is only
valid during the callback, so an eventual ACL adapter must copy values into its
`OpAttr`/descriptor arguments before returning. The host contract exercises
that callback and its wrong-type failure path; it does not construct an
`aclTensor`, invoke `aclnn`, or imply that the attribute channel is wired into
an existing NPU operator.

`AclAttrRunnerContract` freezes the next migration boundary: each registry
entry supplies a fixed list of `AclAttrBinding` names and types. Duplicate,
undeclared, or type-incompatible bindings fail as internal integration errors
before any record is decoded. Its host-only `consume()` callback is the shape
that a future `BaseOpRunner` adapter will use to copy values into `AclOpAttr`;
it does not include CANN headers or allocate device descriptors. The CPU
contract test therefore proves only registration and decoder behavior, not
NPU execution.

On an Ascend 910B3, the first attribute owner must be validated only after its
Python caller, generated `OpAttr` construction, and JIT key are migrated in the
same change. Source the CANN environment, select the allocated card, and run
the owner-specific node (the example below uses the current softmax/triu
cohort):

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>
set -o pipefail
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/acl/test_acl_torch_compat.py -k 'softmax or triu' \
  2>&1 | tee "$TMPDIR/acl-attribute-data.log"
```

Record the 910B3 model, CANN version, selected device, exact node ids, and the
zero native fallback-attempt delta in the handoff. The owner-specific node must
enclose device execution and synchronization in `forbid_backend_fallbacks()`.
A host-only decoder pass does not close
8.06 and must not be reported as NPU hardware validation.


## 各算子族迁移到共享 launcher 的状态（逐条记录）

The ACL binary operator family uses the shared launcher contract for workspace
allocation, execution errors, and synchronization. This source-only change
still requires the 910B3 probe below; hosts without CANN and an Ascend device
must not report hardware validation.

The single-step ACL reduce owners (ReduceSum, Mean, Amax, and Amin) also use
the shared launcher and retain synchronous execution. Product reductions with
multiple axes remain on their dedicated intermediate-buffer path.

The unary runner is the first family migrated to the shared launcher tail. It
keeps its historical asynchronous policy, while workspace allocation and ACL
launch failures now use the same auditable error path as the base runner. When
validating this migration, include one unary operation under
`forbid_backend_fallbacks()` and materialize its result inside the scope.
An `execute launcher failed` message is failure-attribution evidence, not a
replacement for the runtime counter check.

The ACL Cumsum family uses the shared launcher and retains synchronous
execution. This remains source-only until the Ascend 910B3 probe is run.

The ACL MatMul family uses the shared launcher while retaining its synchronous
execution policy. This source-only migration still requires the Ascend 910B3
probe and is not hardware validation on this host.

The ACL Expand family uses the shared launcher and intentionally retains its
historical asynchronous execution policy.

The ACL Floor family uses the shared launcher and retains synchronous
execution. This source-only migration still requires the Ascend 910B3 probe.

The ACL NanToNum family uses the shared launcher while retaining its nan,
posinf, and neginf attribute handling and synchronous execution policy.

The ACL Triu family uses the shared launcher while retaining its diagonal
attribute and synchronous execution policy.

The ACL Sigmoid forward family uses the shared launcher and retains synchronous
execution. Sigmoid backward is intentionally outside this slice. This remains
source-only until the Ascend 910B3 probe is run.

The ACL Transpose/Permute family uses the shared launcher while retaining axes
descriptor cleanup and synchronous execution.

The ACL Softmax forward family uses the shared launcher and retains synchronous
execution. Softmax backward is intentionally outside this slice. This remains
source-only until the Ascend 910B3 probe is run.

The ACL Embedding forward family uses the shared launcher and retains
synchronous execution. Embedding backward is intentionally outside this slice.

The ACL Embedding backward owner uses the shared launcher while retaining
numEmbeddings, paddingIdx, scaleGradByFreq, and synchronous execution.

The ACL Roll family uses the shared launcher while retaining shifts/dims array
cleanup and synchronous execution. This remains source-only until the Ascend
910B3 probe is run.

The ACL Gather family uses the shared launcher and retains its dimension
parameter and synchronous execution policy. Scatter is outside this slice.

The ACL ClampTensor family uses the shared launcher while retaining its three
input query and synchronous execution policy. This remains source-only until
the Ascend 910B3 probe is run.

The ACL Stack family uses the shared launcher while retaining tensor-list setup,
dimension handling, and synchronous execution. Tensor-list lifetime cleanup is
outside this migration slice.

The ACL Flip family uses the shared launcher while retaining axes setup and
synchronous execution. Its pre-existing aclIntArray lifetime issue is outside
this migration slice.

The ACL Scatter family uses the shared launcher while retaining axis/reduction
handling and synchronous execution. Gather remains outside this slice.

The ACL Concat family uses the shared launcher while retaining tensor-list and
dimension handling. SplitWithSize is intentionally outside this slice.

The ACL SplitWithSize family now uses the shared launcher while retaining its
split-size, tensor-list, and dimension handling. Tensor-list lifetime cleanup
is outside this migration slice.

The ACL Nonzero owner uses the shared launcher and retains synchronous
execution. The SWhere owner is outside this slice.

The ACL Range owner uses the shared launcher while retaining scalar creation,
cleanup, and synchronous execution. Scalar cleanup on exceptional exits is
outside this migration slice.

The ACL Dropout forward owner uses the shared launcher while retaining its
probability, training, seed, offset, and dual-output handling. Dropout backward
is intentionally outside this slice.

The ACL LeakyReLU forward owner uses the shared launcher while retaining its
negativeSlope scalar and synchronous execution. LeakyReLU backward is outside
this slice; scalar exceptional cleanup is unchanged.

The ACL LeakyReLU backward owner uses the shared launcher while retaining its
negativeSlope, selfIsResult, scalar cleanup, and synchronous execution.

The ACL SiLU forward owner uses the shared launcher and retains synchronous
execution. Backward, Swish, and SwiGlu remain outside this slice.

The ACL SiLU backward owner uses the shared launcher and retains synchronous
execution. Swish and SwiGlu remain outside this slice.

The ACL Swish forward owner uses the shared launcher and retains synchronous
execution. Swish backward and SwiGlu remain outside this slice.

The ACL Swish backward owner uses the shared launcher and retains synchronous
execution. SwiGlu remains outside this slice.

The ACL SwiGlu owner uses the shared launcher and retains synchronous execution.

The ACL BatchMatMul owner uses the shared launcher while retaining
cube_math_type and synchronous execution. This remains source-only until the
Ascend 910B3 probe is run.

The ACL TruthReduce owner routes both all and any through the shared launcher
while retaining reduce_all selection, keepdims, and RAII axes descriptors.

The ACL Conv2d forward owner uses the shared launcher while retaining its
group, bias, and convolution descriptor handling. Conv2d backward is outside
this slice.

The ACL Conv2d backward owner uses the shared launcher while retaining its
three-output gradient query and descriptor cleanup.

The ACL RmsNorm forward owner uses the shared launcher while retaining eps and
dual-output handling. Its gradient owner is outside this slice.

The ACL RmsNorm gradient owner now uses the shared launcher and retains its
multi-input, dual-output, and synchronous execution handling.

The ACL LayerNorm forward owner uses the shared launcher while retaining
normalizedShape, eps, three outputs, and descriptor cleanup. Backward remains
outside this slice.

The ACL LayerNorm backward owner uses the shared launcher while retaining its
normalizedShape/outMask descriptors, three-output gradient query, and cleanup.

The ACL GroupNorm forward owner uses the shared launcher while retaining its
group, epsilon, and three-output query. GroupNorm backward remains outside this
slice.

The ACL GroupNorm backward owner uses the shared launcher while retaining its
output-mask descriptor, group attributes, three-output query, and cleanup.

The ACL MaskedSelect owner uses the shared launcher while retaining its
two-input mask query and synchronous execution.

The ACL Index owner uses the shared launcher while retaining its index query
and synchronous execution. SliceV2 remains outside this slice.

The ACL SliceV2 owner uses the shared launcher while retaining begins, ends,
steps, and axes descriptor handling with synchronous execution.

The ACL StridedSliceAssignV2 owner uses the shared launcher while retaining its
gradient memset branch and slice descriptor handling.

The ACL InplaceMaskedScatter owner uses the shared launcher while retaining its
tracked base-to-output device copy dependency and synchronous execution.

The ACL IndexPutImpl owner uses the shared launcher while retaining its
index-tensor-list handling and synchronous execution. Accumulate remains
outside this slice.

The ACL IndexPutImpl accumulate owner uses the shared launcher while retaining
its tracked output memset and index tensor-list dependency.

The ACL FlashAttention forward owner uses the shared launcher while retaining
prefix/qstart/kvstart RAII descriptors and synchronous execution. Backward,
incremental, and KV-cache owners remain outside this slice.

The ACL FlashAttention backward owner uses the shared launcher while retaining
its RAII descriptors, three gradient outputs, and synchronous execution.

The ACL incremental FlashAttention owner uses the shared launcher while
retaining block-table, actual-sequence, cache-view cleanup, and synchronization.
KVCacheMemcpy remains outside this slice.

The ACL AdamWList owner uses the shared launcher for each tensor update while
retaining its fused device-copy checks and single synchronization point after
the update loop.

The ACL Dropout backward owner uses the shared launcher while retaining its
scale query and synchronous execution policy.

The ACL Softmax backward owner now uses the shared launcher while retaining its
dimension query and synchronous execution policy.

The ACL RotaryPositionEmbedding forward owner uses the shared launcher and
retains its three input tensors and synchronous execution. Its gradient owner
is outside this slice.

The ACL RotaryPositionEmbedding gradient owner uses the shared launcher while
retaining its four-input, three-output query and synchronous execution.

The ACL Maxpool forward owner uses the shared launcher while retaining its
kernel, stride, padding, dilation, and ceil-mode descriptors. Avgpool and
backward owners remain outside this slice.

The ACL Avgpool forward owner uses the shared launcher while retaining its
pool descriptors, ceil mode, divisor, and padding semantics. Maxpool and
backward owners remain outside this slice.

The ACL Avgpool backward owner uses the shared launcher while retaining its
countIncludePad/divisorOverride semantics and descriptor cleanup.

The ACL Maxpool backward owner uses the shared launcher while retaining its
pool descriptors, ceil mode, output handling, and cleanup.

The ACL UpsampleNearest2d forward owner uses the shared launcher while retaining
output-size RAII and synchronous execution. Backward is outside this slice.

The ACL UpsampleNearest2d backward owner uses the shared launcher while
retaining output/input-size RAII descriptors and synchronous execution.
