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
