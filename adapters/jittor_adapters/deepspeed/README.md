# DeepSpeed explicit adapter

## Scope and ownership

This independently installed adapter targets DeepSpeed **0.17.6**, with exact
source hashes in `source.py`. The six critical upstream source files each
have exactly one accepted hash. The generated `git_version_info_installed.py`
has exactly two accepted hashes, from these fixed builds of the original sdist
SHA256 `b3318064ee5798e8a27d201ea8b888f0439973c4eac9af9ab381dd1862ebdf45`:

- NPU manifest: `ad44588cefdf97440706c2c62229619ca0f3801501d9ddc71e43abaa3b49adff`.
- CPU manifest: `ab1206bd635f789c618086e23bfffd24849012a564a1b7852dec0e537ba60f58`.
  Built with genuine PyTorch 2.7.1, `DS_ACCELERATOR=cpu DS_BUILD_OPS=0
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 pip wheel --no-deps --no-build-isolation`.
  This CPU wheel passed independent import/config/model construction comparison;
  admitting its source identity does not transfer results from another build.

All other manifests fail closed, even if their version string is identical.
`status()["source_sha256"]` lists accepted hashes per file; this is source
identity information, not a device or L-level support claim. These checks cover
the six named upstream files and build manifest, not every file of the distribution.

This is a third-party protocol adaptation: DeepSpeed chooses its accelerator
while importing, eagerly imports optional DeepCompile code, and bypasses the
public `Module.add_module` API. Tensor operations, gradients, optimizers and
communication remain owned by Jittor/torch compat. The adapter implements none
of that mathematics and provides no PyTorch ABI extension replacement.

Importing this adapter alone is inert. It has no automatically activated entry
point. It uses the public `jittor.compat.module_patcher.patch_method` and
`jittor.compat.transaction` ownership contracts plus a private finder limited
to six DeepSpeed module names. Installed DeepSpeed files remain unchanged.

## Explicit activation

After selecting Jittor torch mode, and **before any DeepSpeed import**:

```python
from jittor_adapters.deepspeed import activate, status
activate(device="npu")  # or cpu
import deepspeed
print(status())
```

CPU covers import, configuration and user model construction. CPU engine
construction deliberately raises: an identity single-process collective is
not a Gloo backend. NPU engine construction requires native HCCL WORLD to be
actually initialized, as reported by public
`jittor.distributed.get_hccl_world_info()`. Stage 0 allows world size 1 or 2;
Stage 1/2/3 require a single-node world size 2. Use the native HCCL launcher;
environment variable strings alone are not proof.

NPU scope is FP32 parameters on logical `npu:0`, FP32 or boolean buffers on that
same device, eager ZeRO Stage 0/1/2/3, and an explicitly constructed public
`torch.optim.AdamW`, with foreach/fused/AMSGrad/capturable/differentiable/maximize
modes disabled (`None` or `False`). Stage 1/2 admit default or explicit boolean
`contiguous_gradients`; Stage 3 admits only its default mode. Both persistent
and nonpersistent buffers are allowed. Local-rank settings must select logical
NPU 0.
The configuration must be a dict with only these keys:

- `train_batch_size`, `train_micro_batch_size_per_gpu`, `steps_per_print`;
- `gradient_accumulation_steps=1`, `gradient_clipping=0`;
- `zero_optimization={"stage": 0|1|2|3}`; Stage 1/2 may also set
  `contiguous_gradients` to a boolean, while Stage 3 accepts no extra ZeRO key;
- `fp16={"enabled": False}`, `bf16={"enabled": False}`;
- `wall_clock_breakdown=False`, `memory_breakdown=False`.

Unlisted configuration keys are rejected, including offload, hybrid engine,
activation checkpointing, compiler/profiler configuration and automatic
optimizer construction. Training data loaders, schedulers, MPU and mesh
arguments are outside this scope. Streams/events expose only rejecting types
needed by evaluated annotations; constructing them raises. Allocator/RNG
state/graph methods raise, and extension builders report incompatibility.
Stage 0/1/2/3 training support is limited to the fixed model, device, dtype,
optimizer and rank configurations recorded in the dated maintainer report.
HCCL reduce-scatter currently uses all-reduce plus a device-local rank slice,
so no speed or memory claim can use this provider.

## Lifecycle

Activation is idempotent for the same device and rejects a different device or
an already imported DeepSpeed. Reloading managed modules is rejected before execution.
Version checks use the shared adapter
`require_version` contract, reading the selected source tree's own literal
version before and after import; installed distribution metadata never
substitutes for another source tree. Engine rewriting is exact-source checked
and transforms only the established library-private incompatibilities.

`deactivate()` removes the owned finder, restores owned provider/method writes,
and evicts the DeepSpeed modules imported under this activation. Do not retain
or use references to those modules/models after deactivation. Foreign
replacements are preserved and reported with `TransactionConflict`. Failed
imports roll back owned mutations; retry in a fresh process for acceptance.

## Offline contracts

The default PR `nox -s structure` gate runs the following isolated entry.
Each portable suite runs in a fresh process so controlled fixture modules do
not enter the Torch-mode test process. A core-only checkout without the
independent adapters tree does not require this entry.

```bash
python adapters/tests/test_deepspeed_offline.py -v
```

The portable suites can also be run directly:

```bash
python adapters/jittor_adapters/deepspeed/tests/test_contracts.py -q
PYTHONPATH=adapters python adapters/jittor_adapters/deepspeed/tests/test_relocatable.py -q
```

Outside the monorepo set `JITTOR_COMPAT_SOURCE` to the separately installed
compatibility source package directory. Tests exercise the real public import
and transaction machinery with controlled third-party fixtures. These tests
provide no model or device support evidence. The source audit applies the same
public-import/private-attribute/framework-mutation boundary as the existing
vLLM adapter, scans recursively, excludes tests and includes negative cases.

Review when: the fixed DeepSpeed sources, public `get_hccl_world_info`,
`owned_runtime_hook`, transaction or module-patcher contracts change. Hardware
acceptance and L-level claims belong to the dated maintainer results report.
