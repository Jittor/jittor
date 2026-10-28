# Checkpoint StateDictOptions and submodule contracts

- Status: verified for single-process CPU and CUDA; distributed provider paths retained.
- Reviewed: 2026-10-15.
- Baseline: `9d9310e3c9a4af737702e12d2d6d47062fa015cd`, freshly fetched from
  `origin/2.0-refactor`; implementation and evidence are committed together.
- Owner: Torch compatibility and checkpoint maintainers.
- Recheck when: optimizer ownership, parameter traversal, FSDP checkpoint hooks,
  supported Torch versions, or device-copy semantics change.

The original checkpoint continuation report established ordinary FQN export and
restore, but explicitly excluded nonempty submodule selection. A new minimal
reproduction confirmed that exclusion, an ignored optimizer `strict` option,
and rejection of an alternate FQN for a genuinely shared parameter. Independent
binary PyTorch 2.11.0+cu128 established the supported behavior before changes.
No downstream library or adapter was added.

The implementation stays in `jittor.compat.torch.checkpoint_state`. Native
parameter traversal supplies canonical names and aliases; model loading,
parameter copies, optimizer state installation, and optimizer mathematics remain
with their existing owners. The distributed installer only publishes the
checkpoint functions and options. No Module, inplace, native optimizer, or FSDP
provider implementation changed.

## Supported options and combinations

| Surface | Verified behavior |
| --- | --- |
| `submodules` | Model export selects requested child modules, including persistent buffers, using a complete dotted prefix; selecting `a` does not select `ab`. Multiple children retain distinct prefixes. |
| `keep_submodule_prefixes` | `True` retains model FQNs; `False` produces a selected child's local keys, which can be loaded into that child. |
| Joint `get_state_dict` | Consumes the same submodule and offload options as standalone model export. Optimizer state remains complete and uses root-model FQNs, matching the binary Torch oracle. |
| Module-keyed model loading | A dictionary keyed by a child Module expands local state keys to root FQNs and delegates to ordinary model loading; `strict=False` reports untouched model keys. |
| `ignore_frozen_params` | Filters frozen model parameters, retaining buffers. It does not filter existing optimizer state or parameter groups, matching Torch. |
| `strict` optimizer loading | Rejects a missing trainable parameter state in an otherwise populated optimizer checkpoint; allows missing frozen state. `strict=False` restores supplied state and resets the omitted parameter's state through the optimizer owner. |
| FQNs and groups | Restores by parameter identity despite changed live parameter order and changed serialized FQN order. Saved learning rates and betas are consumed. Partial group membership with `strict=False` preserves the live parameter list and consumes saved options. |
| Shared parameters | Accepts a valid alternate parameter FQN for ordinary and flattened checkpoints, preserves the alias, and exports the canonical FQN again. Duplicate aliases for one saved parameter are rejected. |
| `cpu_offload` | Exports CPU tensors; restore moves moments to the parameter device. The next optimization step, every parameter gradient, and the input gradient match the saved trajectory. |
| `full_state_dict` | Ordinary single-process FULL plus CPU offload roundtrips independently of flattening. FSDP FULL gather/shard continues through the existing provider hooks. |
| `flatten_optimizer_state_dict` | Flatten plus CPU offload roundtrips independently of FULL, including shared-parameter alias restoration. |
| Empty selections and payloads | An empty optimizer iterable makes joint get return `{}` for optimizer state. Joint set also accepts an empty iterator. An empty model dictionary allows optimizer-only joint loading. Supplying an optimizer with an empty optimizer dictionary raises before joint model loading. |

Nonsharded optimizer restoration validates all missing-state load plans before
calling any optimizer's loader. A regression supplies two optimizers, changes
the first one's saved learning rate, and removes a state needed by the second;
the strict failure leaves the first optimizer untouched. This additional
transaction boundary is not a claim about atomic rollback of distributed
collectives or native loader failures.

## Explicit limits

- Loading `full_state_dict=True` together with
  `flatten_optimizer_state_dict=True` raises `NotImplementedError`, including
  joint loading before model writes. The independent Torch oracle also fails
  to load that combination; the three portable continuation combinations are
  ordinary/offload, FULL/offload, and flatten/offload.
- `broadcast_from_rank0=True` still requires `full_state_dict=True`. It delegates
  to existing distributed broadcasts; this report adds no multirank broadcast
  certification. No new multirank FSDP or sharded-submodule acceptance is claimed.
- Prefix removal is verified for one selected child. Callers must avoid
  colliding stripped FQNs across multiple children; binary Torch also permits
  such collisions to overwrite an entry. Root-module and foreign-module
  selections are outside this acceptance matrix.
- Strict loading follows Torch's missing-state contract, rather than imposing
  strict equality on every extra state key: unrelated state entries are ignored,
  and stateless SGD's empty `state` mapping remains loadable. An empty optimizer
  dictionary `{}` is distinct from a valid dictionary containing `state` and
  `param_groups`.
- Frozen model filtering is verified for nonshared child parameters. Optimizer
  aliases are verified separately; combined frozen-model alias filtering is
  outside this matrix.
- The newer Torch-specific `dsd_fqn_modifiers` extension is not added. Existing
  seven-option construction and the tested native optimizer layouts remain the
  supported surface. The public fidelity registration remains `APPROXIMATE`.
- ROCm, ACL, other CUDA architectures, and multirank option combinations were
  not exercised here. The optional Accelerate end-to-end test was skipped
  because that dependency is not installed in the isolated environment; the
  checkpoint tests require no downstream package.

## Device and oracle evidence

Tests ran with Python 3.11.16, GCC 11.5.0, nvcc 12.2.140, the existing pip-cu122
Jittor libraries, and a single visible RTX 4090 (SM89), driver 580.178.04. The
reference interpreter loaded independent PyTorch 2.11.0+cu128 and its binary
`torch._C` extension, with the Jittor source path and CUDA library override
removed. CUDA availability and tensor residency were checked on the actual GPU.
The reference uses CUDA 12.8 libraries; this is a numerical comparison across the
two runtimes on the same device, not certification of a shared toolkit stack.

Fixed input and weight arrays make the trajectory reproducible without RNG or
downstream dependencies. Public `torch.clone` creates immutable snapshots. The
same test file runs in the shim and binary Torch processes. CUDA shim tests
forbid backend fallback, synchronize before exit, verify that model parameters
and restored moment tensors reside on CUDA, and assert zero fallback-count
increase. CPU tests execute with CUDA disabled inside their runtime scope.

The captured next-step trajectory compares 24 numeric fields: loss, input and
all parameter gradients, every model parameter and buffer, and optimizer moments
and step values. Maximum absolute error against binary Torch was `7.45e-9` on
CPU and `1.49e-8` on CUDA, within `atol=2e-6, rtol=2e-5`; fallback increase was
zero on both devices. Parameter group option consumption has separate assertions.

Reproduce with isolated source, home, temporary, and JIT directories according
to `agent/manuals/environment.md`:

```bash
JITTOR_TORCH_SHIM=1 CHECKPOINT_DEVICE=cpu PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_checkpoint_state_options.py \
  compat/tests/torch/test_optimizer_checkpoint_state.py
JITTOR_TORCH_SHIM=1 CHECKPOINT_DEVICE=cuda PYTHONPATH=python \
  python -m pytest -q compat/tests/torch/test_checkpoint_state_options.py \
  compat/tests/torch/test_optimizer_checkpoint_state.py

# Independent binary Torch: use its interpreter, with no deployed shim path.
env -u PYTHONPATH -u LD_LIBRARY_PATH CHECKPOINT_DEVICE=cpu \
  python compat/tests/torch/test_checkpoint_state_options.py -v
env -u PYTHONPATH -u LD_LIBRARY_PATH CHECKPOINT_DEVICE=cuda \
  python compat/tests/torch/test_checkpoint_state_options.py -v
```

Set `CHECKPOINT_OPTIONS_REPORT` to an external JSON destination to capture the
trajectory. Raw logs, JSON reports, private environment overlays, and JIT caches
are unversioned under `$JITTOR_LAB_ROOT/_state/refactor-goal/checkpoint-options/`.

## Validation results

| Selection | Result |
| --- | --- |
| New option tests, binary Torch CPU | 22 passed, 1 intentionally skipped shim-only explicit refusal |
| New option tests, binary Torch CUDA | 22 passed, 1 intentionally skipped shim-only explicit refusal |
| New and existing checkpoint tests, CUDA shim | 29 passed, 1 optional Accelerate skip |
| CPU checkpoint, planner, FSDP2 configuration/lifecycle/parameter-role and optimizer owner regressions | 95 passed, 1 optional Accelerate skip |
| CPU `core`, native mode | 98 passed, 41 skipped, 1 existing expected failure |
| CPU `core`, Torch mode | 224 passed, 1 skipped |
| CPU `core`, both modes combined | 322 passed, 42 skipped, 1 existing expected failure |
| Complete `tests/structure` | 1432 passed, 4 skipped, 2 documentation-integration failures, 628.44 s |
| Repository layout and whitespace | Passed |
| Changed Python files parsed with Python 3.7 grammar | Passed; this is a syntax check, not a Python 3.7 runtime claim |

The final CUDA selection synchronizes and checks physical `location()`, native
dispatch backend, and public device for inputs, outputs, loss, gradients, model
parameters and buffers, and restored moment tensors. CPU uses the same native
dispatch check with CUDA disabled. A separate external interoperability probe
loaded checkpoints actually produced by binary Torch into the shim, using
reversed live parameter slots, then reproduced the reference's next step on both
CPU and CUDA with zero fallback. This probe also consumed real Torch's additional
parameter-group metadata without altering the native AdamW mathematics.

Direct CPU and CUDA checkpoint tests ran sequentially in the task's warmed
CUDA-capable cache, under explicit CPU or CUDA runtime scopes. The CPU core
runner built separate CPU-only native and Torch caches. Initial core and
external-library builds ran serially; no benchmark used these caches. OneDNN
3.9.1 was built in the task state and reused as a read-only external library for
the CPU core sessions.

The two complete-structure failures are
`test_docs_structure.py::TestDocsStructure::test_every_page_is_reachable_and_every_toctree_entry_exists`
and
`test_packaging_structure.py::TestPackagingStructure::test_manifest_covers_runtime_trees_without_cache_payloads`.
Their traces identify only this new report: it needs an entry in the results
toctree and one generated root `MANIFEST.in` include line. The compatibility
manifest already matches its generator. Shared index and manifest updates are
reserved for unified integration; this isolated result does not claim a passing
complete structure gate. Of the four skips, two are declared and two are the
isolated interpreter's unavailable installed-package resolution checks.

Main-tree verification after unified integration is described in
[integration regressions and the NCCL defect closeout](2026-10-15-integration-nccl-coverage.en.md):
the report index and MANIFEST are registered, and the complete structure gate reports 1441 passed, 6 skipped, 0 failed.
All 30 CPU checkpoint items and 37 CUDA checkpoint/Module-state items pass, including
actual Accelerate save and continued training; NCCL 2.18.3 dual-card bf16 FULL restoration and continued training
pass 1 per rank with 0 skipped and fallback 0. This supplements the integration evidence for this selected multi-rank
path, and still does not extend to acceptance of every StateDictOptions multi-rank combination.
