# Tensor detach and factory gradient boundaries

- Status: CPU and CUDA contracts verified; DeepSpeed multi-rank bf16 acceptance remains incomplete
- Date: 2026-10-15
- Baseline commit: `4a80afb6606c428c200ba84a29d178abe52284b9`
- Source fix: `4d2936e4340995767694f9c1794e7b7690f8e803`
- Synchronized verification: `9f1dc97df75898791e7f8e33e537c09ba7b1afd0`, including remote `aec3b683bffdd85fae42b3b94d8609c0eed7cf2f`
- Owner: Jittor core and Torch compatibility maintainers
- Review when: native view provenance, holder rebinding, gradient publication, tensor factories or forward-mode AD support changes

## Confirmed defects and owner fixes

Real two-rank DeepSpeed 0.19.7 bf16 training first reached native NCCL initialization
and SUM reduction, then failed because `Tensor.detach_` was absent. Independent
binary Torch CPU/CUDA contracts confirm that it returns the same tensor, retains
storage and an existing gradient, preserves consumers created before detachment,
and rejects views, including views whose temporary base object has died.

The compatibility method now reuses the native leaf grad policy or native
storage-sharing detached alias. A native read-only provenance query retains view
identity after its weak writeback owner expires; storage and view writeback
ownership are unchanged. Fidelity remains approximate: the tested reverse-mode
contracts apply, while forward-mode AD is unsupported.

Default backward also published non-optimizer gradients with differentiable
history. It now detaches them when `create_graph=False`; `create_graph=True`
retains the history and passes a real second-derivative test. Repeated backward
still accumulates into the same gradient object.

Finally, constructing `torch.tensor(..., dtype=bfloat16, requires_grad=True)` on
CUDA produced a cast non-leaf and omitted its requested gradient. The factory
now detaches its completed allocation/conversion before enabling gradients.
`tensor(source)` still copies and cuts history; `as_tensor(source, dtype=...)`
preserves conversion history. CPU float32/float64 and CUDA float32/float64/bf16
are exercised; this report does not claim CPU bf16 arithmetic.

## Validation

Python 3.11.16, Linux x86-64, RTX 4090/SM89, driver 580.178.04. Native CUDA uses
real nvcc 12.2.140 and provider `pipcu122_a89fa4fed08c`. The independent reference
is binary PyTorch 2.11.0+cu128; its `_C` extension and absent shim marker are
checked. Each runtime/device and distributed rank has a separate external cache;
first native/NCCL builds were serial. Concurrent unrelated jobs prevent a
performance conclusion.

Raw logs are unversioned under
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/`. Portable reproduction uses
separate device processes with `JITTOR_TORCH_SHIM=1`, source-pinned `PYTHONPATH`,
and task-local `JITTOR_HOME`/`TMPDIR`; CUDA additionally sets
`JITTOR_TEST_DEVICES=cuda`, an allocated `CUDA_VISIBLE_DEVICES`, and real nvcc.

| Validation on synchronized source | Result | Log |
| --- | --- | --- |
| New detach/factory CUDA contracts inside `forbid_backend_fallbacks()` | 9 passed, zero skipped, fallback delta zero | `detach-factory-cuda-synced-no-fallback.log` |
| Autograd/hooks/callbacks/optimizer/view/factory/dtype regression, CPU | 170 passed, one existing xfail, zero skipped | `detach-expanded-cpu-synced-v3.log` |
| Same expanded regression, CUDA | 170 passed, one existing xfail, zero skipped | `detach-expanded-cuda-synced-v2.log` |
| Official `python tools/run_test_suite.py --tier core`, CPU, both process modes | 322 passed, 42 skipped, one existing xfail | `detach-core-cpu-synced-v2.log` |
| Complete Torch-mode `python -m pytest -q tests/structure` | 1460 passed, six skipped | `detach-structure-synced-v1.log` |
| Independent Torch executes the same seven detach and two factory methods, CPU and CUDA separately | 9 methods per device passed | `detach-oracle-contract-{cpu,cuda}.log`, `factory-leaf-oracle-{cpu,cuda}.log` |
| `bash tools/check_repo_layout.sh` | Passed before report registration | terminal output |

The existing xfail is `test_placeholder_dtypes_are_metadata_only_at_every_boundary`
(`KI-COMPAT-014`). Core skips reflect its existing device/opt-in coverage; the
six structure skips comprise two accelerator checks, two declared checks and
two isolated interpreters without an installed Jittor distribution. Counts in
the table overlap and are not additive.

Run new contracts with
`python -m pytest -q compat/tests/torch/test_tensor_detach_inplace.py compat/tests/torch/test_tensor_factory_leaf.py`.
The expanded command adds the existing gradient-hook, callback, task-participation,
autograd/semantics, view-storage, optimizer, autograd-library-owner, factory-fidelity,
dtype and tensor-to-dtype test files. Source hashes are sealed externally in
`detach-source-synchronized.json`.

The first core attempt waited on a fresh oneDNN source download and was interrupted;
the accepted run explicitly used an existing real oneDNN include/library pair.
Earlier CUDA factory failure and DeepSpeed `detach_` failures remain in external
logs. After these fixes, actual two-rank bf16 backward progressed to the next
missing interface, `torch._foreach_zero_`; no completed multi-rank Engine training
or numerical comparison is claimed here. That acceptance requires separate
optimizer partition, all-gradient, training and fresh-Engine resume evidence.

## Main-branch integration verification

The changes were integrated over `9cc83da6e2b32a170c23b7558ab445b75c5b4fd4`.
All seven source and test files match the synchronized verification commit
listed above; hashes are recorded in the unversioned
`$JITTOR_LAB_ROOT/_state/tensor-detach-integration/source-equivalence.json`.
On the integrated tree, CPU core passed 322 tests with 42 skips and one existing
xfail, the two new contract files passed nine tests each on CPU and actual CUDA
with zero skips, and the complete Torch-mode structure suite passed 1460 tests
with the same six skips described above. Repository layout and diff checks passed.
The corresponding logs are `core.log`, `cpu-contracts.log`, `cuda-contracts.log`
and `structure.log` in that external integration directory. These checks validate
the integrated tensor changes; multi-rank DeepSpeed acceptance remains pending.
