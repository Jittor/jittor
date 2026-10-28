# A temporary input freeze must not erase a recorded grad_fn

- Status: verified for the stated CPU/CUDA contracts; KI-COMPAT-004 remains open.
- Date: 2026-10-16
- Production source: `819182d1b333b4f300f8c530ecb97c3a96287084`.
- Baseline before fix: `dfe462aa10bb3abc6aedce30c11e54a5b42867e7`.
- Owner: native autograd / Torch compatibility maintainers.
- Review when: producer queries, edge snapshots, reversible requires-grad policy or alias creation changes.

## Problem and fix

Create a gradient-requiring `result = torch.Tensor(source)`, `source[:]` or `source * 2` first,
then call `source.requires_grad_(False)`; genuine Torch still treats result as a nonleaf node
carrying the original grad_fn. The old candidate instead reclassified result as a leaf and
returned an empty grad_fn. All six CPU directed cases for FP32/FP64 failed; the same six
assertions pass under independent binary Torch 2.11 on CPU and on real CUDA respectively.

The native `backward_grad_fn` query read the input's current temporary freeze flag, conflating
the recorded producer with the current gradient-receiving target. The fix still checks the
output's own state, permanent stop-grad, control edges and the snapshot of edges disabled at
operation-creation time, but a reversible freeze applied to the input afterwards no longer
removes an existing producer. No grad_fn, leaf identity or gradient is forged in the
compatibility layer. The native JIT test covers both a producer frozen afterwards staying
unchanged and an edge disabled before creation remaining excluded.

The new regression also actually computes result's gradient and, after the source is unfrozen
again, computes the source gradient along the old graph, checking complete numerics and device.

## Validation results

| Scope | CPU | Real CUDA |
| --- | --- | --- |
| Four-file shared regression: constructor alias / tensor state / autograd / autograd semantics | 118 passed / 47 accelerator skips | 165 passed / 0 skipped |
| Native `test_backward_leaf_query.py` (including C++ JIT tests) | 20 passed | 20 passed |
| Same six new assertions in genuine Torch | 6 passed, 2.11.0+cpu | 6 passed, 2.11.0+cu128 |

The CUDA shared regression took 152.79 seconds and the native query 4.65 seconds; these are
validation timings and grant no performance conclusion. The candidate's shim-only fixture uses
fallback=error and synchronizes real tensors; each new gradient check inspects the actual device.
The unit-test cache is reused normally without copying compiled binaries; it is separate from the
downstream benchmark cache.

Canonical core: **322 passed / 42 skipped / 1 existing xfail**, with both modes exiting zero.
Complete `tests/structure + compat/tests/structure`: **1622 passed / 6 skipped**, 224.75 seconds.
The unit tests run against a frozen five-file workspace whose bytes were verified file by file
against 819182d1b before committing.

## Reproduction and original evidence

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TORCH_SHIM=1 \
  <jittor-python> -m pytest -q compat/tests/torch/test_tensor_constructor_alias.py \
  compat/tests/torch/test_tensor_state.py compat/tests/torch/test_torch_compat_autograd.py \
  compat/tests/torch/test_torch_compat_autograd_semantics.py
CUDA_VISIBLE_DEVICES=<gpu> JT_USE_CUDA=1 JITTOR_TORCH_SHIM=1 \
  <jittor-python> -m pytest -q compat/tests/torch/test_tensor_constructor_alias.py \
  compat/tests/torch/test_tensor_state.py compat/tests/torch/test_torch_compat_autograd.py \
  compat/tests/torch/test_torch_compat_autograd_semantics.py
JITTOR_TORCH_SHIM=0 <jittor-python> -m pytest -q tests/core/test_backward_leaf_query.py
```

The independent oracle calls the same `test_recorded_grad_fn_survives_source_freeze` test
function directly with six parameter combinations and asserts genuine Torch's binary identity
and device; it loads neither Jittor nor the shim-only fixture. The first three pytest oracle
launch attempts were rejected by the dynamic-library ABI, a shim conflict and the process-mode
policy respectively; none was treated as a numerical validation pass. The final direct-call
method is consistent with the existing independent constructor oracle. All original logs are
retained.

Unversioned evidence is at `$JITTOR_LAB_ROOT/_state/tensor-alias-guards-dfe/20261016/`:
`directed-before.log`, `oracle-recorded-{cpu,cuda}-v1.log`, `shared-{cpu,cuda}-v1.log`,
`native-{cpu,cuda}-v1.log`, `core-v1.log`, `structure-v1.log`, the complete source fingerprint
and the original diagnostic scripts. Rechecking 32 real lifecycle combinations reduced the
candidate differences from 18 to 17; the fixed source-freeze scenario matches genuine Torch
exactly.

## Still open

The no-grad alias rejection guard after a source or alias write, the dynamic unfreeze flag
inherited from a frozen source, and the nonleaf requires-grad modification rule still differ;
KI-COMPAT-004 remains open. This grants no complete view semantics, complete C-extension ABI,
other accelerator backends or support for unexecuted downstream consumers, and it does not mean
that R33's PR maintainer disposition is complete.
