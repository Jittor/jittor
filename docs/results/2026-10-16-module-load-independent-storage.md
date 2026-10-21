# Checkpoint loads preserve independent model storage

- Status: KI-COMPAT-006 closed for the verified CPU/CUDA copying-load contract.
- Reviewed: 2026-10-16.
- Baseline: `9fdc2b57f1f5deefb585e6145e5ed97fdc3b0ed9` plus this report's two production changes.
- Owner: Torch compatibility and native optimizer maintainers.
- Review when: module loading, strided copies, storage aliases or fused AdamW changes.

Default `load_state_dict(assign=False)` now copies into noncontiguous parameters
through the existing public native `copy_` owner, as it already did for contiguous
parameters. The target keeps its Python object, allocation, strides, dtype and
detached aliases. The prior noncontiguous path rebound source storage: a subsequent
fused CUDA AdamW update of the source silently changed the target too. Both FP32
and BF16 reproduced the allocation/stride defect on CPU; FP32 reproduced the silent
cross-model update on CUDA. CPU's rebinding optimizer happened to hide that update.

The BF16 CUDA reproduction also exposed a native fused AdamW compilation defect.
Half-precision overloads in the enclosing namespace hid the double-precision
`pow` overload. Qualifying the two existing calls as `::pow` fixes FP16/BF16 CUDA
compilation without changing the bias-correction formula or its precision.

Before the fix, the maintained loading regression passed two CPU cases and failed
two strided CPU cases; four CUDA cases were skipped. The maintained native CUDA
regression passed FP32 and failed compilation for FP16/BF16. Original failures
are retained. After the fix:

| Command scope | Result |
| --- | --- |
| Three module-load files, CPU | 14 passed, 4 CUDA skips |
| Same three files, CUDA-visible | 18 passed |
| New loading test unchanged in genuine Torch, CPU | 4 passed, 4 CUDA skips |
| New loading test unchanged in genuine Torch, CUDA-visible | 8 passed |
| Native fused AdamW, optimizer core and RMSprop contracts, CUDA | 49 passed |

CUDA allocation checks use actual driver pointer attributes. The loading regression
executes a real backward and the original fused AdamW kernel, counting its two
parameter entries; it checks that only the source model changes. Whole-call
candidate guards report zero fallback on CPU and CUDA. Genuine references are
independent Torch 2.11 CPU and CUDA installations, without the Jittor shim.

The native fused test covers FP32/FP16/BF16, two parameters and three steps, checking
all 54 parameter/moment arrays against fresh genuine Torch CUDA goldens. It uses
`beta1=beta2=0`, zero weight decay and `lr=eps=.125`, isolating compilation and
storage writes from unrelated low-precision rounding differences. Low-precision
arrays match exactly; FP32 uses `rtol=1e-6, atol=1e-7`. Every output keeps
its input GPU storage pointer. Host snapshots copy first to avoid migrating the
original allocation.

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_module_load_independence.py \
  compat/tests/torch/test_module_load_grad_mode.py \
  compat/tests/torch/test_module_load_storage.py

# In a real CUDA-enabled native environment:
JITTOR_TORCH_SHIM=0 PYTHONPATH=python python -m pytest -q \
  tests/optim/test_fused_adamw_device_contract.py \
  tests/optim/test_optim_core.py tests/optim/test_rmsprop_device_contract.py
```

This result does not establish generic saved-value/version semantics, complete
optimizer ABI, other accelerators or full-model support. `assign=True` retains
its separate assignment behavior. Raw commands, failed attempts, exact source
hashes, genuine golden provenance, loaded core copies and compiled CUDA kernel
snapshots are unversioned under
`$JITTOR_LAB_ROOT/_state/module-load-independence/20261016/`.
Two early CPU launch attempts were rejected before collection because they
inherited a CUDA-required gate; the final CPU command explicitly selects CPU.

Integration core passes **322 / 42 skips / 1 existing xfail**; combined native and
compatibility structure gates pass **1628 / 6 skips**. Repository layout, generated
manifest and new-test Ruff checks pass. These checks do not expand device coverage.
