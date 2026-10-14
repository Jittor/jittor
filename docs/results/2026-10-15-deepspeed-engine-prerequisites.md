# DeepSpeed Engine prerequisites: Module storage and final callbacks

- Status: bounded prerequisite validation; **DeepSpeed R14 is not complete**.
- Date: 2026-10-15.
- Baseline commit: `a1a3f44afdcec9d0ce3670ff77e78c09f16b80a2`, plus this report's commit.
- Owner: native Module/autograd and Torch compatibility maintainers.
- Review when: native child traversal, Module serialization/adoption, Tensor backward
  accumulation, optimizer execution policy, or runtime callback ownership changes.

DeepSpeed 0.19.7 reads `engine.__dict__['_modules']` and writes its client model
there. A property returning a temporary dict could not satisfy that contract.
The native Module now exposes a persistent, weakly owned registration view whose
values are the actual child attributes. Assignment, replacement, deletion and
registered `None` values affect the live module, its parameters and checkpoint
keys. Traversal deduplicates aliases where requested and uses a backtracking
ancestor set to avoid copying an ancestor collection at every node.

Pickling a Module rebinds only registration caches that already existed. Native
pooling instances that never cached their children keep their original field
layout. A detached view's copy/deepcopy/pickle produces a usable dict snapshot;
holding the original live view does not keep its parent alive, and accessing that
view after its parent dies raises `ReferenceError`. Native child adoption discards
the source instance's cache and binds a new view to the adopted child.

The published `torch.autograd.Variable` Tensor alias now owns a real final-callback
queue. Callbacks from a successful backward run after gradient accumulation, in
FIFO order; callbacks may append further callbacks. Hook/callback failures propagate
and do not leave callbacks in the next graph. Calls outside backward fail explicitly.
The implementation shares the native differentiation and existing accumulation;
it does not implement a second backward engine. Fidelity remains approximate:
PyTorch worker graph IDs and compiled autograd are not provided.

Adam/AdamW initialization now preserves `foreach` for the existing update selector,
including parameter-group overrides. Dense Var storage exposes `is_sparse=False`.
`GradBucket` and `Adagrad` are importable dependencies of DeepSpeed's module body,
but remain **unimplemented**: construction raises rather than fabricating native
buckets or an optimizer update. Importability does not count as optimizer or
distributed support.

Validation used Python 3.11.16, NumPy 1.26.4, pytest 7.4.4, g++ 11.5 and an isolated
RTX 4090. Native CUDA used genuine nvcc 12.2.140/pip-cu122; the independent binary
oracle was PyTorch 2.11.0+cu128. CPU processes hid all GPUs and explicitly disabled
CUDA. CUDA tensor execution used a separate compilation cache and real device
residency. Raw logs and arrays remain under the lab's
`_state/refactor-goal/deepspeed/`, outside the repository.

```bash
python tools/run_test_suite.py --tier core
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_module_child_storage.py \
  compat/tests/torch/test_module_custom_state.py \
  compat/tests/torch/test_torch_module_registration_order.py \
  compat/tests/torch/test_backward_final_callbacks.py \
  compat/tests/torch/test_deepspeed_public_dependencies.py
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

The official CPU core gate passed **322**, skipped **42**, and retained **1** existing
xfail. The final targeted CPU/CUDA owner gate passed **40**, skipped **0**, including
the existing custom-state/`_apply` behavior. The first full structure run found
three pooling pickle field regressions; their owner was corrected, and all **12**
pool structure cases then passed. The repeated full structure scan passed
**1441**, skipped **6**, and failed **0** before this report was added; registering
the new report in shared documentation indices is the integrating owner's step.
After the final adoption change, the independent frontend and NN compatibility
CPU/CUDA regressions also passed **36**, skipped **0**.

A separate, still-unreleased DeepSpeed adapter/Engine gate experiment used the
clean upstream `v0.19.7` checkout
`f5af15c2b4da10411e340850c00ba0cb5b14fd87` from
<https://github.com/deepspeedai/DeepSpeed>. It completed genuine Engine
forward/`backward`/`step`, six microsteps, gradient accumulation into three AdamW
updates, and save/load into a fresh Engine followed by resumed training. Both
runtimes used the same source-built DeepSpeed wheel. All eight parameter gradients
and both input gradients were requested; 256 output/gradient/parameter/optimizer
arrays matched the independent oracle. Maximum deviation normalized by each
reference magnitude and a common gradient floor was `3.37e-5` on CPU and
`1.26e-5` on CUDA. The Jittor runs synchronized under an error policy with zero
backend-fallback attempts. This is bounded **FP32, world-size-one, ZeRO-disabled**
evidence, and does not establish multi-rank NCCL, ZeRO, bf16, fused extensions,
DeepCompile, or elastic launch support. The optional import adapter and its own
admission gate are not included in this prerequisite commit.

At this prerequisite baseline, the negative leaf-hook reproduction in
`cpu-owner-target.log` showed native `register_hook` on a leaf did not fire,
while independent Torch did; final callbacks were tested via intermediate/loss
hooks. The subsequent native owner fix and actual ZeRO 1/2 Engine acceptance are
recorded in [the bounded FP32 Engine report](2026-10-15-deepspeed-zero-engine.md).
The original reproduction remains available, and single-rank FP32 acceptance
still does not complete multi-rank bf16 R14.
