# One Torch frontend, native Jittor preserved

- Status: Implemented; focused activation and CPU/CUDA composition verified
- Baseline: `c618d841d`
- Date: 2026-09-08
- Owner: compatibility/runtime maintainers
- Review when: an activation entry, installation target, alias callback or deployment stub changes

The native-as-Torch installation path is removed. Import-time composition,
explicit `activate()` and the deployed Torch entry all select an independent
Torch namespace. `install(jittor)` is rejected before creating a context;
`activate(independent_namespace=False)` and old false
`JITTOR_TORCH_INDEPENDENT` settings produce explicit migration errors. The
historical keyword provides that error, not a second implementation. Activation
no longer writes a frontend-mode environment flag or switches native autograd
policy to implement Torch semantics.

The historical `jittor.torch_compat` import callback now invokes independent
activation; it no longer calls `install(root)`. Native bootstrap reads only the
explicit shim/project/runtime environment requests. A Torch module's private
placeholder attribute is no longer a native activation trigger. The deployed
entry sets the explicit shim request before importing Jittor and rejects an old
false mode setting before changing its marker or environment.

Native-target branches were removed from core/Tensor, Function/autograd, NN/init,
distribution, optimizer and numerical publication. The legacy native Function
patch helper, plain-tensor parameter-registration setter and dtype cast-converter
table are removed. `torch.dtype` objects are not callable; native `jt.float32`
remains a native cast function. None of this changes native Var FollowRuntime,
TensorPlacement, storage/dtype conversion or the shared Var/Op graph.

## Callers and migration

Use `import torch` for the compatibility API. Keep a separate `import jittor as jt`
when a test or tool needs native flags, compiler inspection or synchronization.
Replace `torch.array` with `torch.tensor`, and dtype-function casts with
`tensor.to(dtype=...)`. Remove old false mode settings/arguments and update an
already deployed Torch stub when updating an environment.

The 56 compatibility test modules with explicit `import jittor as torch` statements
were migrated to the real Torch entry. Native instrumentation imports remain
explicit. Torch-grade NN/RNN fixtures now import their NN surface from Torch;
the native checkpoint/oracle test deliberately retains native NN. Installer
idempotency/failure fixtures use independent targets. Their rollback check compares
the actual publication ledger instead of the obsolete 199-module constant.
Historical result reports and approved content hashes were not rewritten.

## Evidence and limits

Focused runtime verification: **16 passed, 0 skipped, 49.41 s** using the previous
TensorPlacement CUDA core/cache and a source overlay of the changed compatibility
Python. The run deliberately unset `JITTOR_TORCH_INDEPENDENT`; `JITTOR_TORCH_SHIM=1`
alone produced independent types. It covered old argument/environment rejection,
direct native-target rejection, repeated activation, required-install failures,
native Function preservation, custom Function contexts, and the full existing
independent-frontend subprocess with failure/retry, training, dtype, storage,
deepcopy and pickle checks. The new storage/gradient node executed both CPU and
CUDA and checked real residency. No new native core build was started.

The actual installer/parent rollback pure-Python fixture passed separately:
**1 passed, 0.02 s**. After finding the remaining historical-alias callback and
native placeholder trigger, two additional pure-host checks executed those actual
entry functions: **2 passed, 0.01 s**. The former forbids calling `module.install`
on the native owner and verifies independent activation; the latter proves a
private Torch marker does not request compatibility without explicit environment.
Unreachable native-target branch cleanup after the runtime run preserves the
already exercised independent bodies. Final source/test syntax and diff checks
passed. A test regex warning from the runtime run was corrected to a raw string.

The 56 migrated test files were not all executed as full suites. No claim is made
that the full repository gate is green, or that this Python-only removal adds
NPU/ROCm/Corex hardware validation. The separate vLLM adapter extraction and test
directory reorganization are coordinated by their owners.

Unversioned runner, subprocess source pinning and runtime log:
`$JITTOR_LAB_ROOT/_state/remove-legacy-frontend-20260908/{run.py,bootstrap/sitecustomize.py,focused.log}`.
The core is the already verified `refactor-tensor-placement-integrated-20260908`
tree with its private cache; the compatibility overlay is this batch's source.
Git integration and any final renamed test paths belong to the coordinator.
