# Torch Installation Target

- Status: Independent-mode installation target migrated; 7.12 remains open
- Baseline: `8b486d567`
- Date: 2026-09-08
- Owner: coord
- Review when: activation targets, TensorState or installer retries change

Independent activation now creates its target before running the installers.
The target is retained for retry and subsequent runtime configuration. API
bindings, the installation context and the publication registry therefore
belong to that namespace from the start. Native backend references remain
explicit; legacy activation continues to use its original target.

An explicit transactional backend-to-owner binding makes all existing
get_tensor_state callers share the target's single state. Native private names
remain same-object compatibility aliases. Binding conflicts and missing bound
state raise; they do not create a second registry. Leaf/retained mappings and
optimizer entries are recorded individually for rollback, and retries reuse
the same state object. Rolled-back step markers now roll back too, so a retry
replays work that was undone instead of skipping it.

Implicit transaction lookup follows the bound owner. Ownership validation and
composition use the published target's local context. FSDP FlatParameter uses
the parameter implementation directly, signal operations resolve from_numpy
on the target, and complex linalg checks float32 by dtype meaning instead of
comparing it with a native conversion function. Tensor.data resolves the active
owner when deciding its compatibility behavior.

Verification: 108 existing/extended state, bootstrap, transaction and context
checks passed in 2.89 seconds. A strengthened target-before-install composition
check passed separately. The injected failure/retry case first reproduced a
stale completed-step marker, then passed after its rollback fix. Three
preflight fixtures were corrected to isolate canonical environment settings;
the product preflight was unchanged.

Real CPU and CUDA processes used independent_namespace=True and verified that
the native tensor binding did not change, new APIs were local to the target,
state aliases were identical, square backward was exact, data-view writes
preserved trainability, hann_window executed, complex matrix inverse matched
NumPy, FlatParameter was a trainable tensor, and repeated installation worked.
CPU used nvcc_path=""; CUDA used nvcc 12.2/sm89 with use_cuda=1 and a separate
JITTOR_HOME. Full repository gates and a new wheel were not run for this slice.

Native Var and NN classes are still shared and patched, and autograd policy is
still selected on the shared runtime. Thus this is not complete native/Torch
type isolation. Default legacy activation is unchanged; 7.12 remains unfinished.
