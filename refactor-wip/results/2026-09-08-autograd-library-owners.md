# Autograd/library owner migration

- Status: Implemented and focused CPU validation passed
- Baseline: `c68b0cf7d`
- Date: 2026-09-08
- Owner: compatibility maintainers
- Review when: native Function/tape semantics, registry transactions or API ownership changes

The autograd installer now binds real module-level implementations in
`compat/torch/autograd.py`; `library.py` owns its public Library and registration
functions instead of defining them during publication. The independent Function and
custom-op Function bridge retain the native Var/Op graph and per-call context.
Native delegates and the dispatch registry belong to InstallContext state.
The parent package's installer import was renamed to avoid overwriting the new
implementation module's attribute. See the [owner contract](../architecture/torch-autograd-library-owners.md).

All autograd installer entry points and library publication/schema-factory entry
points have zero nested function/class/lambda definitions. Call-time decorators
remain permitted. Fidelity/pickle checks cover the actual installed objects,
including metadata-only APIs; no `__qualname__` rewriting is used.

The first focused run exposed an existing independent-mode defect: a library kernel
returning a detached float tensor did not reach its registered backward. Repeating
that exact node with the baseline library implementation still failed. The boundary
now enables differentiable results before native output tape creation when a
registered backward and differentiable inputs exist. Backward mathematics are
unchanged; the regression checks cubic values `[8, 27]` and gradients `[12, 27]`.

Validation used the existing warm CPU core, source Torch entry and
`JITTOR_TORCH_INDEPENDENT=1`. A temporary import finder overlaid only this worktree's
autograd owner, installer, library and the four-line parent import adjustment onto
the coordinator's native package. No new core compilation or full-suite run was
requested. The six nodes in `tests/compat/torch/test_autograd_library_owners.py`
passed in **1.20 s**, with **0 skipped**:

- Stable public identity, fidelity and pickle resolution.
- Native graph gradients and two calls through one reusable Function instance.
- Actual CPU dispatch, detached custom-op forward, registered backward and schema inference.
- Failing RuntimeHook restores an existing operator and removes a newly created namespace.
- Rebinding retains dispatcher, operators and public classes.
- Installer ASTs contain no nested implementations.

The baseline-library detached-gradient node was **1 failed**; this is a targeted
baseline comparison, not evidence about an entire baseline suite. Initial migration
run was 4 passed / 2 failed before the module collision and gradient boundary fixes.
Raw runner and logs are unversioned under `$JITTOR_LAB_ROOT/_state/`:
`autograd-library-owner-check.py`, `autograd-library-owner-check.log` and
`autograd-library-baseline-gradient.log`.

CUDA/NPU were not run in this focused owner batch. No accelerator or performance
claim is made. Aggregate autograd option compatibility, placeholder APIs and the
full repository gate remain outside this result. Adapters using the new
`Library._transactional_registry` marker must suppress their old duplicate registry
snapshots; coordinator integration owns the vLLM caller adjustment.
