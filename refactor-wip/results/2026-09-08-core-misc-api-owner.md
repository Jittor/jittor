# Core misc APIs: stable module ownership

- Status: this owner cohort implemented and CPU-verified; not completion of all task 7.03.
- Reviewed: 2026-09-08.
- Owner: Torch compatibility maintainers.
- Baseline: `7e83d6da4`, with the coordinator's Runtime/context accessor used for integration verification.
- Recheck when: misc API bindings, installation context or RNG/default-policy ownership changes.

`compat/torch/installers/core.py` now owns module-level objects for storage
metadata classes, random/RNG helpers, norm/where/bincount/segment_reduce,
autocast/grad/determinism queries, finfo/iinfo, promotion and default dtype/device.
`install_misc` binds those objects and initializes installation-owned state;
its nested def/class/lambda count is zero (previously 47). Public functions and
storage aliases have fidelity records and stable pickle identities.

Runtime functions obtain the current owner via `get_install_context(jittor)`.
No mutable module-level context pointer was introduced. Rebinding preserves the
random module identity and the owner's configured default dtype. Native random
delegation is saved once in that context, avoiding chains of callable modules.
The independent `install(ctx)` function was kept byte-identical; its transaction
and version hook remain owned by the coordinator's other partition.

Existing mathematical bodies and tables were preserved: 24 function/class ASTs
match after removing the added owner lookups and normalizing doc indentation.
The result_type helper moved out of its enclosing function without changing its
promotion logic. Seed-only RNG state, metadata-only storages and ignored norm /
where out parameters remain conservatively approximate. Existing no-op
autocast/determinism setters and PyTorchFileReader are explicitly marked
unimplemented rather than described as supported controls.

One successful CPU batch: `tests/compat/torch/test_core_misc_owner.py`, **5 passed**
in 9.62 seconds, covering all root binding identities/fidelity/pickle paths,
storage/random aliases, independent NumPy values, promotion/limits, RNG/default
dtype state and repeat binding. AST/symbol checks also passed. No GPU or
performance validation is claimed for this ownership-only change.

The integration runner overlaid only this core module onto the coordinator's
existing CPU runtime/context and selected the current source Torch entrypoint.
The inherited system-site Torch entry was legacy-only, so it was not used as
evidence for independent activation. Earlier attempts exposed two coordinator
CUDA-package import errors; those were fixed by that owner before the passing
batch. No context implementation or test finder is duplicated in the delivery.
Raw log and temporary runner are unversioned under `$JITTOR_LAB_ROOT/_state/`
as `core-misc-owner-check.log` and `core-misc-owner-check.py`.
