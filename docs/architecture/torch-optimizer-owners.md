# Optimizer and scheduler API owners

Status: 7.03 family migration, 2026-09-08. Owner: compatibility maintainers.
Recheck when a native optimizer initializer, step or state layout changes.

`compat/torch/lr_scheduler.py` owns the scheduler classes, LR-group helpers,
SWA/EMA averaging callables and BatchNorm refresh implementation at module
scope. Its installer creates namespace modules and binds those same objects.
Scheduler instances own their epoch, group rates and trajectory state. SWA's
averaging function is a stable function; a requested EMA decay is stored on a
module-owned callable object rather than in an anonymous closure.

`compat/torch/optimizer_api.py` owns state views, serialization, gradient
management, closure handling and update adaptation. `optimizers.py` binds these
objects to the installation's optimizer classes. SGD, RMSprop and Adan delegate
to their captured native steps. Adam and AdamW call the existing shared native
`adam_update`; this migration does not create a second mathematical update rule.
FSDP dispatch still uses the registered provider and the actual class/step
identity recognized by `optimizer_kinds`.

`optim_frontend.py` creates the installation-specific optimizer subclasses with
stable module-level initializer implementations. Native initialization runs
inside the existing Tensor frontend scope so algorithm buffers keep their
frontend type. Native optimizer class dictionaries remain untouched in
independent mode. LBFGS remains explicitly unsupported.

`InstallContext.state["optimizer_native_api"]` is a read-only mapping of the
original base methods, native step functions and per-algorithm initializers.
`optimizer_frontend_native` similarly owns the frontend Tensor type and original
native constructors. Runtime methods query the existing context with
`get_install_context()`; they do not create an installation or retain a separate
global context. Capture occurs before rebinding so wrappers cannot delegate
back to themselves.

The existing scheduler formulas and limitations are preserved. In particular,
the migration does not establish full scheduler resume semantics, fill ignored
optional arguments, change AveragedModel's native Module base or implement its
missing buffer-averaging behavior. Installed API records are conservative
`approximate`, with `LBFGS.step` recorded as `unimplemented`. They appear in the
shared fidelity report and generated coverage table.

Focused tests in `test_optimizer_scheduler_api_owners.py` cover import/pickle
identity, absence of install-time implementations, two-step SGD/Adam/AdamW
updates against NumPy formulas, StepLR progression, state restoration, gradient
retention and SWA/EMA averaging. Existing closure and native-backward tests
guard the optimizer step-counter boundary. GPU validation is a separate shared
integration run; no performance claim is made here.
