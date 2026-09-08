# Runtime hook transaction ownership

Installation and delayed hooks have different lifetimes. `InstallTransaction`
records an installation until commit. `RuntimeHook` records one delayed
activation and retains its undo ledger while active. Both use the same
reentrant process lock; module-patch and source-resolution locks are acquired
after that lock.

`runtime_hook(owner)` / `owned_runtime_hook(owner)` establish the current
recording scope without importing Jittor. `current_transaction()` is likewise
bootstrap-free. Attribute writes through `set_attr` or `patch_method` and module
publication through `RuntimeHook.replace_module` belong to that scope. Class
descriptors are recorded from their local dictionaries, preserving staticmethod
and inherited-attribute behavior. Cleanup checks the current object's identity;
foreign replacements are preserved and reported as `TransactionConflict`.

`release_runtime_hooks(owner)` releases active effects in reverse order.
Conflicts do not prevent independent earlier entries from being restored.
Delayed module patch callbacks have owners of the form
`("module_patch", module_name, callback)` and can be released by
`release_module_patch_hooks`. Callback authors must use the mutation helpers for
class or mutable registry changes; arbitrary third-party Python side effects
cannot be inferred or reversed from a module dictionary snapshot.

vLLM activation owns its extension stand-ins, API-version selection, registered
operators and installed marker. Flash-attention publication has a nested owner
and an independently releasable lifetime. Existing foreign module slots or
operator names are not overwritten. The C++ extension facade publishes before
executing a module so recursive imports see the same object; a failed loader
restores the prior owned slot, and a foreign replacement causes a hard failure.
Build products are not deleted by module rollback.

Source-root resolution uses identity-tagged path insertions and a temporary,
current-thread import loader tracker. Rollback touches only recorded source
publications and displaced entries. Concurrent path changes and unrelated
imports remain intact. Custom source loaders must publish through
`publish_source_module`; an untracked publication under the source root is
preserved and causes a hard conflict rather than being guessed to be ours.
Loader wrappers detach after execution so imported modules do not retain a
resolver's entry snapshot. The public resolver entry is `load()`; helpers which
would modify import state require its active scope.

`torch.install(target, parent_transaction=activation)` transfers the successful
child ledger with `activation.adopt(child)` before child commit. A later outer
failure can therefore restore classes, modules, completion markers and cached
tensor state. The child releases its lock/active handle normally. Retrying the
outer operation rebuilds installation steps instead of trusting reverted
completion markers. Fidelity registration records individual API records and
implementation attributes through the same current scope.

The standard-library tests in
`tests/compat/torch/test_runtime_hook_ownership.py` exercise actual hook,
extension-publication and source-loader code, plus the actual install function
against controlled state owners. They require neither Jittor compilation nor
accelerator hardware. Integrated activation/context tests remain the control
for the complete native frontend.
