# Torch Namespace Local Ownership

- Status: Namespace/context boundary implemented; 7.12 remains open
- Baseline: `fdf344c92`
- Date: 2026-09-08
- Owner: coord
- Review when: installer targets, namespace publication or rollback changes

TorchNamespace writes now create local bindings. Deletion hides a native
fallback without deleting it from the backend owner. Immutable hidden-name
snapshots let transactions restore exact local ownership; rollback no longer
leaves a shadow attribute or an extra deletion marker. Foreign changes are
checked before transaction restoration. Ordinary Flags descriptors retain their
existing transaction behavior.

InstallContext distinguishes its target_namespace from native_backend while
preserving its existing constructor and jittor_module spelling. Context,
installation markers and completion checks read the target's own dictionary,
so a namespace does not inherit a native owner's completed installation.
Explicit registries remain supported. Core installation flags use the explicit
native backend instead of the installer's module-global Jittor reference.

Real opt-in activation exposed an existing publication inconsistency:
torch.torch changed in the registry but not the import mapping. Both mappings
now change and roll back together. This allows repeated compatibility install
after independent namespace activation without falsely reporting a changed
module graph.

Verification: 90 namespace/context/transaction tests passed in 1.94 seconds;
the strengthened self-alias rollback node also passed. The flag-rollback test
now exercises its declared backend fixture rather than trying to enable CUDA
on a CPU-only runtime. Real CPU and CUDA processes activated
independent_namespace=True, confirmed local public writes did not alter Jittor,
computed square backward with gradient [2, 4], and repeated installation
successfully. CPU used nvcc_path="", CUDA used nvcc 12.2/sm89 on an independent
cache. No full repository gate or wheel rebuild was run for this boundary change.

Default activation and installer families still operate on the native owner;
native Var and NN classes are still patched. These changes provide the explicit
boundary for moving installation before publication, but do not establish a
fully independent Torch Tensor/Module implementation.
