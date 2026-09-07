# Independent Tensor and Module installation

- Status: explicit independent installation owns Tensor and NN types; 7.12 open
- Baseline: `7c9110da4`
- Date: 2026-09-08
- Owner: coord
- Review when: installers, module construction or frontend policy scopes change

Independent activation now installs methods on a real Tensor subclass of
native Var. Constructors, typed factory spellings and eye return that type;
native input checks still accept native Vars. Descriptor and numeric-slot
lookup follows the MRO rather than assuming descriptors live directly in the
frontend class dictionary. Native Var is not patched by independent activation.

The NN tree has separate writable modules, a real Module subclass and layer
adapters inheriting the native layer implementation and frontend Module.
Constructors and forward calls enter the tensor frontend scope. Internal native
child wrappers and mutable containers are copied with alias-preserving memo
tables; caller-supplied modules are retained without changing their classes.
The native layer mathematics and tensor graph remain shared. NN initializers
and pooling publication write only to the target module tree. Linear/Embedding
read default dtype from the target API instead of assuming it exists on native
Jittor. Constructor-owned parameters are registered with the frontend gradient
state; Embedding preserves its explicit freeze option.

Independent activation no longer permanently selects the explicit-requires-grad
policy. Python frontend scopes and generated native binding scopes apply the
type's policy and restore the caller's policy on exit. No-input factories and
gradient callbacks follow the same rule. Tensor/ones-style factory defaults
explicitly disable gradients; detach uses its frontend owner after the native
call scope has exited. Legacy activation retains its existing behavior.

Actual installation failure/retry exposed stale module publication: replaying
required steps after rollback republished the failed graph first, so the new
torch.types object collided with the abandoned object. Retry now reconstructs
the graph, publication bookkeeping rolls back too, and NN adapters are rebuilt
when the retry creates a new Tensor type.

Verification: CPU state/bootstrap/transaction/context tests plus the new real
independent integration case: 109 passed in 4.34 s, no skips. The integration
case injects a required-step failure after installation, retries, checks native
Var/Module/Linear dictionaries retain identical bindings, verifies independent
NN/init identities, Tensor construction/dtype/backward/data writes, default
gradient behavior, detach, and Linear/Sequential parameter gradients. The same
integration case on real CUDA 12.2/sm89 passed in 1.77 s after the incremental
build; it asserts CUDA availability/use and the tensor's device residency.
CPU and CUDA used separate caches. No full suite, new wheel or NPU run.

Remaining: independent mode is still opt-in; deployment defaults still use the
legacy alias. Parameter retains the existing semantic-role implementation and
does not yet have Torch's actual Tensor subclass hierarchy. Opaque user-supplied
native child modules are not silently converted, and shared native child
parameters are not forcibly retyped or given new gradient flags. Broader API,
serialization/subclass and mixed-thread behavior still need verification;
native backend state is not generally per-thread merely because a frontend
scope restores it. This result does not close 7.12.
