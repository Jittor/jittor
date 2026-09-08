# Torch API and runtime state ownership

- Status: Accepted
- Last reviewed: 2026-09-08
- Owner: compatibility maintainers
- Review when: a new API family, frontend owner, or activation side effect is added

Torch compatibility keeps the native Var/Op graph. The independent frontend
owns its Tensor, Module and installation state; it does not create a second
execution graph. Native mathematical implementations remain the source of
truth where an API can delegate to them.

## API implementations and installation

An installer publishes existing module-level functions and classes. Put the
actual implementation in its family module, with an importable identity;
changing a nested function's `__qualname__` does not satisfy this boundary.
Per-call algorithm callbacks may close over inputs, but installation must not
create a new public implementation on each attempt.

Use `get_install_context(native_backend)` to resolve the explicitly bound
frontend owner at call time. This is a read operation: it neither activates
Torch nor creates an installation. It rejects an absent or inconsistent owner.
Code being installed already has its `InstallContext` and should use it
directly. Keep captured native implementations in context-owned state before
publishing wrappers, so later lookup cannot call the wrapper recursively.

`compat/torch/installers/cuda/api.py` owns the CUDA facade implementations;
`cuda/bindings.py` publishes those objects. Its `CudaRuntimeState` belongs to
`InstallContext.state`, including logical streams, NVTX state, memory-query
caches and the matmul precision refinement. Utility pytree definitions also
belong to the installation. API identity remains stable while mutable state
belongs to the active frontend.

Register the final published object with `register_fidelity`. `approximate`
means limitations remain; `unimplemented` also covers existing annotation or
no-op placeholders, not just functions that raise. A stable Python object does
not establish numerical equivalence or working hardware support. CUDA logical
streams are serialized, event timing uses synchronized host timestamps, and
memory peaks are sampled at queries. These are recorded limitations.

The independent root namespace permits native fallback only during bootstrap.
After publication it is sealed: an undeclared name raises `AttributeError`;
deleting a local name cannot reveal a native implementation. `native_api.py`
declares stable `NativeOperation` objects for shared mathematical operations.
They resolve a read-only, transaction-owned delegate table and enter the
frontend result-type/autograd scope. Native-only `flags`, `core` and `runtime`
are not public Torch attributes. Internal services resolve the explicit owner.
FFT/linalg facades copy public exports, excluding imported implementation
modules, and retain their own namespace identity.

`api_manifest.py` records the final composed API paths after family installation.
Family-specific restrictions take precedence over generic composition metadata.
`fidelity_report()` returns sorted immutable records; `fidelity_table()` renders
the coverage table. The registry is authoritative per API spelling: aliases can
share an object while declaring different supported behavior. Metadata must not
mutate shared native Python classes/functions. Neither coverage registration nor
namespace sealing proves full Torch semantics; native placement is still tracked
under 7.12 and `KI-BACKEND-PLACEMENT-001`.

Serialization owners separate portable values, restricted pickle loading, Torch
archives and safetensors. Restricted mode rejects native-only fallback paths that
cannot enforce its unpickler policy. Autograd/library owners retain one native
graph and register detached custom-op results with their explicit backward.
See [serialization](torch-serialization-owners.md) and
[autograd/library](torch-autograd-library-owners.md) for the detailed boundaries.

## Runtime services and temporary scopes

`jt.runtime.service_state(namespace, factory=...)` owns extension state without
adding private attributes to `jittor`. A read without a factory does not create
state. A successful factory runs once per Runtime; a failing factory can be
retried, and recursive construction of the same service is rejected. The
service controls synchronization and transaction semantics for its own fields.

Shim activation uses the `jittor.torch.activation` service. The installation
lock covers the activation status check and state transition as well as the
installation itself. Concurrent callers wait; reentrant activation in the same
thread is rejected. Rollback conflicts must release the lock and leave a
queryable failed state.

Tensor bookkeeping uses `jittor.torch.tensor_states`, a Runtime-owned weak
owner table. `get_tensor_state()` resolves an explicit frontend binding;
`latest_optimizer()` resolves weak optimizer references. This applies to both
independent and legacy activation. Historical leaf, retained and optimizer
root aliases are adopted once and removed, with installation rollback restoring
their previous ownership if installation fails.

The vmap getitem lowering hint is context-local and owner-scoped. Use
`TransformGetItemToIndex(owner)` and `getitem_transform_active(owner)`, not a
module attribute. Nested uses of the same scope and exception exits restore
the entry state; another execution context does not inherit active mutations.

Legacy native-as-Torch activation still mutates native API types outside these
state boundaries. Its remaining removal is tracked in 7.12; this contract does
not declare the entire independent frontend migration complete.

## Distribution boundaries

Core, compatibility and optional third-party adapter distributions each declare
their own packaging configuration in `pyproject.toml`. The adapter source lives
under `adapters/`; install it when Transformers/TorchMetrics patches are needed.
The compatibility runtime uses entry points and reports missing adapters without
claiming they were applied. After adding source or documentation, run
`python tools/build/generate_manifest.py`; `--check` verifies the generated
manifests. Building either distribution must not import Jittor or compile the
native runtime. See [packaging evidence](../results/2026-09-08-packaging-ownership.md).
