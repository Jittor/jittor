# Tensor backend placement

- Status: Implemented; focused CPU and two-device CUDA validation passed
- Reviewed: 2026-09-08
- Owner: native Runtime and independent frontend maintainers
- Review when: graph construction, fusion, allocation, device copies or frontend factories change

Independent Tensor placement is a native graph constraint. The Var/Op graph and
seven meta-operators remain shared with native Jittor; there is no separate Torch
executor. A Python `force_cpu` attribute cannot establish this constraint.

## Placement and residency

`Var::placement` has two states. `FollowRuntime` is the default for native Var
construction and retains native Jittor's existing backend policy. An explicit
placement contains `BackendId` and a device index. It survives host staging,
materialization and changes to the Runtime's default backend. `Var::device_id`
continues to hold native accelerator affinity; for explicitly placed CPU Vars it
is `-1`. The allocation remains the authority for physical residency.

`TensorPlacementScope` changes only thread-local graph-construction metadata. It
does not invoke a Runtime flag setter, synchronize, allocate or execute a kernel.
The Python binding's placement ContextVar carries an explicit factory device;
the existing frontend binding scope transfers it into native construction state
for each C++ call and restores it on exit. Without a factory override, a frontend
input's placement takes precedence over the Runtime default.

`Op::propagate_device` adopts explicit input placement for its outputs. Conflicting
explicit tensor backends/devices require an explicit copy. Existing bounded
pending-scalar retargeting remains available for unpublished temporary operands.
Python publication marks the native Var with `_placement_published`; assignment,
update and swap retain this fact independently of any single holder back-pointer.
Published Tensor placement cannot be changed by scalar retargeting.

Binary/ternary constructors permit a CPU 0-D tensor operand alongside an accelerator
operand. Generated constructor adapters insert a differentiable DeviceCopy before
the typed constructor creates members, broadcasting subgraphs or Node edges.
The source remains on CPU and its gradient copies back to CPU. Qualification uses
the actual zero-dimensional shape; the original `_is_scalar` dtype-promotion bit
is preserved, not manufactured. An explicit accelerator scalar with a CPU vector
is rejected. Native graphs
without explicit placement continue to use the original Runtime selection.

## Selection and execution

The native dispatch query and construction-time capability selection consult
placement before the Runtime default. This includes Random, Array, Where,
Transpose, Argsort, ArgReduce and reduction dtype policy; selecting a CUDA-only
replacement before graph propagation would otherwise defeat explicit CPU placement.

Fusion cannot merge different requested backends. Each segment's compiler,
graph-optimization and runner scopes use its graph placement. FusedOp compiler
copies query their constituent Ops, because those copies intentionally omit Node
edges. A plan's default backend remains relevant only to unplaced native nodes.

The runner selects a backend/device allocator per segment. Explicit CPU outputs
are never uploaded by the legacy CPU-fallback tail. Genuine accelerator fallbacks
keep the existing fallback policy and managed-memory behavior. Device-copy Ops
own their transfer and do not migrate their source in the runner. Their gradient
copies back to the input's explicit placement.

`BackendId` permits the same CPU/accelerator split for ACL, ROCm and Corex builds.
The existing runtime still selects one accelerator family per build/process;
this change does not add cross-accelerator-family copies or claim missing hardware
validation.

## Frontend and verification

`tensor_frontend(type, device=..., like=...)` scopes constructors without changing
the Runtime default. Factories, Tensor/new_* constructors, clone/deepcopy/pickle,
`to`/`cpu`/`cuda`, and serialization map-location paths use this boundary. Device
reporting and data-owner routing read native placement. The fftfreq/rfftfreq facade
consumes its device argument before calling the shared native frequency mathematics.
The independent Parameter uses stable module-level `parameter_new`/`parameter_init`
methods; its type factory only binds the native backend and Tensor base. The optional
factory adapter is an explicit callable/descriptor, not an installation closure.

The focused gate demonstrates lazy CPU sources, scalar/binary/view/reduction
descendants and loaded CPU tensors staying on CPU with the global CUDA policy
enabled; CPU/CUDA segments in one submission; copies across two CUDA devices and
back to CPU with gradients; incompatible input rejection; and unchanged native
FollowRuntime behavior. The original CPU-checkpoint placement xfail is removed.
See the [validation record](../results/2026-09-08-tensor-backend-placement.md).

Host scope nesting/thread-isolation and translation-unit syntax checks are useful
preconditions, not evidence that tensors execute on their requested hardware.
