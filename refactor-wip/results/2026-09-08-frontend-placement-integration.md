# Frontend types, placement and lifecycle contracts

- Status: Implemented; focused native CPU/CUDA validation passed; full structure gate remains red
- Baseline: `b2e03162b`
- Reviewed: 2026-09-08
- Owner: native Runtime and compatibility maintainers
- Recheck when: frontend binding, scalar conversion, graph dispatch or provider entry points change

## Type implementation owners

NN and distribution factories now configure their per-installation types and
bind explicit module-level behavior. `NNFrontendOwner`/`LayerInitializer` own
native layer adaptation, `nn_adoption.py` preserves external children and
promotes only newly owned parameters, and parameter-container mixins own their
methods. Distribution method/constructor descriptors own argument adaptation,
sampling policy and native delegation. Generated classes remain separate for
each frontend without modifying native types or introducing another Var/Op graph.

NN's five focused owner tests passed. Its source overlay into the existing CPU
core passed the original failed-install/retry/native-preservation node in
1.44 s. After integrating NN, the distribution overlay passed four nodes in
6.81 s, covering scalar parameter conversion, sampling gradients, property and
pickle identity, owner separation and untouched native classes. Raw distribution
logs are unversioned under `$JITTOR_LAB_ROOT/_state/distribution-owners-nmAxx0/`.

The final API manifest previously named `torch.nn.Dropout3d`, which is not
published by this implementation. That unsupported name has been removed from
the installed-object manifest, without inventing an implementation. The manifest
test now requires every declared namespace/name to exist, so a missing declaration
cannot silently pass through the previous `continue`/non-callable checks.
This inventory concerns installed APIs, not every API in upstream PyTorch.
The typing-only `torch.types.Storage` class and its unimplemented operations
also have explicit records. The final manifest binding/existence node passed
in 0.04 s on the warm CUDA core after these metadata changes.

## Installer, backend and allocator contracts

`Installer` gives the 24 required and six optional composition steps the same
positional-context signature. Callback validation runs before invocation and
uses existing required/optional reporting for `EXPECTED` errors. Unexpected
exceptions still propagate to outer rollback. Python build providers share
`Backend.configure/ install_extern` and optional `BackendPostProcess` contracts.
Previously a configure-only provider was accepted and would fail much later at
the unconditional `install_extern` call. Discovery now rejects missing or
malformed hooks without executing them. Builtin CPU/CUDA selection still avoids
loading optional SDK providers; ACL/ROCm/Corex declare the common context/value
types. These changes do not claim hardware validation for those providers.

Protocol/provider checks passed 20 tests in 0.19 s. Combined with the existing
ACL, legacy backend, ROCm and NN owner contracts, 48 passed in 1.13 s. The two
protocol files also passed focused mypy. Full configured mypy initially exposed
38 errors caused by the earlier physical relocation of the build utilities:
the `jittor_utils` path bootstrap had no static exports. Its new `.pyi` explicitly
re-exports actual implementation members and submodules, with no permissive
`Any`/`__getattr__` fallback or relaxed checks. Full mypy then passed all 38
configured source files. The stub is declared as a core package resource.

`allocator.h` now documents instance-local tokens, release tuples, physical
residency, shared-view ownership and cache-collection semantics at the interface.
This is a comment-only addition; allocation algorithms are unchanged by the
contract documentation. See [lifecycle contracts](../architecture/installer-backend-contracts.md).

## Native placement acceptance

The final isolated integration run passed all nine nodes, with zero skips, in
343.12 s: seven placement cases, the original CPU checkpoint regression and the
full independent-frontend/native-preservation child. CPU and two CUDA devices
execute through the same native graph and built core. The checkpoint xfail is
removed; `KI-BACKEND-PLACEMENT-001` is resolved for the independent frontend.
CPU 0-D operands receive a local differentiable copy when used with accelerator
operands; published tensors are never retargeted. CUDA-scalar/CPU-vector mismatch
is rejected without changing the source. CPU reduction-produced scalars and
frequency factories with explicit devices are included. See the
[detailed native report](2026-09-08-tensor-backend-placement.md) for commands,
earlier failures, source pinning and hardware limits.

The required complete structure run on the warm CUDA tree, using legacy Torch
mode, reported 23 failed / 1293 passed / two skipped in 415.54 s. Two stale
contracts were corrected and passed together in 0.06 s: reduction policy now
reads the input's construction target rather than global use_cuda; the original
device-index error migration is counted separately from the new scalar-placement
boundary. The remaining failures include old source-count inventories, process
mode/child-environment contracts, runtime composition, repeat/allocator behavior
and the unchanged FSDP reflection baseline. Several CPU-only child probes inherit
JT_BACKEND=cuda while clearing their compiler path and therefore reject the
inconsistent environment. This is not a green full structure run, and the entire
remaining set is not claimed to be independently baseline-certified. Raw output:
`$JITTOR_LAB_ROOT/_state/tensor-placement-integrated-20260908/structure-final.log`.
The coordinator compared the integrated native and compatibility implementation
files against the validated worktree. The only remaining source difference was
the 24-line comment-only allocator contract; executable code matched.

This batch closes 7.03's implementation-owner migration and 11.04's interface
contracts. It does not claim full upstream Torch API equivalence: per-API fidelity
still records approximate/unimplemented behavior. 7.12 remains open because
`compat/shim/runtime.py::_installation_target(..., independent=False)` still
returns the native owner and the explicit legacy path can install onto native
types. Removing that path and updating its callers is a separate remaining
architecture step. Native-only FollowRuntime behavior itself is intentional and
remains supported.
