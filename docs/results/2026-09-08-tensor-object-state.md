# Per-Tensor state and original-plan alignment

- Status: first object-state fields migrated; 7.12 remains open
- Baseline: `4bf74c8d0`
- Date: 2026-09-08
- Owner: coord
- Review when: tensor metadata, rollback or independent packaging changes

The original plan requires per-Tensor state and a physically separate compat
distribution. The existing class named TorchTensorState is installation-level
leaf/retained/optimizer bookkeeping, not that per-object state. The handoff now
distinguishes those requirements and records the remaining native-marker and
packaging boundaries instead of replacing them with generic API cleanup.

TensorObjectState now owns grad, data owner, data path and the legacy scalar
marker for each independent Tensor. Slots make those fields explicit. Reads do
not allocate; the first write creates one owner. Compatibility descriptors let
existing getter/setter consumers use it, and old pickle fields are adopted and
removed on first write rather than retained as duplicate references. Native Var
is not patched. Native graph/shape facts remain in the core rather than copied
into this Python state.

Native indexing no longer sets or reads _torch_0d or materializes marked scalar
indices through Python item(). The real zero-dimensional index path handles
them. Module/legacy-Parameter Torch markers remain and need their own protocol
migration; this is not a claim that the outside-compat grep is already zero.

Verification: 15 focused CPU cases passed in 3.37 s and 3 CUDA cases in 11.55 s.
The integration checks lazy state allocation, one owner/no duplicate grad/data
attributes, deepcopy and pickle self-cycles, and collection through weakrefs.
Scalar Tensor indexing forward/backward passes after marker removal. Existing
indexing cases report retained Vars at session end; no full liveness audit,
full suite, new wheel or NPU execution was performed.

Remaining architecture work: other per-object fields and runtime registries,
native Module/Parameter marker dependencies, and moving compat to the top-level
independent distribution. The handoff contains the concrete bootstrap/package
ownership sequence for the next migration batch.
