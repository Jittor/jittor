# Native storage layout contract

The unified Var/Op graph and the seven meta-operators remain the execution and automatic
differentiation model. **The storage layout is native metadata; the frontend keeps no second
stride table.**

## Representation and ownership

`Var::shape` and `num` describe **logical elements**, and `size` is still the logical byte
count. `storage_stride(axis)` returns the stride in **elements**; `storage_strides` is empty
for canonical dense storage and is stated explicitly for other layouts (including the
stride-0 case). `storage_offset_bytes` locates the first logical element relative to the
start of the allocation, and `mem_ptr` already includes that offset.

Allocation identity and lifetime remain `(allocator, allocation)` plus the shared allocation
ring. `share_with()` records a pending alias; allocation establishes the ring and holds a
reference to the other allocation.

**Physical allocation, migration, swapping out and freeing all use `storage_span_bytes()`**;
never copy the logical byte count of an expanded tensor onto its smaller physical storage.
Migration may compact the live storage span and update the offset, **provided the relative
address of every live alias stays unchanged**.

The initial native descriptor supports **non-negative** element strides. Basic slicing with a
positive stride and expand share physical storage; negative-stride selection and advanced
indexing remain computational gather operations. A compatible `view()` reshape only rearranges
the stride block without copying, an incompatible layout raises, and `reshape()` can explicitly
materialize its input as dense. `contiguous()` produces a visible `ContiguousOp` **only when
the input really needs one**.

## Kernel boundary

`Op::accepts_storage_strides` declares read support. The seven meta-operators consume the real
input strides; fused elementwise kernels receive those strides as **runtime scalar
parameters**, and their strided key variants stay cache-separate without baking the stride
values into the key. Index tensors themselves are canonicalized at the construction boundary,
because gather requires dense index storage.

The generated operator factory inserts explicit contiguous graph inputs for dense-only
implementations **before constructing concrete Var members or edges**. This conversion is
**differentiable and visible in the graph**, not an ad-hoc rewrite of existing Vars at
execution time.

Kernels that modify their input storage declare `mutates_storage_inputs`; **a non-contiguous
writable input is rejected** rather than redirecting the writes into a copy nobody notices.
Pre-allocated CodeOp output buffers likewise require contiguous storage.

`Op::is_storage_view()` marks **metadata-only** operations. The executor serves their alias
allocations directly and does not call the provider's compute kernel. **A backend overload must
not launch an Expand kernel into the smaller allocation behind a stride-0 alias.** The
metadata of clone/detach, reshape/view and basic slicing follows this boundary; they are still
graph nodes with normal gradient rules.

## Language and device boundary

`_storage_strides()`, `_storage_offset()`, `_storage_is_contiguous()` and the existing
`_storage_address` expose native facts to the frontend. Torch's stride and contiguity queries
delegate to them. The public basic-view record still governs holder assignment; assignment to
the source refreshes the active slice, reshape, expand and transpose records. **Writing through
mutually overlapping expand views is rejected.**

`numpy()` gathers by storage stride and returns a dense NumPy copy. `data` exports byte strides
with a base that holds the allocation; mutually overlapping stride-0 exports are read-only.
Scalar `item()` reads one physical element. Dense data assignment scatters into non-overlapping
positive-stride storage and **rejects overlapping destinations**. Reinterpreting by bytes
requires an explicitly contiguous source rather than silently changing its storage identity.

CUDA-generated meta kernels consume the native strides. Dense external-library calls use
explicit contiguous graph inputs. The ACL descriptor helper accepts the source Var and passes
its real strides and physical storage span to CANN; the native contiguous operation uses a
same-dtype Cast to write a dense output. The index-copy planner still supports dense only and
**reports that requirement explicitly**.

> Changes to ACL descriptors and kernels need **real CANN/NPU validation**; a host-side syntax
> check alone is not evidence of support for a device or dtype.

## Collaboration boundary

| Layer | Owns |
| --- | --- |
| Core | Var layout, allocation spans, alias establishment, view shape rules |
| Codegen | input layout adaptation, strided access |
| Provider | descriptor conversion, implementation-specific contiguity requirements |
| Frontend | querying native state, recording public view relationships |

**A new consumer should declare its layout requirements at registration time** instead of
hiding an implicit materialization inside the kernel, or rebuilding a stride table from shapes
in Python.

Targeted tests are in `tests/core/test_storage_strides.py` and
`tests/core/test_view_storage.py`, covering forced-expand alias identity, the arithmetic /
reduction / indexing that follows it, allocation budgets, assignment refresh, NumPy export,
positive-stride storage offsets, dense library boundaries and gradients.
