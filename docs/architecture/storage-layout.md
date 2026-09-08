# Native storage layout contract

The unified Var/Op graph and the seven meta-operators remain the execution and
autograd model. Storage layout is native metadata; frontends do not maintain a
second stride table.

## Representation and ownership

`Var::shape` and `num` describe logical elements. `size` remains logical bytes.
`storage_stride(axis)` returns element strides; `storage_strides` is empty for
canonical dense storage and explicitly represents other layouts, including
stride zero. `storage_offset_bytes` locates the first logical element relative
to the allocation origin. `mem_ptr` already includes that offset.

Allocation identity and lifetime remain `(allocator, allocation)` and the shared
allocation ring. `share_with()` records a pending alias; allocation establishes
the ring and owns another allocation reference. Physical allocation, migration,
swap and release use `storage_span_bytes()`. They must never copy an expanded
tensor's logical byte count from its smaller physical storage. A migration may
compact the live storage span and update offsets while preserving every live
alias's relative addresses.

The initial native descriptor supports nonnegative element strides. Positive
basic slices and expand share physical storage. Negative-step selections and
advanced indexing remain computing gather operations. Compatible `view()`
reshapes regroup stride chunks without copying; incompatible layouts raise.
`reshape()` may explicitly materialize a dense input. `contiguous()` creates a
visible `ContiguousOp` only when the input needs one.

## Kernel boundary

`Op::accepts_storage_strides` declares read support. The seven meta-operators
consume actual input strides; fused elementwise kernels receive those strides
as runtime scalar arguments. Their strided key variants preserve cache
separation without baking stride values into the key. Index tensors themselves
are normalized at the construction boundary where a gather requires dense
index storage.

The generated operator factory inserts explicit contiguous graph inputs for
dense-only implementations before constructing their concrete Var members or
edges. This conversion is differentiable and visible in the graph. It is not a
temporary mutation of an existing Var at execution time. Kernels which mutate
input storage declare `mutates_storage_inputs`; noncontiguous writable inputs
are rejected instead of redirecting the write into an unnoticed copy.
Preallocated CodeOp output buffers likewise require contiguous storage.

`Op::is_storage_view()` identifies metadata-only operations. The executor serves
their alias allocation without invoking a provider compute kernel. Backend
overrides must not launch an Expand kernel into a stride-zero alias's smaller
allocation. Clone/detach, reshape/view and basic slice metadata follow this
boundary; they remain graph nodes with normal gradient rules.

## Language and device boundaries

`_storage_strides()`, `_storage_offset()`, `_storage_is_contiguous()` and the
existing `_storage_address` expose native facts to a frontend. Torch stride and
contiguity queries delegate to them. Public basic-view records still govern
holder assignment; source assignment refreshes live slice, reshape, expand and
transpose records. Writes through overlapping expanded views are rejected.

`numpy()` returns a dense NumPy copy by gathering according to storage strides.
`data` exports byte strides with an allocation-owning base; overlapping
stride-zero exports are read-only. Scalar `item()` reads one physical element.
Dense data assignment scatters into non-overlapping positive-stride storage and
rejects overlapping destinations. Byte reinterpretation requires an explicitly
contiguous source rather than silently changing its storage identity.

CUDA's generated meta-kernels consume native strides. Dense external library
calls use explicit contiguous graph inputs. ACL descriptor helpers accept the
source Var and pass its real strides and physical storage span to CANN; the
native contiguous operation uses a same-dtype Cast into dense output. The
indexed-copy planner remains dense-only and reports that requirement explicitly.
ACL descriptor and kernel changes need real CANN/NPU validation; host syntax
checks alone do not establish support for a device or dtype.

## Collaboration boundary

Core owns Var layout, allocation spans, alias establishment and view shape
rules. Code generation owns input-layout adaptation and stride-aware accesses.
Providers own descriptor conversion and any implementation-specific contiguous
requirements. Frontends query native state and record public view relationships.
New consumers should declare their layout requirements at registration rather
than adding hidden materialization inside a kernel or rebuilding shape-based
stride tables in Python.

Focused tests live in `tests/core/test_storage_strides.py` and
`tests/core/test_view_storage.py`; they cover forced expand alias identity,
subsequent arithmetic/reduction/indexing, allocation budget, assignment refresh,
NumPy export, positive-step storage offsets, dense library boundaries and grads.
