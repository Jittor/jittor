# Torch dtype and native argument boundaries

Status: integrated into the coordination worktree, 2026-09-08; CPU/CUDA boundaries verified.
Owner: Torch compatibility and native bindings maintainers.
Recheck when a dtype, array constructor, backend predicate, or serializer is added.

`torch.dtype` is an immutable object, not a Python string. Aliases such as
`torch.long` and `torch.int64` resolve to the same object. `str`, `repr`, and
formatting produce `torch.int64`; comparison with either `"int64"` or
`"torch.int64"` is false. Pickling returns the canonical object and accepts the
former string-subclass pickle state. Dtype objects are not callable. The removed
native-as-Torch mode's captured cast-converter table no longer exists; use
`tensor.to(dtype=...)`, or `jt.float32(...)` for native Jittor code.

There are two different operations at the boundary:

| Operation | Owner | Contract |
| --- | --- | --- |
| Read a dtype name for metadata, dispatch, source generation, or persistence | `jittor._core.dtypes.dtype_name(value)` | Returns a bare canonical name; does not grant computational support. |
| Consume a dtype before a Python/NumPy allocation | `jittor._core.dtypes.dtype_for_compute(value)` | Uses the native checked NanoString converter and rejects unsupported placeholders. |
| Register a frontend dtype class | `jittor._core.dtypes.register_dtype_type(type)` | Registers the same exact class at Python positional-dtype and C++ argument boundaries. |
| Consume an object in a native dtype argument | `bindings/pyjt/py_dtype.{h,cc}` and `py_converter.h` | Accepts only explicitly registered frontend types, in addition to the existing native and NumPy spellings. Arbitrary objects with a `name` or `type` attribute are not accepted. |

The native registry is shared by the core and its JIT extensions. Its checked
converter preserves Python exceptions: a placeholder raises
`NotImplementedError` before an operator is constructed. Canonical bare strings
continue through the existing native fast path. Torch-prefixed strings and
registered placeholder names resolve through the registered dtype objects.
`ones`, `zeros`, and `empty` also recognize a registered dtype in their
historical final positional argument; treating it as a shape component would
silently restore the default float32 dtype.

The current frontend defines 34 distinct dtype objects. Fourteen have native
allocation/computation representations: bool, four signed integers, four
unsigned integers, float16, bfloat16, float32, float64, and complex64. The other
20 objects remain available for import-time dictionaries and metadata, but
cannot be passed to factories, casts, random generation, or direct native
operator dtype parameters. Backend-specific support is still narrower than
this framework-level set; this change does not establish NPU/ROCm execution.

Native Python code must not use `str(tensor.dtype)` as a kernel type name or
compare a frontend dtype object to a string. Use `dtype_name` for those
operations. Pass dtype objects directly to native operators, or use
`dtype_for_compute` before handing the name to NumPy. Preserve NumPy input
dtypes and construct float64 directly rather than creating float32 and casting
after precision has already been lost. BF16 NumPy input uses an explicit
float32 staging array followed by the native BF16 cast.

The focused verification covers immutable identity, old/new pickles, direct
native object and prefixed-string dtype arguments, NumPy values, final
positional dtype handling, mixed promotion, all placeholder rejection paths,
float64 default construction, matmul, and its gradient. CPU and real CUDA run
in both legacy and independent frontend modes. No performance claim or full
API certification is attached to these checks.

Integration note: parallel changes adding `adam_update` must use `dtype_name`
for its dtype comparisons and serialized state names. Runtime-owner moves must
preserve the frontend registration call and the core-owned normalizers. Do not
reintroduce a compatibility import into the native normalizer or restore string
inheritance to make a stale call site work.
