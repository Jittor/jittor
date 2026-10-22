#pragma once

#include <Python.h>
#include "core/common.h"

namespace jittor {

// Native fast paths for the torch frontend's per-tensor accessors: each
// replaces a Python function on a path every operator of a model takes.

// `Tensor.dtype` on a frontend tensor: the frontend's dtype object for the
// native dtype, from the `_frontend_dtype_objects` table its type carries, or
// the native name where the table has none (`method_api._dtype_get`).
// @pyjt(_frontend_dtype)
PyObject* frontend_dtype(PyObject* self);

// What the fast paths below call back to decide: `tracing()`, whether a
// step is being captured (`jittor._runtime.step_capture.tracing`).
// @pyjt(_compat_fast_bind)
void compat_fast_bind(PyObject* tracing);

// `torch.cat(tensors, dim)` for the common case -- two to 64 non-empty Vars
// of one dtype, same rank, on the current device, no concat kernel
// registered, no step being captured -- built as `jittor.concat` builds it:
// an `empty` of the result and one `setitem` per input. Anything else
// answers NotImplemented and the Python `cat` takes the call.
// @pyjt(_fast_cat)
PyObject* fast_cat(PyObject* tensors, int64 dim);

// `tensor[index]` for a basic index -- ints, slices, None and Ellipsis,
// alone or in a tuple -- on a frontend tensor that carries no CPU residency
// hint and no `.data` ownership record, with no getitem kernel registered:
// the native getitem op, recorded as a view of `self`, as the Python
// `_torch_getitem` and `jittor.getitem` together do. NotImplemented otherwise.
// @pyjt(_fast_getitem)
PyObject* fast_getitem(PyObject* self, PyObject* index);

// The native operators the binary fast path calls: `natives` holds the
// jittor Var's own `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__mul__`,
// `__rmul__`, `__truediv__` and `__rtruediv__` as captured before the frontend
// replaced them (slot wrappers; anything else leaves that operator on the
// Python path), and `mark_cpu_like` is the frontend's `_mark_cpu_like`.
// `widen_scalar_division`: take a floating tensor over a Python float here,
// widened as `_true_division` does (everywhere but ACL).
// @pyjt(_compat_fast_bind_binary)
void compat_fast_bind_binary(PyObject* natives, PyObject* mark_cpu_like,
                             bool widen_scalar_division=false);

// `tensor <op> other` for the common case of `method_api._promoting_binary`
// and `_true_division`: a Var of the same dtype (not unsigned; floating for a
// division), or a Python float against a floating tensor or a Python int
// against a floating or signed-integer one, which keep the tensor's dtype.
// `code` indexes the operators above. None for anything else, and the Python
// operator takes the call.
// @pyjt(_fast_binary)
PyObject* fast_binary(PyObject* self, PyObject* other, int code);

// `nn.gelu(x)` (exact) as its Python body builds it, operator for operator:
// `0.5 * x * (1.0 + erf(x * 0.7071067811865476))`, x widened to float32 for a
// half type and the result cast back, each product and sum taken as
// `_fast_binary` takes it. None when the binary operators are not bound, x is
// not floating, a kernel is registered for "nn.gelu", or any step declines.
// @pyjt(_fast_gelu)
PyObject* fast_gelu(PyObject* x);

// `tensor.view(*shape)` / `reshape(*shape)` of a dense tensor, shape given as
// ints or one tuple or list of them: the reshape op, recorded as a storage
// view of `self`, as `jittor.view` does. None for anything else.
// @pyjt(_fast_view)
PyObject* fast_view(PyObject* self, PyObject* shape);

// `tensor.unsqueeze(dim)` of a dense tensor, as the reshape above.
// @pyjt(_fast_unsqueeze)
PyObject* fast_unsqueeze(PyObject* self, int64 dim);

// `tensor.transpose(dim0, dim1)`, as `jittor.transpose` builds it: composed
// with the transpose this tensor is a view of, or is computed by, and
// recorded as a transpose view. None when a transpose kernel is registered.
// @pyjt(_fast_transpose)
PyObject* fast_transpose(PyObject* self, int64 dim0, int64 dim1);

// `tensor.permute(axes)` / `transpose(*axes)` for a permutation of exact,
// in-range, distinct non-negative ints, built as `_fast_transpose` builds it.
// None for anything else, and `jittor.transpose` reports it.
// @pyjt(_fast_permute)
PyObject* fast_permute(PyObject* self, PyObject* axes);

// `jittor.Var.clamp(min_v, max_v)` for a plain Var of a float or integer
// dtype between Python int/float bounds (either may be None, not both): the
// graph `_clamp_cpu` builds, op for op -- each bound converted the way the
// generated bindings convert a Python number, the integer path's maximum and
// minimum, the general path's select and NaN keep -- built in one call
// instead of sixteen. NotImplemented for anything else, and for an empty or
// dynamically shaped Var, which `_clamp_cpu` answers itself.
// @pyjt(_fast_clamp)
PyObject* fast_clamp(PyObject* self, PyObject* min_v, PyObject* max_v);

// `jittor.any(x, dims, keepdims)` (`all` when `is_and`) of a bool Var on CPU
// or CUDA: the logical reduction with a bool result, where `any_`/`all_`
// reduce a bool input into int32 and the caller then casts that back -- a
// second kernel for every truth test. NotImplemented for any other dtype or
// backend, for `dims` other than an int or a sequence of in-range ints,
// and `any_(...).bool()` takes the call.
// @pyjt(_logical_reduce)
PyObject* logical_reduce(PyObject* self, PyObject* dims, bool keepdims, bool is_and);

// `_fast_getitem` and `_fast_cat` as jittor's own `getitem` and `concat`
// reach them: each Var operand is first read the way a generated binding
// reads it (`VarHolder::grad_input`, which brings an aliased view up to date
// with its source's in-place writes while gradients are recorded), so the
// native path sees what the Python path it stands in for would have built on.
//
// `gather`: also answer the index forms holding index Vars -- a lone bool
// mask, a lone index Var, a tuple with index Vars in it -- as jittor's
// `_fast_getitem` answers them. The caller passes whether that path applies,
// i.e. whether `Var.getitem` is still jittor's own.
// @pyjt(_jt_basic_getitem)
PyObject* jt_basic_getitem(PyObject* self, PyObject* index, bool gather=false);
// @pyjt(_jt_concat)
PyObject* jt_concat(PyObject* tensors, int64 dim);

// `x[index] = value` as jittor's `setitem` and `var_setitem` together build
// it, for a plain Var `x` (not complex64, not on ACL) and a Var or exact
// Python int, float or bool `value`: an index of ints, slices, None,
// Ellipsis and index Vars, or a lone bool mask -- the `ternary` a mask of
// `x`'s own shape becomes under a scalar or one-element value, its `where()`
// coordinates otherwise. Returns `x`, assigned. NotImplemented for anything
// else, and the Python path takes the call. The caller has checked that no
// setitem kernel is registered and that `Var.setitem` is still jittor's own.
// @pyjt(_jt_setitem)
PyObject* jt_setitem(PyObject* self, PyObject* index, PyObject* value);

} // namespace jittor
