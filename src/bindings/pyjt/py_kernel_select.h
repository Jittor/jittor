#pragma once

#include <Python.h>
#include "core/common.h"

namespace jittor {

// The dispatcher's selection loop (`jittor/_runtime/dispatch.py`), native.
//
// `select_kernel` runs for every dispatched operator -- a thousand times a
// token of an eager decode -- and in Python it was a walk over the arguments,
// a placement query, a dtype-name lookup per tensor and a loop over the
// candidates, each a handful of interpreter frames. The registry stays in
// Python, which owns registration, priorities and overrides; this keeps a
// snapshot of what it resolved and is told to drop it on every change.

// The functions the native side calls back into: `candidates(op, backend)`
// returns the priority-ordered registrations, `dtype_name(dtype)` the
// canonical name of a native dtype.
// @pyjt(_kernel_select_bind)
void kernel_select_bind(PyObject* candidates, PyObject* dtype_name);

// Every registration changed: forget the resolved candidates and take the
// new set of op names that have any.
// @pyjt(_kernel_select_invalidate)
void kernel_select_invalidate(PyObject* op_names);

// Whether any backend has a registration under `op` (C++ callers).
bool kernel_op_registered(const char* op);

// Declares that the Python `supports` predicate `fn` answers what the named
// native rule answers, so the loop below can evaluate the rule instead of
// calling back into Python for it. A rule may also say it does not know --
// an argument it does not recognise, a library not loaded yet -- and then
// `fn` itself is asked, so a rule only ever has to be right when it answers.
// Rules:
//   "same_float:<op>"     both leading arguments are Vars of one floating
//                         dtype, and native op <op> is registered
//   "rms_norm_inference"  `rms_norm_cuda._rms_norm_contract` holds
//   "rms_norm_training"   declines under `no_grad`
//   "layer_norm_inference" `layer_norm_cuda._supports_layer_norm_inference`
//                         holds, for a Var weight and bias
// @pyjt(_kernel_select_native_rule)
void kernel_select_native_rule(PyObject* fn, const string& rule);

// The implementation `select_kernel(op, *args, **kwargs)` picks, or None.
// @pyjt(_kernel_select)
PyObject* kernel_select(PyObject* op, PyObject* args, PyObject* kwargs);

} // namespace jittor
