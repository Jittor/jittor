#pragma once
#include <Python.h>
#include "core/kernel_func.h"

namespace jittor {
KernelFunc kernel_func_from_python(PyObject* callable);
// Internal, already-materialized export. Only the executor's callback uses it.
PyObject* kernel_buffer_capsule(Var* var);
// Lazily materialize a DLPack capsule for an active external_kernel callback view.
// @pyjt(_kernel_buffer_dlpack)
PyObject* kernel_buffer_dlpack(PyObject* scope_capsule, bool is_output, int index);
} // namespace jittor
