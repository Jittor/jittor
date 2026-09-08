#pragma once
#include <Python.h>
#include "core/common.h"
#include "type/nano_string.h"

namespace jittor {
// Explicit registration avoids probing arbitrary objects for dtype-like fields.
// @pyjt(_register_python_dtype_type)
void register_python_dtype_type(PyObject* type);
EXTERN_LIB bool is_python_dtype(PyObject* object);
EXTERN_LIB bool is_python_dtype_name(const char* name);
// Returned name owns one reference. Unsupported placeholders retain their
// original Python NotImplementedError across the generated binding boundary.
EXTERN_LIB PyObject* python_dtype_name(PyObject* object);
EXTERN_LIB PyObject* python_dtype_from_name(const char* name);

// @pyjt(_checked_dtype_name)
string checked_dtype_name(NanoString value);
}
