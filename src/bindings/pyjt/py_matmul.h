#pragma once

#include <Python.h>
#include "core/common.h"

namespace jittor {

// Execute the ordinary matrix dispatch under its contraction AMP policy.
// Common valid matrix shapes are checked directly; other cases use the
// Python path for its rank handling, AMP policy and original exceptions.
// The callback still selects the live provider and builds fresh operations.
// @pyjt(_matmul_contraction_call)
PyObject* matmul_contraction_call(PyObject* body, PyObject* fallback,
                                PyObject* a, PyObject* b);

} // namespace jittor
