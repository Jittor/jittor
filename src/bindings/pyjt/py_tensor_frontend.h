#pragma once

#include <Python.h>
#include "core/common.h"

namespace jittor {

// Returns a new reference; absent an explicit frontend, this is native Var.
EXTERN_LIB PyObject* current_tensor_frontend_type();

// @pyjt(_set_tensor_frontend_type)
PyObject* set_tensor_frontend_type(PyObject* type);

// @pyjt(_reset_tensor_frontend_type)
void reset_tensor_frontend_type(PyObject* token);

// Generated binding scopes infer a frontend only when none is already active.
// A plain native Var never overrides its caller's explicit frontend choice.
class PyTensorFrontendScope {
    PyObject* token_ = nullptr;
    int previous_policy_ = -1;
    void apply_policy(PyObject* type);
    void restore() noexcept;
    void select(PyObject* self, PyObject** args, int64 count, bool scan_sequences);
public:
    PyTensorFrontendScope();
    explicit PyTensorFrontendScope(PyObject* candidate);
    explicit PyTensorFrontendScope(PyTypeObject* type);
    PyTensorFrontendScope(PyObject* self, PyObject** args, int64 count,
                          bool scan_sequences = false);
    ~PyTensorFrontendScope() noexcept;
    PyTensorFrontendScope(const PyTensorFrontendScope&) = delete;
    PyTensorFrontendScope& operator=(const PyTensorFrontendScope&) = delete;
};

} // namespace jittor
