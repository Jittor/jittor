#pragma once

#include <Python.h>
#include "core/common.h"
#include "runtime/tensor_placement.h"
#include "runtime/float32_precision.h"

namespace jittor {

// Returns a new reference; absent an explicit frontend, this is native Var.
EXTERN_LIB PyObject* current_tensor_frontend_type();

// @pyjt(_set_tensor_frontend_type)
PyObject* set_tensor_frontend_type(PyObject* type);

// @pyjt(_reset_tensor_frontend_type)
void reset_tensor_frontend_type(PyObject* token);

// Construction-only placement override; independent of Runtime execution flags.
// @pyjt(_set_tensor_placement)
PyObject* set_tensor_placement_context(int backend, int device=0);
// @pyjt(_get_tensor_placement)
PyObject* get_tensor_placement_context();
// @pyjt(_reset_tensor_placement)
void reset_tensor_placement_context(PyObject* token);

// @pyjt(_set_float32_precision)
PyObject* set_float32_precision_context(int matmul, int cudnn);
// @pyjt(_reset_float32_precision)
void reset_float32_precision_context(PyObject* token);

// Generated binding scopes infer a frontend only when none is already active.
// A plain native Var never overrides its caller's explicit frontend choice.
class PyTensorFrontendScope {
    PyObject* token_ = nullptr;
    int previous_policy_ = -1;
    TensorPlacement previous_placement_;
    bool restore_placement_ = false;
    Float32PrecisionPolicy previous_precision_;
    bool restore_precision_ = false;
    void apply_policy(PyObject* type, PyObject* candidate=nullptr);
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
