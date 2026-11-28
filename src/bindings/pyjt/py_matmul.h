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

// The CUDA batched relay after provider selection, for inference operands
// whose batch axes already agree. None leaves broadcasting to its caller.
// @pyjt(_cublas_matmul_matching_batch)
PyObject* cublas_matmul_matching_batch(PyObject* a, PyObject* b,
                                      bool trans_a, bool trans_b, float alpha);

// @pyjt(_cublas_matmul_frontend)
PyObject* cublas_matmul_frontend(PyObject* a, PyObject* b,
                                PyObject* tensor_type, PyObject* provider);

} // namespace jittor
