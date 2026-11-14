#pragma once

#include <Python.h>

namespace jittor {

// The selected CUDA RMSNorm provider's FP32 no-grad construction path.
// Consult the live backend source generator and construct a fresh CodeOp;
// retain no tensors, graph, execution plan or source callback. Unsupported
// inputs return None for the provider's general Python implementation.
// @pyjt(_try_rms_norm_cuda)
PyObject* try_rms_norm_cuda(PyObject* x, PyObject* weight, PyObject* epsilon,
                           PyObject* source_fn);

} // namespace jittor
