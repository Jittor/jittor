#pragma once

#include <Python.h>
#include "core/common.h"

namespace jittor {

// A torch-frontend module call, native (`jittor/compat/torch/nn_frontend.py`).
//
// Every `module(...)` of a model running on the torch frontend entered a
// Python `tensor_frontend` scope -- frontend type, precision tiers, placement
// from the first tensor argument, autograd policy -- and then went through
// `Module.__call__`, the frontend's `_dispatch_call` and its bookkeeping
// before reaching `forward`: five frames and a dozen attribute reads, about a
// fifth of the host time of an eager Qwen3 decode token, 426 module calls of
// it. This enters the same scope natively and, when nothing on the way would
// act -- no hooks, not the outermost call, parameters already published, no
// FSDP state, no pipelining -- calls the frontend's `_dispatch_module_call`
// directly. Anything else takes the native module's own `__call__`, as before.

// The Python pieces the fast path stands in for, once per install:
// `module_cls` (the native Module, whose `_call_depth` says whether a call is
// outermost), `dispatch_call` (the frontend's `_dispatch_call`, which a type
// must still have for the shortcut to be the same call), `dispatch_module_call`,
// `published` (ids of modules whose parameters are already registered),
// `pipeline` (its "threshold" > 0 turns the shortcut off), `device_context`
// (the active `with torch.device(...)`, or None) and `slow` (the Python
// `module_call`, for a call whose placement this does not resolve).
//
// `dispatch_parts` lets the shortcut run `_dispatch_module_call` itself:
// "prefer_forward" (`_prefer_forward`), "standard_rms_norm", and, for the
// bias-free `nn.Linear` it computes directly, "linear_execute" (the native
// `Linear.execute`) and "matmul_kernel" (the cuBLAS relay the dispatcher must
// pick for it). Without the first two the shortcut calls
// `dispatch_module_call` as before.
// @pyjt(_module_call_bind)
void module_call_bind(PyObject* module_cls, PyObject* dispatch_call,
                      PyObject* dispatch_module_call, PyObject* published,
                      PyObject* pipeline, PyObject* device_context, PyObject* slow,
                      PyObject* dispatch_parts);

// @pyjt(_module_call)
PyObject* module_call_native(PyObject* module, PyObject* args, PyObject* kwargs);

// The inference group norm of a dense [N, H, W, C] `source` with activation
// `act` ("" for none) applied in its last pass, as the NCHW view of its
// storage -- what `group_norm_cuda._group_norm_nhwc`'s `build(act)` makes. None
// for an activation it has no pass for. The offer an inference `nn.GroupNorm`
// leaves for the activation after it (see py_module_call.cc).
// @pyjt(_group_norm_nhwc_build)
PyObject* group_norm_nhwc_build(PyObject* source, PyObject* weight, PyObject* bias,
                                int64 groups, double eps, const string& act);

} // namespace jittor
