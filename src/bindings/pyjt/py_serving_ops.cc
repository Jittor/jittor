#include "bindings/pyjt/py_serving_ops.h"
#include "bindings/pyjt/py_converter.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "core/var_holder.h"
#include "mem/allocator.h"
#include "ops/composite/code_op.h"
#include "ops/op_register.h"
#include "runtime/dispatch_context.h"
#include "runtime/executor_entry.h"
#include "runtime/graph_capture.h"
#include <cmath>

namespace jittor {

DECLARE_FLAG(bool, no_grad);
DECLARE_FLAG(int, amp_reg);

namespace {
auto make_code = op_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&,
    string&&, vector<string>&&, string&&, string&&, vector<string>&&, string&&,
    DataMap&&, string&&>("code");
}

PyObject* try_rms_norm_cuda(PyObject* x, PyObject* weight, PyObject* epsilon,
                           PyObject* source_fn) {
    if (!no_grad || amp_reg || !PyObject_TypeCheck(x, &PyjtVarHolder.ht_type)
            || !PyObject_TypeCheck(weight, &PyjtVarHolder.ht_type)
            || !(PyFloat_CheckExact(epsilon) || PyLong_CheckExact(epsilon)))
        Py_RETURN_NONE;
    Var* input = GET_RAW_PTR(VarHolder, x)->var;
    Var* gamma = GET_RAW_PTR(VarHolder, weight)->var;
    if (input->dtype() != ns_float32 || gamma->dtype() != ns_float32
            || !input->shape.size())
        Py_RETURN_NONE;
    for (int i = 0; i < input->shape.size(); ++i)
        if (input->shape[i] <= 0) Py_RETURN_NONE;
    int64 hidden = input->shape[input->shape.size()-1];
    if (hidden > 4096 || gamma->shape.size() != 1 || gamma->shape[0] != hidden)
        Py_RETURN_NONE;
    double eps = PyFloat_AsDouble(epsilon);
    if (PyErr_Occurred()) {
        // The Python contract declines an integer too large for a float.
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) return nullptr;
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    if (!std::isfinite(eps) || eps <= 0) Py_RETURN_NONE;
    if (query_dispatch_context({input, gamma}).backend != "cuda") Py_RETURN_NONE;

    int64 threads = 32;
    while (threads < std::min<int64>(hidden, 1024)) threads *= 2;
    PyObjHolder source(PyObject_CallFunction(source_fn, "LLLd", (long long)hidden,
        (long long)threads, (long long)(threads/32), eps));
    if (!source.obj) return nullptr;
    Py_ssize_t length;
    const char* text = PyUnicode_AsUTF8AndSize(source.obj, &length);
    if (!text) return nullptr;

    PyTensorFrontendScope scope(x, nullptr, 0, false);
    // The generated code-op constructor normalizes storage strides, just as
    // jt.code does. Source text is owned and cached by the CUDA backend.
    // Pass eps via DataMap so different eps values share one compiled kernel.
    unique_ptr<VarHolder> result(new VarHolder(make_code(input->shape,
        input->dtype(), {input, gamma}, "", {}, "", string(text, length), {}, "",
        DataMap{{"eps", eps}}, "")));
    result->stop_grad();
    return to_py_object<VarHolder*>(result.release());
}

PyObject* try_snapshot_cuda_step_output(PyObject* source) {
    if (!no_grad || state_capture_active() || graph_capture_active()
            || !PyObject_TypeCheck(source, &PyjtVarHolder.ht_type))
        Py_RETURN_NONE;
    Var* value = GET_RAW_PTR(VarHolder, source)->var;
    if (!value->is_finished() || !value->is_contiguous() || !value->size
            || !value->mem_ptr || !value->allocator || value->flag(VarFlags::_execution_failed))
        Py_RETURN_NONE;
    const Device device = allocation_device(value->allocator);
    if (device.backend != BackendId::Cuda) Py_RETURN_NONE;

    PyTensorFrontendScope frontend(Py_TYPE(source));
    ExecutorEntryScope entry;
    VarPtr snapshot(value->shape, value->dtype());
    snapshot->placement = value->placement;
    snapshot->device_id = value->device_id;
    snapshot->set_stop_grad();
    // The snapshot is owned by the caller and may outlive the graph pool.
    CHECK(snapshot->alloc(get_regular_allocator(device, false)));
    backend_copy(snapshot->mem_ptr, device, value->mem_ptr, device, value->size, true);
    snapshot->finish_pending_liveness();
    return to_py_object<VarHolder*>(new VarHolder(move(snapshot)));
}

} // namespace jittor
