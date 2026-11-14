#include "bindings/pyjt/py_serving_ops.h"
#include "bindings/pyjt/py_converter.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "core/var_holder.h"
#include "ops/composite/code_op.h"
#include "ops/op_register.h"
#include "runtime/dispatch_context.h"
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
    unique_ptr<VarHolder> result(new VarHolder(make_code(input->shape,
        input->dtype(), {input, gamma}, "", {}, "", string(text, length), {}, "", {}, "")));
    result->stop_grad();
    return to_py_object<VarHolder*>(result.release());
}

} // namespace jittor
