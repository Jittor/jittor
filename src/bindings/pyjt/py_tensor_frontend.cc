#include "bindings/pyjt/py_tensor_frontend.h"
#include <stdexcept>

namespace jittor {

extern PyHeapTypeObject PyjtVarHolder;

namespace {
// One shared symbol owner across the core and all loaded JIT extensions.
// Access occurs with the GIL held, just like the Python conversion boundary.
PyObject* frontend_context = nullptr;

PyObject* context_variable() {
    if (!frontend_context) {
        frontend_context = PyContextVar_New("jittor.tensor_frontend", nullptr);
        if (!frontend_context)
            throw std::runtime_error("cannot create tensor frontend context");
    }
    return frontend_context;
}

PyObject* selected_type() {
    PyObject* value = nullptr;
    if (PyContextVar_Get(context_variable(), nullptr, &value) < 0)
        throw std::runtime_error("cannot read tensor frontend context");
    return value;
}

bool is_frontend_instance(PyObject* value) {
    return value && Py_TYPE(value) != &PyjtVarHolder.ht_type
        && PyObject_TypeCheck(value, &PyjtVarHolder.ht_type);
}

PyObject* frontend_candidate(PyObject* value, bool scan_sequences) {
    if (is_frontend_instance(value)) return value;
    // Match the vector converter's accepted containers without invoking any
    // user iterator, recursing, or scanning ordinary array/list factories.
    if (scan_sequences && value
            && (PyList_CheckExact(value) || PyTuple_CheckExact(value))) {
        auto items = PySequence_Fast_ITEMS(value);
        for (Py_ssize_t i = 0; i < Py_SIZE(value); ++i)
            if (is_frontend_instance(items[i])) return items[i];
    }
    return nullptr;
}
} // namespace

PyObject* current_tensor_frontend_type() {
    PyObject* value = selected_type();
    if (!value) {
        value = reinterpret_cast<PyObject*>(&PyjtVarHolder.ht_type);
        Py_INCREF(value);
    }
    return value;
}

PyObject* set_tensor_frontend_type(PyObject* type) {
    USER_CHECK(type && PyType_Check(type)
        && PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(type),
                            &PyjtVarHolder.ht_type))
        << "tensor frontend must be a real Var type or subtype";
    PyObject* token = PyContextVar_Set(context_variable(), type);
    if (!token)
        throw std::runtime_error("cannot set tensor frontend context");
    return token;
}

void reset_tensor_frontend_type(PyObject* token) {
    if (PyContextVar_Reset(context_variable(), token) < 0)
        throw std::runtime_error("cannot reset tensor frontend context token");
}

void PyTensorFrontendScope::select(
    PyObject* self, PyObject** args, int64 count, bool scan_sequences) {
    PyObject* candidate = is_frontend_instance(self) ? self : nullptr;
    for (int64 i = 0; !candidate && args && i < count; ++i)
        candidate = frontend_candidate(args[i], scan_sequences);
    if (!candidate) return;
    PyObject* existing = selected_type();
    if (existing) {
        Py_DECREF(existing);
        return;
    }
    token_ = set_tensor_frontend_type(
        reinterpret_cast<PyObject*>(Py_TYPE(candidate)));
}

PyTensorFrontendScope::PyTensorFrontendScope(PyObject* candidate) {
    select(candidate, nullptr, 0, false);
}

PyTensorFrontendScope::PyTensorFrontendScope(PyTypeObject* type) {
    token_ = set_tensor_frontend_type(reinterpret_cast<PyObject*>(type));
}

PyTensorFrontendScope::PyTensorFrontendScope(
    PyObject* self, PyObject** args, int64 count, bool scan_sequences) {
    select(self, args, count, scan_sequences);
}

PyTensorFrontendScope::~PyTensorFrontendScope() noexcept {
    if (!token_) return;
    PyObject *error_type = nullptr, *error_value = nullptr, *error_traceback = nullptr;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    if (PyContextVar_Reset(frontend_context, token_) < 0) PyErr_Clear();
    Py_DECREF(token_);
    PyErr_Clear();
    PyErr_Restore(error_type, error_value, error_traceback);
}

} // namespace jittor
