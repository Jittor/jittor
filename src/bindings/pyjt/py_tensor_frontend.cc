#include "bindings/pyjt/py_tensor_frontend.h"
#include "core/grad.h"
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
    // Factories may have no tensor inputs. Their explicit context still owns
    // both the Python result type and the native autograd policy for this call.
    PyObject* existing = selected_type();
    if (existing) {
        try {
            apply_policy(existing);
        } catch (...) {
            Py_DECREF(existing);
            throw;
        }
        Py_DECREF(existing);
        return;
    }
    PyObject* candidate = is_frontend_instance(self) ? self : nullptr;
    for (int64 i = 0; !candidate && args && i < count; ++i)
        candidate = frontend_candidate(args[i], scan_sequences);
    if (!candidate) return;
    token_ = set_tensor_frontend_type(
        reinterpret_cast<PyObject*>(Py_TYPE(candidate)));
    apply_policy(reinterpret_cast<PyObject*>(Py_TYPE(candidate)));
}

void PyTensorFrontendScope::apply_policy(PyObject* type) {
    PyObject* value = PyObject_GetAttrString(type, "_frontend_autograd_policy");
    if (!value) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
            return;
        }
        throw std::runtime_error("cannot read tensor frontend autograd policy");
    }
    bool integer = PyLong_Check(value);
    if (!integer) {
        Py_DECREF(value);
        USER_CHECK(integer) << "tensor frontend autograd policy must be an integer";
    }
    long bits = PyLong_AsLong(value);
    Py_DECREF(value);
    if (PyErr_Occurred())
        throw std::runtime_error("invalid tensor frontend autograd policy integer");
    USER_CHECK(bits >= 0 && bits <= 3)
        << "tensor frontend autograd policy must be in [0, 3]";
    previous_policy_ = get_autograd_policy();
    set_autograd_policy((bits & 1) != 0, (bits & 2) != 0);
}

PyTensorFrontendScope::PyTensorFrontendScope()
    : PyTensorFrontendScope(static_cast<PyObject*>(nullptr)) {}

PyTensorFrontendScope::PyTensorFrontendScope(PyObject* candidate) {
    try {
        select(candidate, nullptr, 0, false);
    } catch (...) {
        restore();
        throw;
    }
}

PyTensorFrontendScope::PyTensorFrontendScope(PyTypeObject* type) {
    try {
        token_ = set_tensor_frontend_type(reinterpret_cast<PyObject*>(type));
        apply_policy(reinterpret_cast<PyObject*>(type));
    } catch (...) {
        restore();
        throw;
    }
}

PyTensorFrontendScope::PyTensorFrontendScope(
    PyObject* self, PyObject** args, int64 count, bool scan_sequences) {
    try {
        select(self, args, count, scan_sequences);
    } catch (...) {
        restore();
        throw;
    }
}

void PyTensorFrontendScope::restore() noexcept {
    if (!token_ && previous_policy_ < 0) return;
    PyObject *error_type = nullptr, *error_value = nullptr, *error_traceback = nullptr;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    if (previous_policy_ >= 0) {
        // This setter only writes the two native policy bits; it cannot invoke
        // Python or allocate. Keep destruction safe even during unwinding.
        set_autograd_policy((previous_policy_ & 1) != 0,
                            (previous_policy_ & 2) != 0);
        previous_policy_ = -1;
    }
    if (token_) {
        if (PyContextVar_Reset(frontend_context, token_) < 0) PyErr_Clear();
        Py_DECREF(token_);
        token_ = nullptr;
    }
    PyErr_Clear();
    PyErr_Restore(error_type, error_value, error_traceback);
}

PyTensorFrontendScope::~PyTensorFrontendScope() noexcept {
    restore();
}

} // namespace jittor
