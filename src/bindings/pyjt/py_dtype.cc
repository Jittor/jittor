#include "bindings/pyjt/py_dtype.h"
#include <cstring>
#include <stdexcept>

namespace jittor {
namespace {
vector<PyObject*> dtype_types;
const char* bare_name(const char* name) {
    return std::strncmp(name, "torch.", 6) == 0 ? name + 6 : name;
}
}

void register_python_dtype_type(PyObject* type) {
    USER_CHECK(PyType_Check(type)) << "dtype registration requires a Python type";
    for (auto* current : dtype_types)
        if (current == type) return;
    dtype_types.push_back(type);
    Py_INCREF(type);
}

bool is_python_dtype(PyObject* object) {
    for (auto* type : dtype_types)
        if (Py_TYPE(object) == reinterpret_cast<PyTypeObject*>(type)) return true;
    return false;
}

PyObject* python_dtype_from_name(const char* name) {
    for (auto* type : dtype_types) {
        auto* registry = PyObject_GetAttrString(type, "_registry");
        if (!registry) throw std::runtime_error("registered dtype has no registry");
        if (!PyDict_Check(registry)) {
            Py_DECREF(registry);
            throw std::runtime_error("registered dtype registry must be a dictionary");
        }
        auto* value = PyDict_GetItemString(registry, bare_name(name));
        Py_XINCREF(value);
        Py_DECREF(registry);
        if (value) return value;
    }
    return nullptr;
}

bool is_python_dtype_name(const char* name) {
    auto* value = python_dtype_from_name(name);
    if (!value) return false;
    Py_DECREF(value);
    return true;
}

PyObject* python_dtype_name(PyObject* object) {
    auto* name = PyObject_GetAttrString(object, "_jittor_compute_name");
    if (!name) throw std::runtime_error("dtype cannot participate in Jittor computation");
    if (!PyUnicode_Check(name)) {
        Py_DECREF(name);
        throw std::runtime_error("registered dtype must supply a string compute name");
    }
    return name;
}

string checked_dtype_name(NanoString value) {
    USER_CHECK(value.is_dtype()) << "expected a tensor dtype, received " << value;
    return value.to_cstring();
}
}
