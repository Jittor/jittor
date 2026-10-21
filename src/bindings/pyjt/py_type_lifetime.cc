#include "bindings/pyjt/py_type_lifetime.h"
#include <stdexcept>
#include <unordered_map>

namespace jittor {

namespace {
struct Watch {
    PyObject* weak = nullptr;
    vector<void (*)(PyObject*)> forget;
};
// Guarded by the GIL.
std::unordered_map<PyObject*, Watch> watched;
PyObject* collected_callback = nullptr;

PyObject* type_collected(PyObject*, PyObject* weak) {
    for (auto it = watched.begin(); it != watched.end(); ++it) {
        if (it->second.weak != weak) continue;
        PyObject* type = it->first;
        Watch watch = move(it->second);
        // Out of the table before the caches let go of what they hold, which
        // may collect another watched type.
        watched.erase(it);
        for (auto forget : watch.forget) forget(type);
        Py_DECREF(watch.weak);
        break;
    }
    Py_RETURN_NONE;
}

PyMethodDef collected_def = {"_type_collected", type_collected, METH_O, nullptr};
} // namespace

void on_type_collected(PyObject* type, void (*forget)(PyObject* type)) {
    if (!PyType_Check(type) || !PyType_HasFeature((PyTypeObject*)type, Py_TPFLAGS_HEAPTYPE))
        return;
    auto found = watched.find(type);
    if (found == watched.end()) {
        if (!collected_callback) {
            collected_callback = PyCFunction_New(&collected_def, nullptr);
            if (!collected_callback)
                throw std::runtime_error("cannot create the type collection callback");
        }
        PyObject* weak = PyWeakref_NewRef(type, collected_callback);
        if (!weak) throw std::runtime_error("cannot watch a type for collection");
        found = watched.emplace(type, Watch{weak, {}}).first;
    }
    for (auto known : found->second.forget)
        if (known == forget) return;
    found->second.forget.push_back(forget);
}

} // jittor
