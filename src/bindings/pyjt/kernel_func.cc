#include "bindings/pyjt/kernel_func.h"
#include "bindings/pyjt/py_obj_holder.h"
#include "core/var.h"
#include "runtime/async_exec.h"

namespace jittor {
namespace {

constexpr const char* kActiveLaunchScopeCapsule = "jittor_kernel_launch_scope";
constexpr const char* kExpiredLaunchScopeCapsule = "jittor_expired_kernel_launch_scope";

struct ActiveLaunchScope {
    const vector<Var*>* inputs = nullptr;
    const vector<Var*>* outputs = nullptr;
    bool active = false;
};

struct LaunchScopeGuard {
    ActiveLaunchScope* scope = nullptr;
    PyObject* capsule = nullptr; // borrowed; owned by PyObjHolder declared before guard
    ~LaunchScopeGuard() {
        if (scope) scope->active = false;
        if (capsule) {
            PyObject *err_type = nullptr, *err_value = nullptr, *err_tb = nullptr;
            PyErr_Fetch(&err_type, &err_value, &err_tb);
            if (PyCapsule_SetName(capsule, kExpiredLaunchScopeCapsule) < 0)
                PyErr_Clear();
            PyErr_Restore(err_type, err_value, err_tb);
        }
    }
};

static string format_and_clear_python_error() {
    PyObject *type = nullptr, *value = nullptr, *tb = nullptr;
    PyErr_Fetch(&type, &value, &tb);
    if (!type && !value) return "Python callback failed";
    PyErr_NormalizeException(&type, &value, &tb);

    string type_name;
    if (type) {
        if (PyType_Check(type)) {
            const char* name = reinterpret_cast<PyTypeObject*>(type)->tp_name;
            if (name) type_name = name;
        } else {
            PyObject* s = PyObject_Str(type);
            if (s) {
                const char* utf8 = PyUnicode_AsUTF8(s);
                if (utf8) type_name = utf8;
                Py_DECREF(s);
            } else {
                PyErr_Clear();
            }
        }
    }

    string message;
    if (value) {
        PyObject* s = PyObject_Str(value);
        if (s) {
            const char* utf8 = PyUnicode_AsUTF8(s);
            if (utf8) message = utf8;
            Py_DECREF(s);
        } else {
            PyErr_Clear();
        }
    }

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(tb);
    PyErr_Clear();

    if (!type_name.empty() && !message.empty()) return type_name + ": " + message;
    if (!type_name.empty()) return type_name;
    if (!message.empty()) return message;
    return "Python callback failed";
}

static PyObject* buffer_info(Var* var, PyObject* scope_capsule, bool is_output, int index) {
    PyObjHolder shape(PyTuple_New(var->shape.size()));
    PyObjHolder strides(PyTuple_New(var->shape.size()));
    for (int i = 0; i < var->shape.size(); ++i) {
        PyObject* s = PyLong_FromLongLong(var->shape[i]);
        PyObject* st = PyLong_FromLongLong(var->storage_stride(i));
        if (!s || !st) {
            Py_XDECREF(s);
            Py_XDECREF(st);
            USER_ERROR << "external_kernel buffer metadata failed:"
                       << format_and_clear_python_error();
        }
        PyTuple_SET_ITEM(shape.obj, i, s);
        PyTuple_SET_ITEM(strides.obj, i, st);
    }
    // Defer DLPack DLManagedTensor/VarPtr allocation until __dlpack__() is called.
    PyObjHolder lazy_dlpack(Py_BuildValue("(Oii)", scope_capsule, is_output ? 1 : 0, index));
    PyObject* tuple = Py_BuildValue("(KOOsO)", (unsigned long long)var->mem_ptr,
        shape.obj, strides.obj, var->dtype().to_cstring(), lazy_dlpack.obj);
    if (!tuple) {
        USER_ERROR << "external_kernel buffer view creation failed:"
                   << format_and_clear_python_error();
    }
    return tuple;
}

static PyObject* buffer_list(const vector<Var*>& vars, PyObject* scope_capsule, bool is_output) {
    PyObjHolder list(PyTuple_New(vars.size()));
    for (size_t i = 0; i < vars.size(); ++i)
        PyTuple_SET_ITEM(list.obj, i, buffer_info(vars[i], scope_capsule, is_output, (int)i));
    return list.release();
}

} // namespace

PyObject* kernel_buffer_dlpack(PyObject* scope_capsule, bool is_output, int index) {
    USER_CHECK(PyCapsule_IsValid(scope_capsule, kActiveLaunchScopeCapsule))
        << "external_kernel: kernel buffer view cannot export DLPack after callback exit";
    auto* scope = static_cast<ActiveLaunchScope*>(
        PyCapsule_GetPointer(scope_capsule, kActiveLaunchScopeCapsule));
    USER_CHECK(scope && scope->active)
        << "external_kernel: kernel buffer view cannot export DLPack after callback exit";
    const vector<Var*>& vec = is_output ? *scope->outputs : *scope->inputs;
    USER_CHECK(index >= 0 && (size_t)index < vec.size())
        << "external_kernel: kernel buffer index out of range";
    PyObject* capsule = kernel_buffer_capsule(vec[index]);
    if (!capsule) {
        USER_ERROR << "external_kernel: failed to export DLPack capsule:"
                   << format_and_clear_python_error();
    }
    return capsule;
}

KernelFunc kernel_func_from_python(PyObject* callable) {
    Py_INCREF(callable);
    return KernelFunc([callable](const KernelLaunch& launch) {
        // external_kernel is excluded from worker batches, as are fetch and
        // numpy_code. No Python callback runs while the worker owns execution.
        ASSERT(PyGILState_Check());
        ActiveLaunchScope scope{&launch.inputs, &launch.outputs, true};
        PyObjHolder scope_cap(PyCapsule_New(&scope, kActiveLaunchScopeCapsule, nullptr));
        LaunchScopeGuard guard{&scope, scope_cap.obj};

        PyObjHolder inputs(buffer_list(launch.inputs, scope_cap.obj, false));
        PyObjHolder outputs(buffer_list(launch.outputs, scope_cap.obj, true));
        PyObjHolder args(Py_BuildValue("(OOs iL)", inputs.obj, outputs.obj,
            launch.device.backend == BackendId::Cuda ? "cuda" : "cpu",
            launch.device.index, (long long)launch.stream));
        PyObject* result = PyObject_CallObject(callable, args.obj);
        if (!result) {
            string err = format_and_clear_python_error();
            USER_ERROR << "external_kernel callback failed:" << err;
        }
        Py_DECREF(result);
    }, [callable]() {
        auto drop = [callable]() { Py_DECREF(callable); };
        if (!async_exec_defer_python_release(drop)) drop();
    });
}
} // namespace jittor
