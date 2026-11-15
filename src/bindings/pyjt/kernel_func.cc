#include "bindings/pyjt/kernel_func.h"
#include "bindings/pyjt/py_obj_holder.h"
#include "core/var.h"
#include "runtime/async_exec.h"

namespace jittor {
static PyObject* buffer_info(Var* var) {
    PyObjHolder shape(PyTuple_New(var->shape.size()));
    PyObjHolder strides(PyTuple_New(var->shape.size()));
    for (int i = 0; i < var->shape.size(); ++i) {
        PyTuple_SET_ITEM(shape.obj, i, PyLong_FromLongLong(var->shape[i]));
        PyTuple_SET_ITEM(strides.obj, i, PyLong_FromLongLong(var->storage_stride(i)));
    }
    PyObjHolder capsule(kernel_buffer_capsule(var));
    return Py_BuildValue("(KOOsO)", (unsigned long long)var->mem_ptr,
        shape.obj, strides.obj, var->dtype().to_cstring(), capsule.obj);
}

static PyObject* buffer_list(const vector<Var*>& vars) {
    PyObjHolder list(PyTuple_New(vars.size()));
    for (size_t i = 0; i < vars.size(); ++i)
        PyTuple_SET_ITEM(list.obj, i, buffer_info(vars[i]));
    return list.release();
}

KernelFunc kernel_func_from_python(PyObject* callable) {
    Py_INCREF(callable);
    return KernelFunc([callable](const KernelLaunch& launch) {
        // external_kernel is excluded from worker batches, as are fetch and
        // numpy_code. No Python callback runs while the worker owns execution.
        ASSERT(PyGILState_Check());
        PyObjHolder inputs(buffer_list(launch.inputs));
        PyObjHolder outputs(buffer_list(launch.outputs));
        PyObjHolder args(Py_BuildValue("(OOs iL)", inputs.obj, outputs.obj,
            launch.device.backend == BackendId::Cuda ? "cuda" : "cpu",
            launch.device.index, (long long)launch.stream));
        PyObjHolder result(PyObject_CallObject(callable, args.obj));
    }, [callable]() {
        auto drop = [callable]() { Py_DECREF(callable); };
        if (!async_exec_defer_python_release(drop)) drop();
    });
}
} // namespace jittor
