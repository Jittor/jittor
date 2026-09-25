#include "bindings/pyjt/py_tensor_frontend.h"
#include "core/grad.h"
#include "core/var_holder.h"
#include "bindings/pyjt/py_converter.h"
#include "runtime/device.h"
#include <stdexcept>

namespace jittor {

extern PyHeapTypeObject PyjtVarHolder;

namespace {
// One shared symbol owner across the core and all loaded JIT extensions.
// Access occurs with the GIL held, just like the Python conversion boundary.
PyObject* frontend_context = nullptr;
PyObject* placement_context = nullptr;

PyObject* placement_variable() {
    if (!placement_context) {
        placement_context = PyContextVar_New("jittor.tensor_placement", nullptr);
        if (!placement_context)
            throw std::runtime_error("cannot create tensor placement context");
    }
    return placement_context;
}

TensorPlacement selected_placement() {
    PyObject* value = nullptr;
    if (PyContextVar_Get(placement_variable(), nullptr, &value) < 0)
        throw std::runtime_error("cannot read tensor placement context");
    if (!value) return {};
    int backend = int(PyLong_AsLong(PyTuple_GET_ITEM(value, 0)));
    int index = int(PyLong_AsLong(PyTuple_GET_ITEM(value, 1)));
    Py_DECREF(value);
    return TensorPlacement({static_cast<BackendId>(backend), index});
}

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

PyObject* set_tensor_placement_context(int backend, int device) {
    USER_CHECK(backend >= 0 && backend <= int(BackendId::Corex) && device >= 0)
        << "tensor placement requires a registered backend id and a non-negative device index";
    PyObject* value = Py_BuildValue("(ii)", backend, backend == 0 ? 0 : device);
    if (!value) throw std::runtime_error("cannot create tensor placement value");
    PyObject* token = PyContextVar_Set(placement_variable(), value);
    Py_DECREF(value);
    if (!token) throw std::runtime_error("cannot set tensor placement context");
    return token;
}

void reset_tensor_placement_context(PyObject* token) {
    if (PyContextVar_Reset(placement_variable(), token) < 0)
        throw std::runtime_error("cannot reset tensor placement context");
}

PyObject* current_tensor_placement_request() {
    TensorPlacement placement = selected_placement();
    if (!placement.explicit_backend) Py_RETURN_NONE;
    return Py_BuildValue("(ii)", int(placement.device.backend), placement.device.index);
}

PyObject* set_float32_precision_context(int matmul, int cudnn) {
    USER_CHECK(matmul >= 0 && matmul <= 2 && cudnn >= 0 && cudnn <= 2)
        << "frontend precision tiers must be in [0, 2]";
    auto previous = current_float32_precision_policy();
    PyObject* token = Py_BuildValue("(ii)", previous.matmul, previous.cudnn);
    if (!token) throw std::runtime_error("cannot create precision scope token");
    set_float32_precision_policy({matmul, cudnn});
    return token;
}

void reset_float32_precision_context(PyObject* token) {
    int matmul, cudnn;
    if (!PyArg_ParseTuple(token, "ii", &matmul, &cudnn))
        throw std::runtime_error("invalid precision scope token");
    USER_CHECK(matmul >= -1 && matmul <= 2 && cudnn >= -1 && cudnn <= 2)
        << "invalid previous precision tiers";
    set_float32_precision_policy({matmul, cudnn});
}

void PyTensorFrontendScope::select(
    PyObject* self, PyObject** args, int64 count, bool scan_sequences) {
    // Factories may have no tensor inputs. Their explicit context still owns
    // both the Python result type and the native autograd policy for this call.
    PyObject* candidate = is_frontend_instance(self) ? self : nullptr;
    for (int64 i = 0; !candidate && args && i < count; ++i)
        candidate = frontend_candidate(args[i], scan_sequences);
    PyObject* existing = selected_type();
    if (existing) {
        try {
            apply_policy(existing, candidate);
        } catch (...) {
            Py_DECREF(existing);
            throw;
        }
        Py_DECREF(existing);
        return;
    }
    if (!candidate) return;
    PyObject* actual_type = reinterpret_cast<PyObject*>(Py_TYPE(candidate));
    PyObject* result_type = PyObject_GetAttrString(actual_type, "_frontend_result_type");
    if (!result_type) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            throw std::runtime_error("cannot read tensor frontend result type");
        PyErr_Clear();
        result_type = actual_type;
        Py_INCREF(result_type);
    }
    try {
        // Parameters retain their Python identity when explicitly constructed,
        // while their operations may request ordinary frontend Tensor results.
        // The shared setter validates a marker just as strictly as an explicit
        // frontend selection before changing the context.
        token_ = set_tensor_frontend_type(result_type);
        apply_policy(result_type, candidate);
    } catch (...) {
        Py_DECREF(result_type);
        throw;
    }
    Py_DECREF(result_type);
}

namespace {
// What apply_policy reads from a frontend type. Every native call made through
// a frontend tensor enters a scope that needs it, and reading it through the
// type's attributes -- a string attribute lookup, a second one, and a call into
// a Python function for the precision tiers -- was 13-16% of all host time in
// a diffusers sampling loop, a DDPM training step and a Qwen3 decode alike.
// It changes only when the precision policy is set, which is rare, so it is
// read once per type and kept until `invalidate_frontend_policies`.
struct FrontendPolicy {
    bool has_autograd = false;
    long bits = 0;
    bool has_precision = false;
    long matmul = 0, cudnn = 0;
    uint64 epoch = 0;
};
// Guarded by the GIL: apply_policy runs with it held (it may call Python).
unordered_map<PyObject*, FrontendPolicy> frontend_policies;
uint64 frontend_policy_epoch = 1;

FrontendPolicy read_frontend_policy(PyObject* type) {
    FrontendPolicy policy;
    policy.epoch = frontend_policy_epoch;
    PyObject* value = PyObject_GetAttrString(type, "_frontend_autograd_policy");
    if (!value) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
            return policy;
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
    policy.has_autograd = true;
    policy.bits = bits;
    PyObject* precision_getter = PyObject_GetAttrString(type, "_frontend_precision_policy");
    if (!precision_getter) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) { PyErr_Clear(); return policy; }
        throw std::runtime_error("cannot read frontend precision policy");
    }
    PyObject* precision = PyObject_CallObject(precision_getter, nullptr);
    Py_DECREF(precision_getter);
    if (!precision) throw std::runtime_error("cannot resolve frontend precision policy");
    bool valid = PyTuple_Check(precision) && PyTuple_GET_SIZE(precision) == 2
        && PyLong_Check(PyTuple_GET_ITEM(precision, 0)) && PyLong_Check(PyTuple_GET_ITEM(precision, 1));
    if (!valid) {
        Py_DECREF(precision);
        USER_CHECK(valid) << "frontend precision policy requires two integer tiers";
    }
    long matmul = PyLong_AsLong(PyTuple_GET_ITEM(precision, 0));
    long cudnn = PyLong_AsLong(PyTuple_GET_ITEM(precision, 1));
    Py_DECREF(precision);
    if (PyErr_Occurred()) throw std::runtime_error("invalid frontend precision tier");
    USER_CHECK(matmul >= 0 && matmul <= 2 && cudnn >= 0 && cudnn <= 2)
        << "frontend precision tiers must be in [0, 2]";
    policy.has_precision = true;
    policy.matmul = matmul;
    policy.cudnn = cudnn;
    return policy;
}

const FrontendPolicy& frontend_policy(PyObject* type) {
    auto found = frontend_policies.find(type);
    if (found != frontend_policies.end() && found->second.epoch == frontend_policy_epoch)
        return found->second;
    FrontendPolicy policy = read_frontend_policy(type);
    if (found == frontend_policies.end()) {
        // Held so the address cannot be reused by another type.
        Py_INCREF(type);
        return frontend_policies.emplace(type, policy).first->second;
    }
    found->second = policy;
    return found->second;
}
} // namespace

void invalidate_frontend_policies() { frontend_policy_epoch++; }

void PyTensorFrontendScope::apply_policy(PyObject* type, PyObject* candidate) {
    const FrontendPolicy policy = frontend_policy(type);
    if (!policy.has_autograd) return;
    long bits = policy.bits;
    previous_policy_ = get_autograd_policy();
    set_autograd_policy((bits & 1) != 0, (bits & 2) != 0);
    TensorPlacement placement = selected_placement();
    if (!placement.explicit_backend && candidate && GET_INITED_FLAG(VarHolder, 1, candidate))
        placement = GET_RAW_PTR(VarHolder, candidate)->var->placement;
    if (!placement.explicit_backend) placement = current_tensor_placement();
    if (!placement.explicit_backend)
        placement = TensorPlacement({runtime_use_cuda() ? accelerator_backend_id() : BackendId::Cpu,
                                     runtime_use_cuda() ? current_device() : 0});
    previous_placement_ = current_tensor_placement();
    restore_placement_ = true;
    set_tensor_placement(placement);
    if (!policy.has_precision) return;
    previous_precision_ = current_float32_precision_policy();
    restore_precision_ = true;
    set_float32_precision_policy({int(policy.matmul), int(policy.cudnn)});
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
    if (!token_ && previous_policy_ < 0 && !restore_placement_ && !restore_precision_) return;
    PyObject *error_type = nullptr, *error_value = nullptr, *error_traceback = nullptr;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    if (restore_precision_) {
        set_float32_precision_policy(previous_precision_);
        restore_precision_ = false;
    }
    if (restore_placement_) {
        set_tensor_placement(previous_placement_);
        restore_placement_ = false;
    }
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
