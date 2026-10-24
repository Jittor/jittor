#include "bindings/pyjt/py_tensor_frontend.h"
#include "bindings/pyjt/py_type_lifetime.h"
#include "core/grad.h"
#include "core/var_holder.h"
#include "bindings/pyjt/py_converter.h"
#include "runtime/device.h"
#include <stdexcept>
#include <structmember.h>
#include <unordered_set>
#include <climits>

namespace jittor {

extern PyHeapTypeObject PyjtVarHolder;

namespace {
// GIL-held definition/lifetime state; only the factory below can add roots.
const PyType_Slot* holder_binding_definitions = nullptr;
size_t holder_binding_definition_count = 0;
std::unordered_set<PyTypeObject*> owned_tensor_roots;
void forget_owned_tensor_root(PyObject* type) {
    owned_tensor_roots.erase(reinterpret_cast<PyTypeObject*>(type));
}
constexpr size_t holder_dict_offset = sizeof(PyObject) + sizeof(VarHolder);
constexpr size_t holder_object_size = holder_dict_offset + sizeof(PyObject*) + sizeof(uint64);
bool owned_prefix_layout(PyTypeObject* type) {
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 11 && PY_MICRO_VERSION == 13
    if (Py_Version != 0x030B0DF0UL) return false;
    if (PyInterpreterState_GetID(PyInterpreterState_Get()) != 0) return false;
    return type->tp_basicsize >= Py_ssize_t(holder_object_size)
        && type->tp_itemsize == 0 && type->tp_dictoffset == Py_ssize_t(holder_dict_offset)
        && type->tp_alloc == PyType_GenericAlloc && type->tp_free == PyObject_GC_Del
        && (type->tp_weaklistoffset == 0
            || (type->tp_weaklistoffset >= Py_ssize_t(holder_object_size)
                && type->tp_weaklistoffset <= type->tp_basicsize - Py_ssize_t(sizeof(PyObject*))))
        && (type->tp_flags & Py_TPFLAGS_HAVE_GC)
        && !(type->tp_flags & Py_TPFLAGS_MANAGED_DICT);
#else
    return false;
#endif
}
PyObject* owned_carrier_new(PyTypeObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_TypeError, "inactive TensorBase supports core result construction only");
    return nullptr;
}
int owned_carrier_init(PyObject*, PyObject*, PyObject*) {
    PyErr_SetString(PyExc_TypeError, "inactive TensorBase cannot be reinitialized");
    return -1;
}
} // namespace

void record_varholder_binding_definitions(const PyType_Slot* slots, size_t count) {
    USER_CHECK(slots && count && slots[count-1].slot == 0)
        << "complete native VarHolder binding definitions required";
    if (holder_binding_definitions) {
        USER_CHECK(slots == holder_binding_definitions && count == holder_binding_definition_count)
            << "native definition owner cannot be replaced";
        return;
    }
    holder_binding_definitions = slots; // static generated definitions, no type/dict copy
    holder_binding_definition_count = count;
}

bool is_owned_tensor_carrier_type(PyTypeObject* type) {
    if (!type || !owned_prefix_layout(type)) return false;
    for (auto* root : owned_tensor_roots)
        if (PyType_IsSubtype(type, root)) return true;
    return false;
}

bool is_varholder_carrier_type(PyTypeObject* type) {
    return type && (PyType_IsSubtype(type, &PyjtVarHolder.ht_type)
        || is_owned_tensor_carrier_type(type));
}

bool is_varholder_carrier(PyObject* object) {
    if (!object) return false;
    if (PyObject_TypeCheck(object, &PyjtVarHolder.ht_type)) return true;
    return is_owned_tensor_carrier_type(Py_TYPE(object))
        && GET_INITED_FLAG(VarHolder, 1, object) == 1;
}

PyObject* create_owned_tensor_base_type() {
#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 11 || PY_MICRO_VERSION != 13
    PyErr_SetString(PyExc_NotImplementedError, "owned TensorBase requires reviewed CPython 3.11.13");
    return nullptr;
#else
    if (Py_Version != 0x030B0DF0UL) {
        PyErr_SetString(PyExc_NotImplementedError, "owned TensorBase requires runtime CPython 3.11.13 final");
        return nullptr;
    }
    if (PyInterpreterState_GetID(PyInterpreterState_Get()) != 0) {
        PyErr_SetString(PyExc_NotImplementedError, "owned TensorBase does not support subinterpreters");
        return nullptr;
    }
    USER_CHECK(holder_binding_definitions) << "native VarHolder definitions not initialized";
    static_assert(holder_object_size <= INT_MAX, "PyType_Spec basicsize overflow");
    static PyMemberDef members[] = {
        {const_cast<char*>("__dictoffset__"), T_PYSSIZET, Py_ssize_t(holder_dict_offset), READONLY, nullptr},
        {nullptr, 0, 0, 0, nullptr}
    };
    vector<PyType_Slot> slots(holder_binding_definitions,
        holder_binding_definitions + holder_binding_definition_count);
    slots.pop_back();
    // No Python path may expose an uninitialized native prefix. Core result
    // conversion below placement-news exactly once, bypassing these slots.
    for (auto& slot : slots) {
        if (slot.slot == Py_tp_new) slot.pfunc = reinterpret_cast<void*>(owned_carrier_new);
        if (slot.slot == Py_tp_init) slot.pfunc = reinterpret_cast<void*>(owned_carrier_init);
    }
    slots.push_back({Py_tp_members, members});
    slots.push_back({0, nullptr});
    PyType_Spec spec = {"torch._C.TensorBase", int(holder_object_size), 0,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
        slots.data()};
    // CPython3.11 copies member definitions and owns its new heap type/name.
    PyObjHolder result;
    result.obj = PyType_FromSpec(&spec);
    if (!result.obj) return nullptr;
    auto* type = reinterpret_cast<PyTypeObject*>(result.obj);
    USER_CHECK(type->tp_base == &PyBaseObject_Type && PyTuple_GET_SIZE(type->tp_bases) == 1
        && PyTuple_GET_ITEM(type->tp_bases, 0) == reinterpret_cast<PyObject*>(&PyBaseObject_Type)
        && PyTuple_GET_SIZE(type->tp_mro) == 2
        && PyTuple_GET_ITEM(type->tp_mro, 0) == result.obj
        && PyTuple_GET_ITEM(type->tp_mro, 1) == reinterpret_cast<PyObject*>(&PyBaseObject_Type)
        && owned_prefix_layout(type))
        << "owned TensorBase physical root/layout mismatch";
    // Watch BEFORE membership: any failure leaves no stale admitted pointer.
    on_type_collected(result.obj, forget_owned_tensor_root);
    owned_tensor_roots.insert(type);
    return result.release();
#endif
}

PyObject* wrap_owned_tensor_carrier(VarHolder* value, PyTypeObject* type) {
    unique_ptr<VarHolder> pending(value);
    USER_CHECK(value && value->var) << "constructed native holder payload required";
    USER_CHECK(is_owned_tensor_carrier_type(type)) << "core-created TensorBase carrier required";
    PyObjHolder result;
    result.obj = type->tp_alloc(type, 0);
    if (!result.obj) {
        PyObject *error_type, *error_value, *error_traceback;
        PyErr_Fetch(&error_type, &error_value, &error_traceback);
        pending.reset(); // release graph while preserving the allocation error
        if (PyErr_Occurred()) PyErr_WriteUnraisable(nullptr);
        PyErr_Restore(error_type, error_value, error_traceback);
        return nullptr;
    }
    auto* holder = GET_RAW_PTR(VarHolder, result.obj);
    new (holder) VarHolder(pending.get()); // existing non-allocating holder move
    pending.release(); // move constructor has released old wrapper storage
    GET_INITED_FLAG(VarHolder, 1, result.obj) = 1;
    bind_python_tensor_owner(result.obj);
    schedule_pending_from_python(holder);
    return result.release();
}

void bind_python_tensor_owner(PyObject* object) {
    bool retain = false;
    if (Py_TYPE(object) != &PyjtVarHolder.ht_type) {
        PyObjHolder requested;
        requested.obj = PyObject_GetAttrString(
            reinterpret_cast<PyObject*>(Py_TYPE(object)),
            "_frontend_view_base_ownership");
        if (!requested.obj) {
            if (!PyErr_ExceptionMatches(PyExc_AttributeError))
                throw std::runtime_error("cannot read frontend view ownership");
            PyErr_Clear();
        } else {
            if (!PyBool_Check(requested.obj))
                throw std::runtime_error("frontend view ownership must be a bool");
            retain = requested.obj == Py_True;
        }
    }
    GET_RAW_PTR(VarHolder, object)->bind_python_owner(object, retain);
}

namespace {
// One shared symbol owner across the core and all loaded JIT extensions.
// Access occurs with the GIL held, just like the Python conversion boundary.
PyObject* frontend_context = nullptr;
PyObject* placement_context = nullptr;
// Whether either variable has ever been given a value in this process. Every
// binding call reads both -- the result type, the construction placement --
// and a program that never selects a frontend or a placement (plain Jittor)
// paid two context lookups per operator for an answer that cannot be anything
// but "unset": until one of the setters below runs, no context holds a value.
bool frontend_context_set = false;
bool placement_context_set = false;

PyObject* placement_variable() {
    if (!placement_context) {
        placement_context = PyContextVar_New("jittor.tensor_placement", nullptr);
        if (!placement_context)
            throw std::runtime_error("cannot create tensor placement context");
    }
    return placement_context;
}

TensorPlacement selected_placement() {
    if (!placement_context_set) return {};
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
    if (!frontend_context_set) return nullptr;
    PyObject* value = nullptr;
    if (PyContextVar_Get(context_variable(), nullptr, &value) < 0)
        throw std::runtime_error("cannot read tensor frontend context");
    return value;
}

bool is_frontend_instance(PyObject* value) {
    return value && Py_TYPE(value) != &PyjtVarHolder.ht_type
        && is_varholder_carrier(value);
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
        && is_varholder_carrier_type(reinterpret_cast<PyTypeObject*>(type)))
        << "tensor frontend must be a core-owned VarHolder carrier type";
    frontend_context_set = true;
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
    placement_context_set = true;
    PyObject* token = PyContextVar_Set(placement_variable(), value);
    Py_DECREF(value);
    if (!token) throw std::runtime_error("cannot set tensor placement context");
    return token;
}

void reset_tensor_placement_context(PyObject* token) {
    if (PyContextVar_Reset(placement_variable(), token) < 0)
        throw std::runtime_error("cannot reset tensor placement context");
}

PyObject* set_construction_placement(int backend, int device) {
    USER_CHECK(backend >= 0 && backend <= int(BackendId::Corex) && device >= 0)
        << "construction placement requires a registered backend id and a non-negative device index";
    TensorPlacement previous = current_tensor_placement();
    PyObject* token = Py_BuildValue("(iii)", int(previous.explicit_backend),
                                    int(previous.device.backend), previous.device.index);
    if (!token) throw std::runtime_error("cannot create construction placement token");
    set_tensor_placement(TensorPlacement({static_cast<BackendId>(backend), backend == 0 ? 0 : device}));
    return token;
}

void restore_construction_placement(PyObject* previous) {
    USER_CHECK(PyTuple_Check(previous) && PyTuple_GET_SIZE(previous) == 3)
        << "_restore_construction_placement takes the token _set_construction_placement returned";
    if (!PyLong_AsLong(PyTuple_GET_ITEM(previous, 0))) {
        set_tensor_placement(TensorPlacement());
        return;
    }
    set_tensor_placement(TensorPlacement({
        static_cast<BackendId>(PyLong_AsLong(PyTuple_GET_ITEM(previous, 1))),
        int(PyLong_AsLong(PyTuple_GET_ITEM(previous, 2)))}));
}

TensorPlacement frontend_placement_request() { return selected_placement(); }

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

namespace {
PyObject* frontend_result_type(PyObject* actual_type);
} // namespace

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
    PyObject* result_type = frontend_result_type(reinterpret_cast<PyObject*>(Py_TYPE(candidate)));
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
// Dropped when the type is collected.
unordered_map<PyObject*, FrontendPolicy> frontend_policies;
uint64 frontend_policy_epoch = 1;

void forget_frontend_policy(PyObject* type) { frontend_policies.erase(type); }

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
        on_type_collected(type, forget_frontend_policy);
        return frontend_policies.emplace(type, policy).first->second;
    }
    found->second = policy;
    return found->second;
}

// `type._frontend_result_type`, or `type` itself, as a new reference. Asked
// for every native call made through a frontend tensor outside a frontend
// scope -- and a type without the attribute answered with an AttributeError
// whose message was formatted each time -- so it is kept per type, under the
// same epoch as the policies, until the type is collected. A type that is its
// own result type is not held by its entry (`type` is null).
struct ResultType {
    PyObject* type = nullptr;
    uint64 epoch = 0;
};
unordered_map<PyObject*, ResultType> result_types;

void forget_result_type(PyObject* type) {
    auto found = result_types.find(type);
    if (found == result_types.end()) return;
    PyObject* held = found->second.type;
    result_types.erase(found);
    Py_XDECREF(held);
}

PyObject* frontend_result_type(PyObject* actual_type) {
    auto found = result_types.find(actual_type);
    if (found != result_types.end() && found->second.epoch == frontend_policy_epoch) {
        PyObject* result_type = found->second.type ? found->second.type : actual_type;
        Py_INCREF(result_type);
        return result_type;
    }
    PyObject* result_type = PyObject_GetAttrString(actual_type, "_frontend_result_type");
    if (!result_type) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError))
            throw std::runtime_error("cannot read tensor frontend result type");
        PyErr_Clear();
        result_type = actual_type;
        Py_INCREF(result_type);
    }
    if (found == result_types.end()) {
        try {
            on_type_collected(actual_type, forget_result_type);
        } catch (...) {
            Py_DECREF(result_type);
            throw;
        }
        found = result_types.emplace(actual_type, ResultType()).first;
    }
    PyObject* previous = found->second.type;
    PyObject* held = result_type == actual_type ? nullptr : result_type;
    Py_XINCREF(held);
    found->second = {held, frontend_policy_epoch};
    Py_XDECREF(previous);
    return result_type;
}
} // namespace

void invalidate_frontend_policies() { frontend_policy_epoch++; }

bool frontend_precision_tiers(PyObject* type, int& matmul, int& cudnn) {
    const FrontendPolicy& policy = frontend_policy(type);
    if (!policy.has_precision) return false;
    matmul = int(policy.matmul);
    cudnn = int(policy.cudnn);
    return true;
}

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
