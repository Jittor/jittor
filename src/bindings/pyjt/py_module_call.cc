#include "bindings/pyjt/py_module_call.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "bindings/pyjt/py_converter.h"
#include "bindings/pyjt/py_kernel_select.h"
#include "ops/op_register.h"
#include "ops/composite/code_op.h"
#include <cmath>
#include "core/grad.h"
#include "core/var_holder.h"
#include "runtime/float32_precision.h"
#include <stdexcept>

namespace jittor {

namespace {

PyObject* module_cls = nullptr;
PyObject* dispatch_call = nullptr;
PyObject* dispatch_module_call = nullptr;
PyObject* published = nullptr;
PyObject* pipeline = nullptr;
PyObject* device_context = nullptr;
PyObject* slow = nullptr;
// What `_dispatch_module_call` consults, for the native version of it below.
PyObject* prefer_forward_fn = nullptr;
PyObject* standard_rms_norm = nullptr;
PyObject* linear_execute = nullptr;
PyObject* matmul_kernel = nullptr;
PyObject* empty_kwargs = nullptr;
PyObject* matmul_name = nullptr;
auto make_cublas_matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("cublas_matmul");
// The fused RMS norm `_standard_rms_norm` reaches on an inference call.
PyObject* rms_inference_impl = nullptr;
PyObject* rms_source_fn = nullptr;
bool acl_possible = true;
PyObject* rms_training_name = nullptr;
PyObject* rms_inference_name = nullptr;
auto make_code = op_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&,
    vector<string>&&, string&&, string&&, vector<string>&&, string&&, DataMap&&, string&&>("code");
// Source per (hidden size, epsilon), as `_rms_norm_source` formats it.
std::map<pair<int64, double>, string> rms_sources;

struct TypeInfo {
    PyObject* tensor_type = nullptr;
    PyObject* native_call = nullptr;
    bool frontend_dispatch = false;
    // `_prefer_forward(type)`: the class's own forward wins over an
    // inherited execute. Asked once, as the Python side caches it per class.
    bool prefer_forward = false;
    // The class is named like an RMS norm `_standard_rms_norm` may take.
    bool rms_named = false;
};
std::unordered_map<PyTypeObject*, TypeInfo> type_info;

void assign_ref(PyObject*& slot, PyObject* value) {
    Py_XINCREF(value);
    Py_XDECREF(slot);
    slot = value;
}

const TypeInfo& info_of(PyTypeObject* type) {
    auto found = type_info.find(type);
    if (found != type_info.end()) return found->second;
    TypeInfo info;
    PyObjHolder owner(PyObject_GetAttrString((PyObject*)type, "_nn_frontend_owner"));
    info.tensor_type = PyObject_GetAttrString(owner.obj, "tensor_type");
    PyObjHolder native(PyObject_GetAttrString(owner.obj, "native_module"));
    info.native_call = PyObject_GetAttrString(native.obj, "__call__");
    if (!info.tensor_type || !info.native_call)
        throw std::runtime_error("torch frontend module type without its owner");
    PyObject* own_dispatch = PyObject_GetAttrString((PyObject*)type, "_dispatch_call");
    if (!own_dispatch) { PyErr_Clear(); }
    info.frontend_dispatch = own_dispatch && own_dispatch == dispatch_call;
    Py_XDECREF(own_dispatch);
    if (prefer_forward_fn) {
        PyObjHolder prefer(PyObject_CallFunctionObjArgs(prefer_forward_fn, (PyObject*)type, nullptr));
        info.prefer_forward = PyObject_IsTrue(prefer.obj) == 1;
    }
    string name = type->tp_name;
    auto ends_with = [&](const string& suffix) {
        return name.size() >= suffix.size()
            && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    info.rms_named = ends_with("RMSNorm") && !ends_with("RMSNormGated");
    Py_INCREF(type);
    return type_info.emplace(type, info).first->second;
}

PyObject* first_var(PyObject* args, PyObject* kwargs) {
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* value = PyTuple_GET_ITEM(args, i);
        if (PyObject_TypeCheck(value, &PyjtVarHolder.ht_type)) return value;
    }
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyObject_TypeCheck(value, &PyjtVarHolder.ht_type)) return value;
    }
    return nullptr;
}

// Whether `module` takes the frontend's own dispatch with nothing to do on
// the way there.
bool shortcut(PyObject* module, const TypeInfo& info) {
    if (!info.frontend_dispatch) return false;
    PyObject* dict = PyObject_GenericGetDict(module, nullptr);
    if (!dict) { PyErr_Clear(); return false; }
    PyObjHolder dict_holder(dict);
    for (const char* hook : {"_forward_pre_hooks", "_forward_hooks",
                             "_input_backward_hooks", "_output_backward_hooks"}) {
        PyObject* table = PyDict_GetItemString(dict, hook);
        if (table && PyObject_IsTrue(table)) return false;
    }
    if (PyDict_GetItemString(dict, "_fsdp_state")) return false;
    PyObjHolder depth(PyObject_GetAttrString(module_cls, "_call_depth"));
    if (!PyObject_IsTrue(depth.obj)) return false;
    PyObjHolder id(PyLong_FromVoidPtr(module));
    int done = PySet_Contains(published, id.obj);
    if (done <= 0) { PyErr_Clear(); return false; }
    PyObject* threshold = PyDict_GetItemString(pipeline, "threshold");
    if (!threshold || !PyLong_Check(threshold) || PyLong_AsLong(threshold) > 0) return false;
    return true;
}

PyObject* call_with(PyObject* fn, PyObject* module, PyObject* args, PyObject* kwargs) {
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    PyObjHolder full(PyTuple_New(n + 1));
    Py_INCREF(module);
    PyTuple_SET_ITEM(full.obj, 0, module);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* value = PyTuple_GET_ITEM(args, i);
        Py_INCREF(value);
        PyTuple_SET_ITEM(full.obj, i + 1, value);
    }
    return PyObject_Call(fn, full.obj, kwargs);
}

bool is_var(PyObject* value) {
    return PyObject_TypeCheck(value, &PyjtVarHolder.ht_type);
}

PyObject* interned(PyObject*& slot, const char* text) {
    if (!slot) slot = PyUnicode_InternFromString(text);
    return slot;
}
PyObject* forward_key = nullptr;
PyObject* execute_key = nullptr;
PyObject* name_forward() { return interned(forward_key, "forward"); }
PyObject* name_execute() { return interned(execute_key, "execute"); }

// `nn.Linear` without a bias, as `jittor.nn.Linear.execute` computes it --
// `matmul_transpose` straight into the cuBLAS relay -- when the dispatcher
// would pick that relay. nullptr when anything else would happen.
PyObject* linear_without_bias(PyObject* module, PyObject* dict, PyObject* x) {
    if (!matmul_kernel || !dict || !is_var(x)) return nullptr;
    PyObject* weight = PyDict_GetItemString(dict, "weight");
    PyObject* bias = PyDict_GetItemString(dict, "bias");
    if (!weight || !is_var(weight) || (bias && bias != Py_None)) return nullptr;
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    Var* w = GET_RAW_PTR(VarHolder, weight)->var;
    int rank = a->shape.size();
    if (w->shape.size() != 2 || rank < 2 || a->shape[rank - 1] != w->shape[1]) return nullptr;
    if (a->dtype() != w->dtype() || !a->dtype().is_float()) return nullptr;
    // `matmul_transpose` hands a batched input over flattened in place only
    // when its buffer is dense.
    if (rank > 2 && !a->is_contiguous()) return nullptr;
    PyObjHolder zero(PyLong_FromLong(0)), one(PyLong_FromLong(1));
    PyObjHolder call_args(PyTuple_Pack(4, x, weight, zero.obj, one.obj));
    PyObjHolder kernel(kernel_select(matmul_name, call_args.obj, nullptr));
    if (kernel.obj != matmul_kernel) return nullptr;
    PyTensorFrontendScope scope(x, nullptr, 0, false);
    return to_py_object<VarHolder*>(new VarHolder(make_cublas_matmul(a, w, false, true)));
}

// `_standard_rms_norm` on an inference call, natively: no training kernel
// takes it, the dispatcher picks `_rms_norm_cuda` for it, and that builds its
// code operator from `_rms_norm_source`. nullptr for anything else.
PyObject* rms_norm_inference(PyObject* dict, PyObject* x) {
    if (!rms_inference_impl || !rms_source_fn || acl_possible || !dict || !is_var(x))
        return nullptr;
    PyObject* weight = PyDict_GetItemString(dict, "weight");
    PyObject* eps = PyDict_GetItemString(dict, "variance_epsilon");
    if (!weight || !is_var(weight) || !eps || !(PyFloat_Check(eps) || PyLong_Check(eps)))
        return nullptr;
    double epsilon = PyFloat_AsDouble(eps);
    if (PyErr_Occurred()) { PyErr_Clear(); return nullptr; }
    PyObjHolder call_args(PyTuple_Pack(3, x, weight, eps));
    PyObjHolder training(kernel_select(rms_training_name, call_args.obj, nullptr));
    if (!training.obj || training.obj != Py_None) return nullptr;
    PyObjHolder inference(kernel_select(rms_inference_name, call_args.obj, nullptr));
    if (!inference.obj || inference.obj != rms_inference_impl) return nullptr;
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    Var* gamma = GET_RAW_PTR(VarHolder, weight)->var;
    int64 hidden = a->shape[a->shape.size() - 1];
    int64 threads = 32;
    while (threads < std::min<int64>(hidden, 1024)) threads *= 2;
    auto key = std::make_pair(hidden, epsilon);
    auto found = rms_sources.find(key);
    if (found == rms_sources.end()) {
        PyObjHolder src(PyObject_CallFunction(rms_source_fn, "LLLd",
            (long long)hidden, (long long)threads, (long long)(threads / 32), epsilon));
        Py_ssize_t size;
        const char* text = PyUnicode_AsUTF8AndSize(src.obj, &size);
        if (!text) return nullptr;
        found = rms_sources.emplace(key, string(text, size)).first;
    }
    PyTensorFrontendScope scope(x, nullptr, 0, false);
    unique_ptr<VarHolder> out(new VarHolder(make_code(a->shape, a->dtype(), {a, gamma},
        "", {}, "", string(found->second), {}, "", {}, "")));
    out->stop_grad();
    return to_py_object<VarHolder*>(out.release());
}

// `_dispatch_module_call`, for a call `shortcut` cleared: an instance-level
// forward, a standard RMS norm, the class's forward, or `execute`.
PyObject* dispatch_native(PyObject* module, const TypeInfo& info, PyObject* args, PyObject* kwargs) {
    PyObject** dict_ptr = _PyObject_GetDictPtr(module);
    PyObject* dict = dict_ptr ? *dict_ptr : nullptr;
    if (dict) {
        PyObject* own = PyDict_GetItemString(dict, "forward");
        if (own && PyCallable_Check(own)) return PyObject_Call(own, args, kwargs);
    }
    if (info.rms_named) {
        if (!kwargs && PyTuple_GET_SIZE(args) == 1) {
            PyObject* fast = rms_norm_inference(dict, PyTuple_GET_ITEM(args, 0));
            if (fast || PyErr_Occurred()) return fast;
        }
        PyObject* fused = PyObject_CallFunctionObjArgs(standard_rms_norm, module, args,
                                                       kwargs ? kwargs : empty_kwargs, nullptr);
        if (!fused || fused != Py_None) return fused;
        Py_DECREF(fused);
    }
    PyTypeObject* type = Py_TYPE(module);
    if (info.prefer_forward) {
        PyObject* forward = _PyType_Lookup(type, name_forward());
        if (forward) {
            Py_INCREF(forward);
            PyObjHolder hold(forward);
            return call_with(forward, module, args, kwargs);
        }
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && linear_execute
            && _PyType_Lookup(type, name_execute()) == linear_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* fast = linear_without_bias(module, dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    PyObjHolder execute(PyObject_GetAttr(module, name_execute()));
    return PyObject_Call(execute.obj, args, kwargs);
}

} // namespace

void module_call_bind(PyObject* module_cls_, PyObject* dispatch_call_,
                      PyObject* dispatch_module_call_, PyObject* published_,
                      PyObject* pipeline_, PyObject* device_context_, PyObject* slow_,
                      PyObject* dispatch_parts) {
    assign_ref(module_cls, module_cls_);
    assign_ref(dispatch_call, dispatch_call_);
    assign_ref(dispatch_module_call, dispatch_module_call_);
    assign_ref(published, published_);
    assign_ref(pipeline, pipeline_);
    assign_ref(device_context, device_context_);
    assign_ref(slow, slow_);
    auto part = [&](PyObject*& slot, const char* key) {
        PyObject* value = dispatch_parts && PyDict_Check(dispatch_parts)
            ? PyDict_GetItemString(dispatch_parts, key) : nullptr;
        assign_ref(slot, value == Py_None ? nullptr : value);
    };
    part(prefer_forward_fn, "prefer_forward");
    part(standard_rms_norm, "standard_rms_norm");
    part(linear_execute, "linear_execute");
    part(matmul_kernel, "matmul_kernel");
    part(rms_inference_impl, "rms_norm_inference");
    part(rms_source_fn, "rms_norm_source");
    PyObject* acl = dispatch_parts && PyDict_Check(dispatch_parts)
        ? PyDict_GetItemString(dispatch_parts, "acl_possible") : nullptr;
    acl_possible = !acl || PyObject_IsTrue(acl) != 0;
    if (!rms_training_name) rms_training_name = PyUnicode_InternFromString("nn.rms_norm.training");
    if (!rms_inference_name) rms_inference_name = PyUnicode_InternFromString("nn.rms_norm.inference");
    rms_sources.clear();
    if (!empty_kwargs) empty_kwargs = PyDict_New();
    if (!matmul_name) matmul_name = PyUnicode_InternFromString("matmul");
    type_info.clear();
}

PyObject* module_call_native(PyObject* module, PyObject* args, PyObject* kwargs) {
    if (!slow) throw std::runtime_error("the native module call is not bound");
    if (kwargs == Py_None || (kwargs && !PyDict_GET_SIZE(kwargs))) kwargs = nullptr;
    const TypeInfo& info = info_of(Py_TYPE(module));
    // Placement follows the first tensor argument; without one that has an
    // explicit placement, an active `with torch.device(...)` would decide,
    // and that resolution stays in Python.
    PyObject* like = first_var(args, kwargs);
    Var* like_var = like ? GET_RAW_PTR(VarHolder, like)->var : nullptr;
    if (!like_var || !like_var->placement.explicit_backend) {
        PyObjHolder active(PyObject_CallObject(device_context, nullptr));
        if (active.obj != Py_None) return call_with(slow, module, args, kwargs);
    }
    PyObject* type_token = set_tensor_frontend_type(info.tensor_type);
    PyObject* placement_token = nullptr;
    auto previous_precision = current_float32_precision_policy();
    int previous_policy = -1;
    PyObject* result = nullptr;
    try {
        int matmul, cudnn;
        if (!frontend_precision_tiers(info.tensor_type, matmul, cudnn)) {
            PyObjHolder tiers(PyObject_CallMethod(info.tensor_type, "_frontend_precision_policy", nullptr));
            matmul = int(PyLong_AsLong(PyTuple_GET_ITEM(tiers.obj, 0)));
            cudnn = int(PyLong_AsLong(PyTuple_GET_ITEM(tiers.obj, 1)));
        }
        set_float32_precision_policy({matmul, cudnn});
        if (like_var && like_var->placement.explicit_backend)
            placement_token = set_tensor_placement_context(
                int(like_var->placement.device.backend), std::max(like_var->device_id, 0));
        previous_policy = get_autograd_policy();
        set_autograd_policy(true, true);
        if (!shortcut(module, info))
            result = call_with(info.native_call, module, args, kwargs);
        else if (prefer_forward_fn && standard_rms_norm)
            result = dispatch_native(module, info, args, kwargs);
        else
            result = call_with(dispatch_module_call, module, args, kwargs);
    } catch (...) {
        if (previous_policy >= 0) set_autograd_policy(previous_policy & 1, previous_policy & 2);
        set_float32_precision_policy(previous_precision);
        if (placement_token) { reset_tensor_placement_context(placement_token); Py_DECREF(placement_token); }
        reset_tensor_frontend_type(type_token);
        Py_DECREF(type_token);
        throw;
    }
    PyObject *error_type, *error_value, *error_tb;
    PyErr_Fetch(&error_type, &error_value, &error_tb);
    set_autograd_policy(previous_policy & 1, previous_policy & 2);
    set_float32_precision_policy(previous_precision);
    if (placement_token) { reset_tensor_placement_context(placement_token); Py_DECREF(placement_token); }
    reset_tensor_frontend_type(type_token);
    Py_DECREF(type_token);
    PyErr_Restore(error_type, error_value, error_tb);
    return result;
}

} // namespace jittor
