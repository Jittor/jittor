#include "bindings/pyjt/py_module_call.h"
#include "bindings/pyjt/py_type_lifetime.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "bindings/pyjt/py_converter.h"
#include "bindings/pyjt/py_kernel_select.h"
#include "bindings/pyjt/py_compat_fast.h"
#include "ops/op_register.h"
#include "ops/composite/code_op.h"
#include <cmath>
#include "core/grad.h"
#include "core/var_holder.h"
#include "runtime/float32_precision.h"
#include <stdexcept>
#include <tuple>
#include "runtime/device_state.h"
#include "ops/layout_propagation.h"
#include "runtime/async_exec.h"
#include "core/executor.h"

namespace jittor {

DECLARE_FLAG(bool, no_grad);
DECLARE_FLAG(int, amp_reg);
DECLARE_FLAG(int, auto_mixed_precision_level);


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
// `lt_linear_cuda`'s header and its (cached) source function, and the sources
// it gave, per (rows, cin, cout, half precision).
// `jittor.nn.Conv.execute` and what `select_kernel("conv2d")` answers with
// cuDNN (`_try_cudnn_conv2d`); the cuDNN backend module, for its
// `channels_last_activations` switch; and the key a weight keeps its OHWI
// filter under. See `conv2d_inference`.
PyObject* conv_execute = nullptr;
PyObject* conv_cudnn_kernel = nullptr;
PyObject* cudnn_backend = nullptr;
PyObject* conv_filter_key = nullptr;
PyObject* conv2d_name = nullptr;
PyObject* depthwise_kwargs = nullptr;
// `jittor.nn.BatchNorm.execute` and the key the tracked variance keeps the
// inference scale and shift under. See `batch_norm_eval_channels_last`.
PyObject* batch_norm_execute = nullptr;
PyObject* bn_coefficients_key = nullptr;
// `jittor.nn.Dropout.execute`, which hands its input back when it is not
// training (or p is 0): see `dropout_passthrough`.
PyObject* dropout_execute = nullptr;
// `_FunctionModule.execute`, which `nn.ReLU` runs; `jittor.nn` and the
// `relu` it is expected to find there; the activation offers a pass that
// makes the input may have left (`_RESIDUAL_OFFERS`); and the warnings
// `_arg_policy` has given. See `relu_module`.
PyObject* function_module_execute = nullptr;
PyObject* nn_namespace = nullptr;
PyObject* relu_function = nullptr;
PyObject* silu_function = nullptr;
PyObject* residual_offers = nullptr;
PyObject* arg_policy_warned = nullptr;
PyObject* function_name_key = nullptr;
PyObject* fuse_activation_key = nullptr;
// `jittor.nn.GroupNorm.execute`, `jittor.nn.group_norm` as bound, the kernel
// `select_kernel("nn.group_norm")` answers with by default (the backend hook
// `group_norm` calls first is that same selection), `_group_norm_nhwc_source`,
// the activations it can fuse, and `functools.partial`. See
// `group_norm_inference`.
PyObject* group_norm_execute = nullptr;
PyObject* group_norm_function = nullptr;
PyObject* group_norm_kernel = nullptr;
PyObject* group_norm_source_fn = nullptr;
PyObject* group_norm_activations = nullptr;
PyObject* group_norm_build_fn = nullptr;
PyObject* partial_type = nullptr;
PyObject* group_norm_name = nullptr;
PyObject* lt_linear_header = nullptr;
PyObject* lt_linear_source_fn = nullptr;
std::map<std::tuple<int64, int64, int64, bool>, string> lt_linear_sources;
string lt_linear_header_text;
auto make_code = op_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&,
    vector<string>&&, string&&, string&&, vector<string>&&, string&&, DataMap&&, string&&>("code");
// Source per (hidden size, epsilon), as `_rms_norm_source` formats it.
std::map<pair<int64, double>, string> rms_sources;
// `jittor.nn.LayerNorm.execute`, the implementation registered for
// "nn.layer_norm.inference" and `layer_norm_cuda._affine_source`; see
// `layer_norm_inference`.
PyObject* layer_norm_execute = nullptr;
PyObject* layer_norm_impl = nullptr;
PyObject* layer_norm_source_fn = nullptr;
PyObject* layer_norm_name = nullptr;
std::map<std::tuple<int64, double, bool>, string> layer_norm_sources;

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
// Dropped when the type is collected.
std::unordered_map<PyTypeObject*, TypeInfo> type_info;

void release(TypeInfo& info) {
    Py_XDECREF(info.tensor_type);
    Py_XDECREF(info.native_call);
    info.tensor_type = info.native_call = nullptr;
}

void forget_type_info(PyObject* type) {
    auto found = type_info.find((PyTypeObject*)type);
    if (found == type_info.end()) return;
    TypeInfo info = found->second;
    type_info.erase(found);
    release(info);
}

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
    try {
        on_type_collected((PyObject*)type, forget_type_info);
    } catch (...) {
        release(info);
        throw;
    }
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
    // A module's forward is Python, and the bulk of a model's: the batch on
    // the worker thread, if any, may take the graph lock meanwhile.
    GraphLockSuspend python_call;
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
    // A large product that records gradients goes to `nn.Linear.execute`,
    // whose training route times cuBLASLt's candidates for its three GEMMs
    // (`lt_linear_train_cuda`, whose thresholds these are): for a Qwen3 MLP's
    // weight gradient the heuristic pick taken here ran 219 us, the best
    // candidate 172. That route makes its own checks and falls back to this
    // same product.
    if (runtime_flag_use_cuda() && !no_grad && !amp_reg && !w->is_stop_grad()) {
        int64 rows = a->num / w->shape[1];
        if (rows >= 1024 && rows * w->shape[0] * w->shape[1] >= (int64(1) << 28)) return nullptr;
    }
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

bool pair_of_ints(PyObject* value, int& a, int& b) {
    if (PyLong_Check(value)) {
        a = b = (int)PyLong_AsLong(value);
        return !PyErr_Occurred();
    }
    if (!(PyTuple_Check(value) || PyList_Check(value)) || PySequence_Size(value) != 2) return false;
    PyObjHolder first(PySequence_GetItem(value, 0)), second(PySequence_GetItem(value, 1));
    if (!PyLong_Check(first.obj) || !PyLong_Check(second.obj)) return false;
    a = (int)PyLong_AsLong(first.obj);
    b = (int)PyLong_AsLong(second.obj);
    return !PyErr_Occurred();
}

// `channels_last_source`: the dense NHWC tensor `v` is an NCHW view of, or null.
VarPtr channels_last_source(Var* v) {
    if (v->shape.size() != 4 || v->is_contiguous() || v->storage_offset_bytes) return nullptr;
    int64 c = v->shape[1], h = v->shape[2], w = v->shape[3];
    if (std::min(c, std::min(h, w)) <= 1) return nullptr;
    const auto& st = v->storage_strides;
    if (st.size() != 4 || st[0] != h * w * c || st[1] != 1 || st[2] != w * c || st[3] != c)
        return nullptr;
    return storage_view_transpose(v, {0, 2, 3, 1});
}

PyObject* wrap_like_input(PyObject* x, VarPtr&& value) {
    PyTensorFrontendScope scope(x, nullptr, 0, false);
    return to_py_object<VarHolder*>(new VarHolder(move(value)));
}

// A half-precision `nn.Conv2d` on an inference call, as `_try_cudnn_conv2d`
// builds it with channels-last activations: its filter already moved to OHWI
// (or a 1x1 one relabelled), the input read as NHWC where it lies so, one
// `cudnn_conv` writing NHWC, its bias added, and the result handed out as an
// NCHW view. 12.6 us of host time in Python a call, 53 of them a ResNet-50
// forward. nullptr for anything else -- the first call, which moves the
// filter, among it.
PyObject* conv2d_inference(PyObject* dict, PyObject* x) {
    if (!conv_cudnn_kernel || !cudnn_backend || !conv_filter_key || !no_grad || amp_reg
            || !runtime_flag_use_cuda() || !dict || !is_var(x))
        return nullptr;
    PyObject* weight = PyDict_GetItemString(dict, "weight");
    PyObject* bias = PyDict_GetItemString(dict, "bias");
    PyObject* groups = PyDict_GetItemString(dict, "groups");
    PyObject* mode = PyDict_GetItemString(dict, "padding_mode");
    if (!weight || !is_var(weight) || !bias || !groups || !PyLong_Check(groups)
            || PyLong_AsLong(groups) != 1)
        return nullptr;
    if (mode && !(PyUnicode_Check(mode) && PyUnicode_CompareWithASCIIString(mode, "zeros") == 0))
        return nullptr;
    bool has_bias = bias != Py_None;
    if (has_bias && !is_var(bias)) return nullptr;
    int sh, sw, ph, pw, dh, dw;
    PyObject* stride = PyDict_GetItemString(dict, "stride");
    PyObject* padding = PyDict_GetItemString(dict, "padding");
    PyObject* dilation = PyDict_GetItemString(dict, "dilation");
    if (!stride || !padding || !dilation || !pair_of_ints(stride, sh, sw)
            || !pair_of_ints(padding, ph, pw) || !pair_of_ints(dilation, dh, dw)) {
        PyErr_Clear();
        return nullptr;
    }
    if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0 || ph < 0 || pw < 0) return nullptr;
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    Var* w = GET_RAW_PTR(VarHolder, weight)->var;
    NanoString dtype = a->dtype();
    if ((dtype != ns_float16 && dtype != ns_bfloat16) || w->dtype() != dtype) return nullptr;
    if (has_bias && GET_RAW_PTR(VarHolder, bias)->var->dtype() != dtype) return nullptr;
    if (a->shape.size() != 4 || w->shape.size() != 4 || a->shape[1] != w->shape[1]) return nullptr;
    int64 oh = (a->shape[2] + 2 * ph - dh * (w->shape[2] - 1) - 1) / sh + 1;
    int64 ow = (a->shape[3] + 2 * pw - dw * (w->shape[3] - 1) - 1) / sw + 1;
    if (oh <= 0 || ow <= 0) return nullptr;
    {
        PyObjHolder on(PyObject_GetAttrString(cudnn_backend, "channels_last_activations"));
        if (!on.obj || PyObject_IsTrue(on.obj) != 1) { PyErr_Clear(); return nullptr; }
    }
    // The dispatcher's answer, as `conv2d` asks for it.
    {
        PyObjHolder st(Py_BuildValue("(ii)", sh, sw)), pd(Py_BuildValue("(ii)", ph, pw)),
            dl(Py_BuildValue("(ii)", dh, dw));
        PyObjHolder call_args(PyTuple_Pack(7, x, weight, bias, st.obj, pd.obj, dl.obj, groups));
        PyObjHolder kernel(kernel_select(conv2d_name, call_args.obj, depthwise_kwargs));
        if (kernel.obj != conv_cudnn_kernel) { PyErr_Clear(); return nullptr; }
    }
    // The filter `_inference_filter` gives.
    VarPtr filter;
    int64 out_c = w->shape[0], in_c = w->shape[1];
    if (w->shape[2] == 1 && w->shape[3] == 1 && w->is_contiguous()) {
        static auto make_reshape_mc = op_constructor<VarPtr, Var*, NanoVector>("reshape");
        filter = make_reshape_mc(w, {out_c, 1, 1, in_c});
    } else {
        PyObject** wdict = _PyObject_GetDictPtr(weight);
        PyObject* moved = (wdict && *wdict) ? PyDict_GetItem(*wdict, conv_filter_key) : nullptr;
        if (!moved || !PyTuple_Check(moved) || PyTuple_GET_SIZE(moved) != 2) return nullptr;
        PyObject* ptr = PyTuple_GET_ITEM(moved, 0);
        PyObject* dense = PyTuple_GET_ITEM(moved, 1);
        if (!PyLong_Check(ptr) || (Var*)PyLong_AsVoidPtr(ptr) != w || !is_var(dense)) {
            PyErr_Clear();
            return nullptr;
        }
        filter = VarPtr(GET_RAW_PTR(VarHolder, dense)->var);
    }
    VarPtr source = channels_last_source(a);
    if (!source && !a->is_finished()) {
        Op* op = a->input();
        if (op && op->is_op(op_ids::contiguous()) && op->inputs().size() == 1)
            source = channels_last_source(op->inputs().front());
    }
    static auto make_cudnn_conv = op_constructor<VarPtr, Var*, Var*, int, int, int, int, int, int, int,
                                                 string, string, string>("cudnn_conv");
    VarPtr y = make_cudnn_conv(source ? source.ptr : a, filter.ptr, sh, sw, ph, pw, dh, dw, 1,
                               source ? "acdb" : "abcd", "ohwi", "acdb");
    PyObject* out = wrap_like_input(x, move(y));
    if (has_bias) {
        PyObject* sum = PyNumber_Add(out, bias);
        Py_DECREF(out);
        if (!sum) return nullptr;
        out = sum;
    }
    Var* nhwc = GET_RAW_PTR(VarHolder, out)->var;
    PyObject* view = wrap_like_input(x, storage_view_transpose(nhwc, {0, 3, 1, 2}));
    Py_DECREF(out);
    return view;
}

// An inference `nn.BatchNorm2d` over a channels-last view, as `_batch_norm_eval`
// computes it there -- the CUDA kernel takes dense NCHW only -- from the scale
// and shift it keeps on the tracked variance: x * scale + shift, broadcast over
// (0, 2, 3). nullptr when the kept pair is not for these parameters, or for a
// dense input.
PyObject* batch_norm_eval_channels_last(PyObject* dict, PyObject* x) {
    if (!bn_coefficients_key || !dict || !is_var(x) || acl_possible) return nullptr;
    PyObject* train = PyDict_GetItemString(dict, "is_train");
    if (!train || PyObject_IsTrue(train) != 0) { PyErr_Clear(); return nullptr; }
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    if (a->shape.size() != 4 || a->is_contiguous()) return nullptr;
    PyObject* names[4] = {PyDict_GetItemString(dict, "weight"), PyDict_GetItemString(dict, "bias"),
                          PyDict_GetItemString(dict, "running_mean"),
                          PyDict_GetItemString(dict, "running_var")};
    for (auto* v : names) if (!v || !is_var(v)) return nullptr;
    // As `_batch_norm_eval_coefficients` keeps them: only where no gradient
    // flows through them.
    if (!no_grad)
        for (auto* v : names)
            if (!GET_RAW_PTR(VarHolder, v)->var->is_stop_grad()) return nullptr;
    PyObject* eps = PyDict_GetItemString(dict, "eps");
    if (!eps || !(PyFloat_Check(eps) || PyLong_Check(eps))) return nullptr;
    PyObject** vdict = _PyObject_GetDictPtr(names[3]);
    PyObject* kept = (vdict && *vdict) ? PyDict_GetItem(*vdict, bn_coefficients_key) : nullptr;
    if (!kept || !PyTuple_Check(kept) || PyTuple_GET_SIZE(kept) != 3) return nullptr;
    PyObject* key = PyTuple_GET_ITEM(kept, 0);
    if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 5) return nullptr;
    for (int i = 0; i < 4; i++) {
        PyObject* ptr = PyTuple_GET_ITEM(key, i);
        if (!PyLong_Check(ptr) || (Var*)PyLong_AsVoidPtr(ptr) != GET_RAW_PTR(VarHolder, names[i])->var) {
            PyErr_Clear();
            return nullptr;
        }
    }
    double e = PyFloat_AsDouble(eps);
    PyObject* kept_eps = PyTuple_GET_ITEM(key, 4);
    if (PyErr_Occurred() || !PyFloat_Check(kept_eps) || PyFloat_AS_DOUBLE(kept_eps) != e) {
        PyErr_Clear();
        return nullptr;
    }
    PyObject* scale = PyTuple_GET_ITEM(kept, 1);
    PyObject* shift = PyTuple_GET_ITEM(kept, 2);
    if (!is_var(scale) || !is_var(shift)) return nullptr;
    // `scale.broadcast(x, [0, 2, 3])` and the frontend's `*` and `+`, built
    // here rather than through two method calls and two Python operators:
    // the same broadcasts, and the same native binary path `_tensor_mul` and
    // `_tensor_add` take first -- channels-last propagation and all -- with
    // the Python operators still behind it for what that path declines.
    static auto make_broadcast_to = op_constructor<VarPtr, Var*, Var*, NanoVector>("broadcast_to");
    PyObjHolder s(wrap_like_input(x, make_broadcast_to(GET_RAW_PTR(VarHolder, scale)->var, a, {0, 2, 3})));
    PyObjHolder b(wrap_like_input(x, make_broadcast_to(GET_RAW_PTR(VarHolder, shift)->var, a, {0, 2, 3})));
    auto apply = [](PyObject* lhs, PyObject* rhs, int code, binaryfunc op) -> PyObject* {
        PyObject* out = fast_binary(lhs, rhs, code);
        if (out != Py_None) return out;
        Py_DECREF(out);
        return op(lhs, rhs);
    };
    PyObjHolder product(apply(x, s.obj, 4, PyNumber_Multiply));
    if (!product.obj) return nullptr;
    return apply(product.obj, b.obj, 0, PyNumber_Add);
}

// A Dropout that is not training, or drops nothing, hands its input back --
// `jittor.nn.dropout` returns `x` itself, as PyTorch does -- and a BERT-base
// forward makes 37 such calls at 5.9 us each through Python. nullptr when it
// would do anything else.
PyObject* dropout_passthrough(PyObject* dict, PyObject* x) {
    if (!dict || !is_var(x)) return nullptr;
    PyObject* train = PyDict_GetItemString(dict, "is_train");
    PyObject* p = PyDict_GetItemString(dict, "p");
    if (!train || !p) return nullptr;
    int training = PyObject_IsTrue(train);
    if (training < 0) { PyErr_Clear(); return nullptr; }
    if (training) {
        if (!(PyFloat_Check(p) || PyLong_Check(p))) return nullptr;
        double rate = PyFloat_AsDouble(p);
        if (PyErr_Occurred()) { PyErr_Clear(); return nullptr; }
        if (rate != 0.0) return nullptr;
    }
    Py_INCREF(x);
    return x;
}

// `nn.ReLU` and `nn.SiLU`, as `jittor.nn.relu` and `jittor.nn.silu` build
// them: from the offer of the pass that makes the input, when it made one
// (`_fused_activation`: a group norm applying the activation in its own last
// pass), and otherwise one `unary` relu, or `x * x.sigmoid()`. Through Python
// a ReLU was 3.9 us of host time a call -- the function looked up by name, the
// `inplace` policy, the offers, the dispatcher -- and a ResNet-50 forward makes
// 49; a SiLU 7.8 us, 21 of them a SD1.5 UNet step. nullptr for anything else:
// a residual offer anywhere (those are found through the add), a kernel
// registered for the activation, the `jittor.nn` function replaced, arguments
// beyond `inplace`, or an `inplace=True` not warned about yet.
PyObject* activation_module(PyTypeObject* type, PyObject* dict, PyObject* x) {
    if (!nn_namespace || !residual_offers || !arg_policy_warned || !dict || !is_var(x))
        return nullptr;
    if (!function_name_key) function_name_key = PyUnicode_InternFromString("_function_name");
    PyObject* name = _PyType_Lookup(type, function_name_key);
    if (!name || !PyUnicode_Check(name)) { PyErr_Clear(); return nullptr; }
    bool relu = PyUnicode_CompareWithASCIIString(name, "relu") == 0;
    bool silu = !relu && PyUnicode_CompareWithASCIIString(name, "silu") == 0;
    PyObject* function = relu ? relu_function : silu ? silu_function : nullptr;
    if (!function) return nullptr;
    PyObject* extra = PyDict_GetItemString(dict, "args");
    PyObject* kw = PyDict_GetItemString(dict, "kw");
    if (!extra || !PyTuple_Check(extra) || PyTuple_GET_SIZE(extra) || !kw || !PyDict_Check(kw))
        return nullptr;
    PyObject* inplace = PyDict_GetItemString(kw, "inplace");
    if (PyDict_GET_SIZE(kw) != (inplace ? 1 : 0)) return nullptr;
    if (inplace) {
        int on = PyObject_IsTrue(inplace);
        if (on < 0) { PyErr_Clear(); return nullptr; }
        if (on) {
            static PyObject* relu_key = Py_BuildValue("(ss)", "jittor.nn.relu", "inplace");
            static PyObject* silu_key = Py_BuildValue("(ss)", "jittor.nn.silu", "inplace");
            if (PySet_Contains(arg_policy_warned, relu ? relu_key : silu_key) != 1) {
                PyErr_Clear();
                return nullptr;
            }
        }
    }
    PyObject* current = PyObject_GetAttr(nn_namespace, name);
    Py_XDECREF(current);
    if (current != function) { PyErr_Clear(); return nullptr; }
    if (PyDict_GET_SIZE(residual_offers)) return nullptr;
    Var* v = GET_RAW_PTR(VarHolder, x)->var;
    if (!fuse_activation_key) fuse_activation_key = PyUnicode_InternFromString("_fuse_activation");
    PyObject** xdict = _PyObject_GetDictPtr(x);
    PyObject* entry = xdict && *xdict ? PyDict_GetItem(*xdict, fuse_activation_key) : nullptr;
    if (entry) {
        if (!PyTuple_Check(entry) || PyTuple_GET_SIZE(entry) != 2
                || !PyLong_Check(PyTuple_GET_ITEM(entry, 0)))
            return nullptr;
        long long id = PyLong_AsLongLong(PyTuple_GET_ITEM(entry, 0));
        if (PyErr_Occurred()) { PyErr_Clear(); return nullptr; }
        if (id == (long long)v->id && !v->is_finished()) {
            PyObjHolder build(PyTuple_GET_ITEM(entry, 1));
            Py_INCREF(build.obj);
            PyObject* fused = PyObject_CallOneArg(build.obj, name);
            if (!fused || fused != Py_None) return fused;
            Py_DECREF(fused);
        }
    }
    if (kernel_op_registered(relu ? "nn.relu" : "nn.silu")) return nullptr;
    static auto make_unary_mc = op_constructor<VarPtr, Var*, NanoString>("unary");
    if (relu) return wrap_like_input(x, make_unary_mc(v, ns_relu));
    // `x.sigmoid()` is the native unary unless a kernel is registered for it.
    if (!v->dtype().is_float() || kernel_op_registered("tensor.sigmoid")) return nullptr;
    PyObjHolder gate(wrap_like_input(x, make_unary_mc(v, ns_sigmoid)));
    PyObject* out = fast_binary(x, gate.obj, 4);
    if (out != Py_None) return out;
    Py_DECREF(out);
    return PyNumber_Multiply(x, gate.obj);
}

// The group norm of a dense [N, H, W, C] `source` with the activation `act`
// ("" for none) in its last pass, handed out as the NCHW view of its storage,
// as `group_norm_cuda._group_norm_nhwc` builds it -- source, launch and all.
// None for an activation it has no pass for.
struct GroupNormSource { string header, source; int64 rows, parts; };
std::map<std::tuple<int64, int64, int64, int64, int64, double, string>, GroupNormSource> group_norm_sources;

PyObject* build_group_norm_nhwc(PyObject* like, Var* source, Var* weight, Var* bias,
                                int64 groups, double eps, const string& act) {
    if (!group_norm_source_fn) return nullptr;
    if (source->shape.size() != 4) return nullptr;
    int64 n = source->shape[0], h = source->shape[1], w = source->shape[2], c = source->shape[3];
    auto key = std::make_tuple(n, h, w, c, groups, eps, act);
    auto found = group_norm_sources.find(key);
    if (found == group_norm_sources.end()) {
        PyObjHolder shape(Py_BuildValue("(LLLL)", (long long)n, (long long)h, (long long)w, (long long)c));
        PyObjHolder made(PyObject_CallFunction(group_norm_source_fn, "OLds", shape.obj,
                                               (long long)groups, eps, act.c_str()));
        if (!PyTuple_Check(made.obj) || PyTuple_GET_SIZE(made.obj) != 4) return nullptr;
        GroupNormSource text;
        Py_ssize_t size;
        const char* header = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(made.obj, 0), &size);
        if (!header) return nullptr;
        text.header.assign(header, size);
        const char* source_text = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(made.obj, 1), &size);
        if (!source_text) return nullptr;
        text.source.assign(source_text, size);
        text.rows = PyLong_AsLongLong(PyTuple_GET_ITEM(made.obj, 2));
        text.parts = PyLong_AsLongLong(PyTuple_GET_ITEM(made.obj, 3));
        if (PyErr_Occurred()) return nullptr;
        found = group_norm_sources.emplace(key, std::move(text)).first;
    }
    static auto make_code_multi = op_constructor<vector<VarPtr>, vector<NanoVector>&&, vector<NanoString>&&,
        vector<Var*>&&, string&&, vector<string>&&, string&&, string&&, vector<string>&&, string&&,
        DataMap&&, string&&>("code");
    auto outs = make_code_multi(
        {source->shape, {found->second.rows}, {found->second.rows}, {found->second.parts}},
        {source->dtype(), ns_float32, ns_float32, ns_float32}, {source, weight, bias},
        "", {}, "", string(found->second.source), {}, string(found->second.header), {}, "");
    return wrap_like_input(like, storage_view_transpose(outs[0].ptr, {0, 3, 1, 2}));
}

// An inference `nn.GroupNorm` over a channels-last view, as `jittor.nn.group_norm`
// reaches `_group_norm_nhwc` for it, with the same offer to take a following
// activation into its pass -- through the native build, so that the SiLU taking
// it is native too. 17 us of host time a call through Python, 61 a SD1.5 UNet
// step. nullptr for anything else.
PyObject* group_norm_inference(PyObject* dict, PyObject* x) {
    if (!group_norm_function || !group_norm_kernel || !group_norm_source_fn
            || !group_norm_build_fn || !partial_type || !group_norm_activations
            || !no_grad || amp_reg || auto_mixed_precision_level || !dict || !is_var(x))
        return nullptr;
    PyObject* groups_obj = PyDict_GetItemString(dict, "num_groups");
    PyObject* channels_obj = PyDict_GetItemString(dict, "num_channels");
    PyObject* eps_obj = PyDict_GetItemString(dict, "eps");
    PyObject* weight = PyDict_GetItemString(dict, "weight");
    PyObject* bias = PyDict_GetItemString(dict, "bias");
    if (!groups_obj || !channels_obj || !eps_obj || !weight || !bias || !is_var(weight) || !is_var(bias)
            || !PyLong_CheckExact(groups_obj) || !PyLong_CheckExact(channels_obj)
            || !(PyFloat_CheckExact(eps_obj) || PyLong_CheckExact(eps_obj)))
        return nullptr;
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    if (a->shape.size() != 4 || a->shape[1] != PyLong_AsLongLong(channels_obj)) {
        PyErr_Clear();
        return nullptr;
    }
    PyObject* current = PyObject_GetAttrString(nn_namespace, "group_norm");
    Py_XDECREF(current);
    if (current != group_norm_function) { PyErr_Clear(); return nullptr; }
    VarPtr source = channels_last_source(a);
    if (!source) return nullptr;
    {
        PyObjHolder call_args(PyTuple_Pack(5, x, groups_obj, weight, bias, eps_obj));
        PyObjHolder kernel(kernel_select(group_norm_name, call_args.obj, nullptr));
        if (kernel.obj != group_norm_kernel) { PyErr_Clear(); return nullptr; }
    }
    int64 groups = PyLong_AsLongLong(groups_obj);
    double eps = PyFloat_AsDouble(eps_obj);
    if (PyErr_Occurred()) { PyErr_Clear(); return nullptr; }
    Var* w = GET_RAW_PTR(VarHolder, weight)->var;
    Var* b = GET_RAW_PTR(VarHolder, bias)->var;
    PyObject* y = build_group_norm_nhwc(x, source.ptr, w, b, groups, eps, "");
    if (!y) return nullptr;
    // `offer_activation(y, build)`, the build being the native one above.
    PyObjHolder holder(y);
    PyObjHolder source_obj(wrap_like_input(x, move(source)));
    PyObjHolder build(PyObject_CallFunction(partial_type, "OOOOLd", group_norm_build_fn,
                                            source_obj.obj, weight, bias, (long long)groups, eps));
    if (!build.obj) return nullptr;
    if (!fuse_activation_key) fuse_activation_key = PyUnicode_InternFromString("_fuse_activation");
    PyObjHolder id(PyLong_FromLongLong(GET_RAW_PTR(VarHolder, y)->var->id));
    PyObjHolder entry(PyTuple_Pack(2, id.obj, build.obj));
    PyObject** ydict = _PyObject_GetDictPtr(y);
    if (!ydict) return nullptr;
    if (!*ydict) *ydict = PyDict_New();
    if (!*ydict || PyDict_SetItem(*ydict, fuse_activation_key, entry.obj) < 0) return nullptr;
    Py_INCREF(y);
    return y;
}

// `nn.Linear` with a bias on an inference call, as `lt_linear_cuda` builds it:
// one cuBLASLt GEMM with the bias in its epilogue, from `_source`. Taken only
// where that function would take it with no cast to make -- under `no_grad`,
// outside autocast, all three operands one dtype it serves, dense, and a
// product large enough for it -- so the operator is the one the Python path
// builds. 8.6 us of host time to build in Python, 74 times a BERT-base forward.
PyObject* linear_with_bias_inference(PyObject* dict, PyObject* x) {
    if (!lt_linear_source_fn || lt_linear_header_text.empty() || !no_grad || amp_reg
            || !runtime_flag_use_cuda() || !dict || !is_var(x))
        return nullptr;
    PyObject* weight = PyDict_GetItemString(dict, "weight");
    PyObject* bias = PyDict_GetItemString(dict, "bias");
    if (!weight || !is_var(weight) || !bias || !is_var(bias)) return nullptr;
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    Var* w = GET_RAW_PTR(VarHolder, weight)->var;
    Var* b = GET_RAW_PTR(VarHolder, bias)->var;
    NanoString dtype = a->dtype();
    if ((dtype != ns_float32 && dtype != ns_float16) || w->dtype() != dtype || b->dtype() != dtype)
        return nullptr;
    if (!a->is_contiguous() || !w->is_contiguous() || !b->is_contiguous()) return nullptr;
    int rank = a->shape.size();
    if (w->shape.size() != 2 || b->shape.size() != 1 || rank < 2) return nullptr;
    int64 cin = w->shape[1], cout = w->shape[0];
    if (a->shape[rank - 1] != cin || b->shape[0] != cout || a->num < 0) return nullptr;
    int64 rows = cin ? a->num / cin : 0;
    if (rows * cout * cin < (int64(1) << 18)) return nullptr;
    bool half = dtype == ns_float16;
    auto key = std::make_tuple(rows, cin, cout, half);
    auto found = lt_linear_sources.find(key);
    if (found == lt_linear_sources.end()) {
        PyObjHolder src(PyObject_CallFunction(lt_linear_source_fn, "LLLs",
            (long long)rows, (long long)cin, (long long)cout, half ? "float16" : "float32"));
        Py_ssize_t size;
        const char* text = PyUnicode_AsUTF8AndSize(src.obj, &size);
        if (!text) return nullptr;
        found = lt_linear_sources.emplace(key, string(text, size)).first;
    }
    NanoVector shape;
    for (int i = 0; i < rank - 1; i++) shape.push_back(a->shape[i]);
    shape.push_back(cout);
    PyTensorFrontendScope scope(x, nullptr, 0, false);
    unique_ptr<VarHolder> out(new VarHolder(make_code(shape, dtype, {a, w, b},
        "", {}, "", string(found->second), {}, string(lt_linear_header_text), {}, "")));
    out->stop_grad();
    return to_py_object<VarHolder*>(out.release());
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

// `nn.LayerNorm` with a weight and a bias on an inference call: the operator
// `_layer_norm_no_grad_cuda` builds, when the dispatcher picks it -- asked with
// the arguments `layer_norm` asks with, which its native rule answers without
// a Python predicate. Anything else -- a shape `layer_norm` rejects, a call that
// needs a gradient, autocast -- goes the Python way. 7.6 us of host time a call
// in Python, 25 a BERT-base forward and 48 an SD1.5 UNet step.
PyObject* layer_norm_inference(PyObject* dict, PyObject* x) {
    if (!layer_norm_impl || !layer_norm_source_fn || amp_reg || !runtime_flag_use_cuda()
            || !dict || !is_var(x))
        return nullptr;
    PyObject* shape = PyDict_GetItemString(dict, "normalized_shape");
    PyObject* eps = PyDict_GetItemString(dict, "eps");
    PyObject* affine = PyDict_GetItemString(dict, "elementwise_affine");
    PyObject* weight = PyDict_GetItemString(dict, "weight");
    PyObject* bias = PyDict_GetItemString(dict, "bias");
    if (!shape || !PyTuple_Check(shape) || PyTuple_GET_SIZE(shape) != 1 || affine != Py_True
            || !eps || !PyFloat_Check(eps) || !weight || !is_var(weight)
            || !bias || !is_var(bias))
        return nullptr;
    PyObject* size = PyTuple_GET_ITEM(shape, 0);
    if (!PyLong_CheckExact(size)) return nullptr;
    int64 hidden = PyLong_AsLongLong(size);
    double epsilon = PyFloat_AS_DOUBLE(eps);
    if (PyErr_Occurred()) { PyErr_Clear(); return nullptr; }
    Var* a = GET_RAW_PTR(VarHolder, x)->var;
    Var* gamma = GET_RAW_PTR(VarHolder, weight)->var;
    Var* beta = GET_RAW_PTR(VarHolder, bias)->var;
    int rank = a->shape.size();
    if (!rank || a->shape[rank - 1] != hidden || a->num < 0) return nullptr;
    if (gamma->shape.size() != 1 || gamma->shape[0] != hidden
            || beta->shape.size() != 1 || beta->shape[0] != hidden)
        return nullptr;
    if (!no_grad && (!a->is_stop_grad() || !gamma->is_stop_grad() || !beta->is_stop_grad()))
        return nullptr;
    PyObjHolder call_args(PyTuple_Pack(5, x, shape, weight, bias, eps));
    PyObjHolder chosen(kernel_select(layer_norm_name, call_args.obj, nullptr));
    if (!chosen.obj || chosen.obj != layer_norm_impl) {
        if (!chosen.obj) PyErr_Clear();
        return nullptr;
    }
    bool warp = hidden <= 1024 && a->num / hidden >= 1024;
    auto key = std::make_tuple(hidden, epsilon, warp);
    auto found = layer_norm_sources.find(key);
    if (found == layer_norm_sources.end()) {
        PyObjHolder src(PyObject_CallFunction(layer_norm_source_fn, "LdO",
            (long long)hidden, epsilon, warp ? Py_True : Py_False));
        Py_ssize_t length;
        const char* text = PyUnicode_AsUTF8AndSize(src.obj, &length);
        if (!text) return nullptr;
        found = layer_norm_sources.emplace(key, string(text, length)).first;
    }
    PyTensorFrontendScope scope(x, nullptr, 0, false);
    unique_ptr<VarHolder> out(new VarHolder(make_code(a->shape, a->dtype(), {a, gamma, beta},
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
        if (own && PyCallable_Check(own)) {
            GraphLockSuspend python_call;
            return PyObject_Call(own, args, kwargs);
        }
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
        fast = linear_with_bias_inference(dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && conv_execute
            && _PyType_Lookup(type, name_execute()) == conv_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* fast = conv2d_inference(dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && batch_norm_execute
            && _PyType_Lookup(type, name_execute()) == batch_norm_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* fast = batch_norm_eval_channels_last(dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && layer_norm_execute
            && _PyType_Lookup(type, name_execute()) == layer_norm_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* fast = layer_norm_inference(dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && dropout_execute
            && _PyType_Lookup(type, name_execute()) == dropout_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* same = dropout_passthrough(dict, PyTuple_GET_ITEM(args, 0));
        if (same) return same;
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && function_module_execute
            && _PyType_Lookup(type, name_execute()) == function_module_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* fast = activation_module(type, dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    if (!kwargs && PyTuple_GET_SIZE(args) == 1 && group_norm_execute
            && _PyType_Lookup(type, name_execute()) == group_norm_execute
            && !(dict && PyDict_GetItemString(dict, "execute"))) {
        PyObject* fast = group_norm_inference(dict, PyTuple_GET_ITEM(args, 0));
        if (fast || PyErr_Occurred()) return fast;
    }
    PyObjHolder execute(PyObject_GetAttr(module, name_execute()));
    GraphLockSuspend python_call;
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
    part(dropout_execute, "dropout_execute");
    part(function_module_execute, "function_module_execute");
    part(nn_namespace, "nn_namespace");
    part(relu_function, "relu_function");
    part(silu_function, "silu_function");
    part(group_norm_execute, "group_norm_execute");
    part(group_norm_function, "group_norm_function");
    part(group_norm_kernel, "group_norm_kernel");
    part(group_norm_source_fn, "group_norm_nhwc_source");
    part(group_norm_activations, "group_norm_activations");
    part(group_norm_build_fn, "group_norm_nhwc_build");
    part(partial_type, "partial");
    if (!group_norm_name) group_norm_name = PyUnicode_InternFromString("nn.group_norm");
    group_norm_sources.clear();
    part(residual_offers, "residual_offers");
    part(arg_policy_warned, "arg_policy_warned");
    part(conv_execute, "conv_execute");
    part(conv_cudnn_kernel, "conv_cudnn_kernel");
    part(cudnn_backend, "cudnn_backend");
    part(conv_filter_key, "conv_filter_key");
    part(batch_norm_execute, "batch_norm_execute");
    part(layer_norm_execute, "layer_norm_execute");
    part(layer_norm_impl, "layer_norm_inference");
    part(layer_norm_source_fn, "layer_norm_source");
    if (!layer_norm_name) layer_norm_name = PyUnicode_InternFromString("nn.layer_norm.inference");
    layer_norm_sources.clear();
    part(bn_coefficients_key, "bn_coefficients_key");
    if (!conv2d_name) conv2d_name = PyUnicode_InternFromString("conv2d");
    if (!depthwise_kwargs) {
        depthwise_kwargs = PyDict_New();
        PyDict_SetItemString(depthwise_kwargs, "_depthwise_fast_path", Py_True);
    }
    part(lt_linear_header, "lt_linear_header");
    part(lt_linear_source_fn, "lt_linear_source");
    lt_linear_header_text.clear();
    if (lt_linear_header && PyUnicode_Check(lt_linear_header)) {
        Py_ssize_t size;
        const char* text = PyUnicode_AsUTF8AndSize(lt_linear_header, &size);
        if (text) lt_linear_header_text.assign(text, size);
        else PyErr_Clear();
    }
    lt_linear_sources.clear();
    PyObject* acl = dispatch_parts && PyDict_Check(dispatch_parts)
        ? PyDict_GetItemString(dispatch_parts, "acl_possible") : nullptr;
    acl_possible = !acl || PyObject_IsTrue(acl) != 0;
    if (!rms_training_name) rms_training_name = PyUnicode_InternFromString("nn.rms_norm.training");
    if (!rms_inference_name) rms_inference_name = PyUnicode_InternFromString("nn.rms_norm.inference");
    rms_sources.clear();
    if (!empty_kwargs) empty_kwargs = PyDict_New();
    if (!matmul_name) matmul_name = PyUnicode_InternFromString("matmul");
    for (auto& entry : type_info) release(entry.second);
    type_info.clear();
}

PyObject* group_norm_nhwc_build(PyObject* source, PyObject* weight, PyObject* bias,
                                int64 groups, double eps, const string& act) {
    if (!is_var(source) || !is_var(weight) || !is_var(bias) || !group_norm_activations)
        throw std::runtime_error("_group_norm_nhwc_build needs three tensors and a bound module call");
    PyObjHolder name(PyUnicode_FromStringAndSize(act.data(), act.size()));
    int known = PySequence_Contains(group_norm_activations, name.obj);
    if (known < 0) return nullptr;
    if (!known) { Py_INCREF(Py_None); return Py_None; }
    PyObject* out = build_group_norm_nhwc(source, GET_RAW_PTR(VarHolder, source)->var,
                                          GET_RAW_PTR(VarHolder, weight)->var,
                                          GET_RAW_PTR(VarHolder, bias)->var, groups, eps, act);
    if (!out && !PyErr_Occurred())
        throw std::runtime_error("_group_norm_nhwc_build could not build the group norm");
    return out;
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
    // A module boundary: where a batch is cut for the worker thread.
    if (result && !error_type) runtime_executor().flush_at_module_boundary();
    set_float32_precision_policy(previous_precision);
    if (placement_token) { reset_tensor_placement_context(placement_token); Py_DECREF(placement_token); }
    reset_tensor_frontend_type(type_token);
    Py_DECREF(type_token);
    PyErr_Restore(error_type, error_value, error_tb);
    return result;
}

} // namespace jittor
