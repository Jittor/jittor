#include "bindings/pyjt/py_kernel_select.h"
#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "runtime/dispatch_context.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "runtime/device.h"
#include "ops/op_register.h"
#include <cmath>
#include <stdexcept>

namespace jittor {

DECLARE_FLAG(bool, no_grad);
DECLARE_FLAG(int, amp_reg);

namespace {

// A native stand-in for a `supports` predicate; see kernel_select_native_rule.
struct NativeRule {
    enum Kind { none, same_float, rms_inference, rms_training, ln_inference, depthwise_conv2d,
                group_norm } kind = none;
    string op;
};
enum Verdict { declines = 0, accepts = 1, unknown = 2 };

struct Candidate {
    PyObject* implementation;
    PyObject* dtypes;         // frozenset of names, or nullptr
    PyObject* supports;       // callable, or nullptr
    PyObject* runtime_modes;  // frozenset of ints, or nullptr
    NativeRule rule;
};

PyObject* candidates_fn = nullptr;
PyObject* dtype_name_fn = nullptr;
PyObject* registered_ops = nullptr;                       // frozenset of str
std::unordered_map<string, vector<Candidate>> resolved;   // "op\0backend"
std::unordered_map<string, PyObject*> dtype_names;        // raw -> canonical str
std::unordered_map<PyObject*, NativeRule> native_rules;   // supports -> rule

PyObject* attr_or_null(PyObject* obj, const char* name) {
    PyObject* value = PyObject_GetAttrString(obj, name);
    if (!value) throw std::runtime_error("kernel registration without " + string(name));
    if (value == Py_None) { Py_DECREF(value); return nullptr; }
    return value;
}

void release(vector<Candidate>& entries) {
    for (auto& e : entries) {
        Py_XDECREF(e.implementation);
        Py_XDECREF(e.dtypes);
        Py_XDECREF(e.supports);
        Py_XDECREF(e.runtime_modes);
    }
    entries.clear();
}

const vector<Candidate>& lookup(const string& op, const string& backend) {
    string key = op;
    key.push_back('\0');
    key += backend;
    auto found = resolved.find(key);
    if (found != resolved.end()) return found->second;
    PyObjHolder entries(PyObject_CallFunction(candidates_fn, "ss", op.c_str(), backend.c_str()));
    PyObjHolder seq(PySequence_Fast(entries.obj, "candidates must be a sequence"));
    vector<Candidate> list;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.obj);
    auto items = PySequence_Fast_ITEMS(seq.obj);
    for (Py_ssize_t i = 0; i < n; i++) {
        Candidate c;
        c.implementation = PyObject_GetAttrString(items[i], "implementation");
        if (!c.implementation) throw std::runtime_error("kernel registration without implementation");
        c.dtypes = attr_or_null(items[i], "dtypes");
        c.supports = attr_or_null(items[i], "supports");
        c.runtime_modes = attr_or_null(items[i], "runtime_modes");
        if (c.supports) {
            auto rule = native_rules.find(c.supports);
            if (rule != native_rules.end()) c.rule = rule->second;
        }
        list.push_back(c);
    }
    return resolved.emplace(key, move(list)).first->second;
}

bool is_var(PyObject* value) {
    return PyObject_TypeCheck(value, &PyjtVarHolder.ht_type);
}

// The Vars among `values`, in argument order, through tuples, lists and
// dicts, as `_collect_tensors` finds them.
void collect(PyObject* value, vector<VarHolder*>& tensors, int depth) {
    if (is_var(value)) {
        tensors.push_back(GET_RAW_PTR(VarHolder, value));
        return;
    }
    if (PyTuple_Check(value) || PyList_Check(value)) {
        if (depth > 32) throw std::invalid_argument("cyclic kernel argument container");
        PyObjHolder seq(PySequence_Fast(value, "sequence"));
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.obj);
        auto items = PySequence_Fast_ITEMS(seq.obj);
        for (Py_ssize_t i = 0; i < n; i++) collect(items[i], tensors, depth + 1);
    } else if (PyDict_Check(value)) {
        if (depth > 32) throw std::invalid_argument("cyclic kernel argument container");
        PyObject *key, *item;
        Py_ssize_t pos = 0;
        while (PyDict_Next(value, &pos, &key, &item)) collect(item, tensors, depth + 1);
    }
}

PyObject* canonical_dtype(VarHolder* holder) {
    const char* raw = holder->var->dtype().to_cstring();
    auto found = dtype_names.find(raw);
    if (found != dtype_names.end()) return found->second;
    PyObjHolder text(PyUnicode_FromString(raw));
    PyObject* name = PyObject_CallFunctionObjArgs(dtype_name_fn, text.obj, nullptr);
    if (!name) throw std::runtime_error("dtype_name failed");
    dtype_names.emplace(raw, name);
    return name;
}

bool is_floating(NanoString dtype) {
    return dtype == ns_float16 || dtype == ns_bfloat16 || dtype == ns_float32;
}

bool requires_grad(Var* v) {
    return !v->is_stop_grad() && !v->flag(VarFlags::_requires_grad_disabled);
}

// `rule` for `supports(*args, **kwargs)`, or `unknown` when Python must say.
// `conv2d`'s one keyword argument: whether the depthwise kernel may take the
// call (`select_kernel("conv2d", ..., _depthwise_fast_path=...)`). 1 or 0, or
// -1 for any other keyword argument.
int depthwise_fast_path(PyObject* kwargs) {
    if (!kwargs || kwargs == Py_None || !PyDict_GET_SIZE(kwargs)) return 1;
    PyObject* flag = PyDict_GetItemString(kwargs, "_depthwise_fast_path");
    if (!flag || PyDict_GET_SIZE(kwargs) != 1) return -1;
    int on = PyObject_IsTrue(flag);
    if (on < 0) { PyErr_Clear(); return -1; }
    return on;
}

Verdict evaluate(const NativeRule& rule, PyObject* args, PyObject* kwargs) {
    if (!PyTuple_Check(args)) return unknown;
    Py_ssize_t n = PyTuple_GET_SIZE(args);
    // The conv2d predicates take `(x, weight, bias, stride, padding, dilation,
    // groups, *, _depthwise_fast_path)`; nothing else here takes a keyword.
    bool conv2d_call = n == 7 && depthwise_fast_path(kwargs) >= 0;
    if (kwargs && kwargs != Py_None && PyDict_GET_SIZE(kwargs) && !conv2d_call)
        return unknown;
    if (rule.kind == NativeRule::group_norm) {
        // `_supports_group_norm(x, num_groups, weight, bias, eps)`.
        if (n != 5 || !is_var(PyTuple_GET_ITEM(args, 0))) return unknown;
        PyObject* groups = PyTuple_GET_ITEM(args, 1);
        PyObject* weight = PyTuple_GET_ITEM(args, 2);
        PyObject* bias = PyTuple_GET_ITEM(args, 3);
        PyObject* eps = PyTuple_GET_ITEM(args, 4);
        if (!is_var(weight) || !is_var(bias)) return declines;
        if (!PyLong_CheckExact(groups) || !(PyFloat_CheckExact(eps) || PyLong_CheckExact(eps)))
            return unknown;
        Var* x = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 0))->var;
        if (x->shape.size() != 4) return declines;
        for (int i = 0; i < 4; i++) if (x->shape[i] <= 0) return declines;
        int64 channels = x->shape[1];
        int64 g = PyLong_AsLongLong(groups);
        double e = PyFloat_AsDouble(eps);
        if (PyErr_Occurred()) { PyErr_Clear(); return unknown; }
        if (g <= 0 || channels % g || GET_RAW_PTR(VarHolder, weight)->var->num != channels
                || GET_RAW_PTR(VarHolder, bias)->var->num != channels
                || !std::isfinite(e) || e <= 0)
            return declines;
        return accepts;
    }
    if (rule.kind == NativeRule::depthwise_conv2d) {
        // `_supports_depthwise_conv2d`: groups == weight.shape[0] ==
        // x.shape[1], one dtype, and the caller allowing it.
        if (!conv2d_call || !is_var(PyTuple_GET_ITEM(args, 0)) || !is_var(PyTuple_GET_ITEM(args, 1)))
            return unknown;
        PyObject* groups = PyTuple_GET_ITEM(args, 6);
        if (!PyLong_CheckExact(groups)) return unknown;
        if (!depthwise_fast_path(kwargs)) return declines;
        Var* x = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 0))->var;
        Var* w = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 1))->var;
        if (x->shape.size() < 2 || w->shape.size() < 1) return unknown;
        int64 g = PyLong_AsLongLong(groups);
        if (PyErr_Occurred()) { PyErr_Clear(); return unknown; }
        return g == w->shape[0] && w->shape[0] == x->shape[1] && x->dtype() == w->dtype()
            ? accepts : declines;
    }
    if (rule.kind == NativeRule::same_float) {
        if (n < 2 || !is_var(PyTuple_GET_ITEM(args, 0)) || !is_var(PyTuple_GET_ITEM(args, 1)))
            return unknown;
        Var* a = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 0))->var;
        Var* b = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 1))->var;
        if (a->dtype() != b->dtype() || !a->dtype().is_float()) return declines;
        // Not registered may only mean not loaded yet; Python loads it.
        return has_op(rule.op) ? accepts : unknown;
    }
    if (rule.kind == NativeRule::rms_training)
        return no_grad ? declines : unknown;
    if (rule.kind == NativeRule::ln_inference) {
        // `_supports_layer_norm_inference(x, normalized_shape, weight, bias, eps)`
        // for a Var weight and bias; scalar ones read the environment, so
        // Python answers those.
        if (n != 5 || !is_var(PyTuple_GET_ITEM(args, 0))) return unknown;
        PyObject* shape = PyTuple_GET_ITEM(args, 1);
        PyObject* weight = PyTuple_GET_ITEM(args, 2);
        PyObject* bias = PyTuple_GET_ITEM(args, 3);
        if (!PyTuple_Check(shape)) return unknown;
        Var* x = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 0))->var;
        bool affine = is_var(weight) && is_var(bias);
        if (!affine) return is_var(weight) || is_var(bias) ? declines : unknown;
        Var* w = GET_RAW_PTR(VarHolder, weight)->var;
        Var* b = GET_RAW_PTR(VarHolder, bias)->var;
        // Grad mode is the Python side's to judge.
        if (!no_grad && (requires_grad(x) || requires_grad(w) || requires_grad(b))) return unknown;
        if (x->dtype() != ns_float16 && x->dtype() != ns_float32) return declines;
        if (PyTuple_GET_SIZE(shape) != 1) return declines;
        PyObject* size = PyTuple_GET_ITEM(shape, 0);
        if (!PyLong_CheckExact(size) || !x->shape.size()) return unknown;
        int64 hidden = PyLong_AsLongLong(size);
        if (PyErr_Occurred()) { PyErr_Clear(); return unknown; }
        if (x->shape[x->shape.size() - 1] != hidden) return declines;
        if (w->num != hidden || b->num != hidden) return declines;
        return accepts;
    }
    if (rule.kind != NativeRule::rms_inference) return unknown;
    // `_rms_norm_contract(x, gamma, epsilon)`.
    if (n < 2 || n > 3 || !is_var(PyTuple_GET_ITEM(args, 0)) || !is_var(PyTuple_GET_ITEM(args, 1)))
        return unknown;
    Var* x = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 0))->var;
    Var* gamma = GET_RAW_PTR(VarHolder, PyTuple_GET_ITEM(args, 1))->var;
    double epsilon = 1e-6;
    if (n == 3) {
        PyObject* e = PyTuple_GET_ITEM(args, 2);
        if (!PyFloat_Check(e) && !PyLong_Check(e)) return unknown;
        epsilon = PyFloat_AsDouble(e);
        if (PyErr_Occurred()) { PyErr_Clear(); return unknown; }
    }
    // Grad mode and autocast are the Python side's to judge.
    if (!no_grad && (requires_grad(x) || requires_grad(gamma))) return unknown;
    if (amp_reg) return unknown;
    int rank = x->shape.size();
    if (!rank) return declines;
    for (int i = 0; i < rank; i++) if (x->shape[i] <= 0) return declines;
    int64 hidden = x->shape[rank - 1];
    if (hidden > 4096 || gamma->shape.size() != 1 || gamma->shape[0] != hidden) return declines;
    if (!is_floating(x->dtype()) || !is_floating(gamma->dtype())) return declines;
    if (!std::isfinite(epsilon) || epsilon <= 0) return declines;
    return accepts;
}

} // namespace

void kernel_select_native_rule(PyObject* fn, const string& rule) {
    NativeRule parsed;
    if (rule.compare(0, 11, "same_float:") == 0) {
        parsed.kind = NativeRule::same_float;
        parsed.op = rule.substr(11);
    } else if (rule == "rms_norm_inference") {
        parsed.kind = NativeRule::rms_inference;
    } else if (rule == "rms_norm_training") {
        parsed.kind = NativeRule::rms_training;
    } else if (rule == "layer_norm_inference") {
        parsed.kind = NativeRule::ln_inference;
    } else if (rule == "depthwise_conv2d") {
        parsed.kind = NativeRule::depthwise_conv2d;
    } else if (rule == "group_norm") {
        parsed.kind = NativeRule::group_norm;
    } else {
        throw std::invalid_argument("unknown native kernel rule: " + rule);
    }
    Py_INCREF(fn);
    auto found = native_rules.find(fn);
    if (found != native_rules.end()) Py_DECREF(fn);
    native_rules[fn] = parsed;
    // Resolved candidates carry their rule; take them again.
    for (auto& kv : resolved) release(kv.second);
    resolved.clear();
}

bool kernel_op_registered(const char* op) {
    // Not bound yet: nothing asked for a kernel, so say one may exist and
    // let the caller take the general path.
    if (!registered_ops) return true;
    PyObjHolder name(PyUnicode_FromString(op));
    return PySet_Contains(registered_ops, name.obj) != 0;
}

void kernel_select_bind(PyObject* candidates, PyObject* dtype_name) {
    Py_XINCREF(candidates);
    Py_XINCREF(dtype_name);
    Py_XDECREF(candidates_fn);
    Py_XDECREF(dtype_name_fn);
    candidates_fn = candidates;
    dtype_name_fn = dtype_name;
}

void kernel_select_invalidate(PyObject* op_names) {
    for (auto& kv : resolved) release(kv.second);
    resolved.clear();
    Py_XINCREF(op_names);
    Py_XDECREF(registered_ops);
    registered_ops = op_names;
}

PyObject* kernel_select(PyObject* op, PyObject* args, PyObject* kwargs) {
    if (!candidates_fn || !registered_ops)
        throw std::runtime_error("the native kernel selector is not bound");
    int present = PySet_Contains(registered_ops, op);
    if (present < 0) throw std::runtime_error("cannot test the registered operator set");
    if (!present) { Py_INCREF(Py_None); return Py_None; }
    vector<VarHolder*> tensors;
    collect(args, tensors, 0);
    if (kwargs && kwargs != Py_None && PyDict_Check(kwargs)) collect(kwargs, tensors, 0);
    vector<Var*> vars(tensors.size());
    bool placed = false;
    for (size_t i = 0; i < tensors.size(); i++) {
        vars[i] = tensors[i]->var;
        placed = placed || vars[i]->placement.explicit_backend;
    }
    // With no placed input the answer is where the result is being built:
    // the frontend's request, which this binding (it takes no Var) does not
    // enter a scope for. Without it `torch.arange()` on torch's default CPU
    // device picked the accelerator's kernel -- ACL's `index` code op -- and
    // then ran on the host, which that op has no source for.
    unique_ptr<TensorPlacementScope> requested;
    if (!placed) {
        TensorPlacement wanted = frontend_placement_request();
        if (wanted.explicit_backend) requested.reset(new TensorPlacementScope(wanted));
    }
    auto context = query_dispatch_context(vars);
    requested.reset();
    string backend = context.backend == "acl_legacy" ? "acl" : context.backend;
    const char* op_name = PyUnicode_AsUTF8(op);
    if (!op_name) throw std::runtime_error("kernel operator must be a str");
    const auto& entries = lookup(op_name, backend);
    vector<PyObject*> names;
    for (auto& entry : entries) {
        if (entry.runtime_modes) {
            long mode = backend == "cpu" ? 0 : (runtime_use_cuda() ? runtime_use_cuda() : 1);
            PyObjHolder boxed(PyLong_FromLong(mode));
            int in = PySet_Contains(entry.runtime_modes, boxed.obj);
            if (in < 0) throw std::runtime_error("runtime_modes must be a set");
            if (!in) continue;
        }
        if (entry.dtypes) {
            if (names.empty() && !tensors.empty())
                for (auto* t : tensors) names.push_back(canonical_dtype(t));
            bool all = true;
            for (auto* name : names) {
                int in = PySet_Contains(entry.dtypes, name);
                if (in < 0) throw std::runtime_error("dtypes must be a set");
                if (!in) { all = false; break; }
            }
            if (!all) continue;
        }
        if (entry.supports) {
            Verdict verdict = entry.rule.kind == NativeRule::none
                ? unknown : evaluate(entry.rule, args, kwargs);
            if (verdict == declines) continue;
            if (verdict == accepts) {
                Py_INCREF(entry.implementation);
                return entry.implementation;
            }
            PyObject* ok = PyObject_Call(entry.supports, args,
                                         kwargs && kwargs != Py_None ? kwargs : nullptr);
            if (!ok) return nullptr;
            int truth = PyObject_IsTrue(ok);
            Py_DECREF(ok);
            if (truth < 0) return nullptr;
            if (!truth) continue;
        }
        Py_INCREF(entry.implementation);
        return entry.implementation;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

} // namespace jittor
