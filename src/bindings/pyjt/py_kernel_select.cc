#include "bindings/pyjt/py_kernel_select.h"
#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "runtime/dispatch_context.h"
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
    enum Kind { none, same_float, rms_inference, rms_training } kind = none;
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
Verdict evaluate(const NativeRule& rule, PyObject* args, PyObject* kwargs) {
    if (!PyTuple_Check(args) || (kwargs && kwargs != Py_None && PyDict_GET_SIZE(kwargs)))
        return unknown;
    Py_ssize_t n = PyTuple_GET_SIZE(args);
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
    for (size_t i = 0; i < tensors.size(); i++) vars[i] = tensors[i]->var;
    auto context = query_dispatch_context(vars);
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
