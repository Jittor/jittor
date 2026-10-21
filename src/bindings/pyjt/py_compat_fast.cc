#include "bindings/pyjt/py_compat_fast.h"
#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "core/var_slices.h"
#include "bindings/pyjt/py_kernel_select.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "bindings/pyjt/py_type_lifetime.h"
#include "ops/op_register.h"
#include "ops/composite/code_op.h"
#include <cstring>
#include "runtime/device.h"
#include "runtime/backend.h"
#include "runtime/dispatch_context.h"
#include <stdexcept>
#include <descrobject.h>

namespace jittor {

DECLARE_FLAG(bool, no_grad);
DECLARE_FLAG(int, amp_reg);

namespace {
// Per frontend type: its dtype table, and the entries already looked up by
// native name. Dropped when the type is collected.
struct DtypeTable {
    PyObject* table = nullptr;
    std::unordered_map<string, PyObject*> by_name;
};
std::unordered_map<PyTypeObject*, DtypeTable> dtype_tables;

void forget_dtype_table(PyObject* type) {
    auto found = dtype_tables.find((PyTypeObject*)type);
    if (found == dtype_tables.end()) return;
    DtypeTable entry = move(found->second);
    dtype_tables.erase(found);
    for (auto& cached : entry.by_name) Py_DECREF(cached.second);
    Py_DECREF(entry.table);
}
} // namespace

PyObject* frontend_dtype(PyObject* self) {
    if (!PyObject_TypeCheck(self, &PyjtVarHolder.ht_type))
        throw std::runtime_error("_frontend_dtype needs a Var");
    PyTypeObject* type = Py_TYPE(self);
    auto found = dtype_tables.find(type);
    if (found == dtype_tables.end()) {
        DtypeTable entry;
        entry.table = PyObject_GetAttrString((PyObject*)type, "_frontend_dtype_objects");
        if (!entry.table) throw std::runtime_error("frontend type without a dtype table");
        try {
            on_type_collected((PyObject*)type, forget_dtype_table);
        } catch (...) {
            Py_DECREF(entry.table);
            throw;
        }
        found = dtype_tables.emplace(type, move(entry)).first;
    }
    auto& entry = found->second;
    const char* name = GET_RAW_PTR(VarHolder, self)->var->dtype().to_cstring();
    auto cached = entry.by_name.find(name);
    if (cached != entry.by_name.end()) {
        Py_INCREF(cached->second);
        return cached->second;
    }
    PyObjHolder key(PyUnicode_FromString(name));
    PyObject* value = PyObject_GetItem(entry.table, key.obj);
    if (!value) {
        PyErr_Clear();
        value = key.obj;
        Py_INCREF(value);
    }
    // The table is fixed for an installation; the entry keeps a reference.
    Py_INCREF(value);
    entry.by_name.emplace(name, value);
    return value;
}

namespace {
PyObject* tracing_fn = nullptr;
auto make_empty = op_constructor<VarPtr, NanoVector, NanoString>("empty");
auto make_setitem = op_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>("setitem");
auto make_getitem = op_constructor<VarPtr, Var*, VarSlices&&>("getitem");

bool basic_item(PyObject* item) {
    return PyLong_CheckExact(item) || PySlice_Check(item) || item == Py_None
        || item == Py_Ellipsis;
}

PyObject* not_implemented() {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}
} // namespace

void compat_fast_bind(PyObject* tracing) {
    Py_XINCREF(tracing);
    Py_XDECREF(tracing_fn);
    tracing_fn = tracing;
}

PyObject* fast_cat(PyObject* tensors, int64 dim) {
    if (!tracing_fn || !(PyList_CheckExact(tensors) || PyTuple_CheckExact(tensors)))
        return not_implemented();
    PyObjHolder seq(PySequence_Fast(tensors, "tensors"));
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.obj);
    PyObject** items = PySequence_Fast_ITEMS(seq.obj);
    if (n < 2 || n > 64) return not_implemented();
    vector<Var*> vars(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        if (!PyObject_TypeCheck(items[i], &PyjtVarHolder.ht_type)) return not_implemented();
        vars[i] = GET_RAW_PTR(VarHolder, items[i])->var;
    }
    Var* first = vars[0];
    int ndim = first->shape.size();
    if (ndim == 0) return not_implemented();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return not_implemented();
    NanoString dtype = first->dtype();
    if (dtype.is_unsigned()) return not_implemented();
    NanoVector shape = first->shape;
    int64 total = 0;
    for (auto* v : vars) {
        if (v->dtype() != dtype || (int)v->shape.size() != ndim || v->num <= 0)
            return not_implemented();
        for (int d = 0; d < ndim; d++)
            if (d != dim && v->shape[d] != shape[d]) return not_implemented();
        total += v->shape[dim];
    }
    // A tensor moved to another device keeps no placement to follow; there
    // `jittor.concat` switches devices for the allocation.
    int device = first->device_id;
    if (device >= 0 && device != current_device()) return not_implemented();
    if (kernel_op_registered("tensor.concat")) return not_implemented();
    PyObjHolder capturing(PyObject_CallObject(tracing_fn, nullptr));
    if (PyObject_IsTrue(capturing.obj)) return not_implemented();

    // The frontend and placement the generated bindings of `empty` and
    // `setitem` would each have entered.
    PyTensorFrontendScope scope(nullptr, items, n, false);
    NanoVector out_shape;
    for (int d = 0; d < ndim; d++) out_shape.push_back(d == dim ? total : shape[d]);
    VarPtr out = make_empty(out_shape, dtype);
    int64 offset = 0;
    for (auto* v : vars) {
        VarSlices slices(ndim);
        for (int d = 0; d < ndim; d++) {
            auto& s = slices.slices[d].slice;
            if (d == dim) {
                s.start = offset; s.stop = offset + v->shape[dim]; s.step = 1; s.mask = 4;
            } else {
                s.start = 0; s.stop = 0; s.step = 1; s.mask = 7;
            }
        }
        out = make_setitem(out, move(slices), v, ns_void);
        offset += v->shape[dim];
    }
    return to_py_object<VarHolder*>(new VarHolder(move(out)));
}

PyObject* fast_getitem(PyObject* self, PyObject* index) {
    if (!PyObject_TypeCheck(self, &PyjtVarHolder.ht_type)) return not_implemented();
    if (PyTuple_CheckExact(index)) {
        Py_ssize_t n = PyTuple_GET_SIZE(index);
        for (Py_ssize_t i = 0; i < n; i++)
            if (!basic_item(PyTuple_GET_ITEM(index, i))) return not_implemented();
    } else if (!basic_item(index)) {
        return not_implemented();
    }
    if (kernel_op_registered("tensor.getitem")) return not_implemented();
    auto* holder = GET_RAW_PTR(VarHolder, self);
    Var* var = holder->var;
    // A CPU-resident hint or a `.data` record changes what the result must
    // carry; the Python path owns those.
    if (var->placement.explicit_backend && var->placement.device.backend == BackendId::Cpu)
        return not_implemented();
    PyObject** dict_ptr = _PyObject_GetDictPtr(self);
    if (dict_ptr && *dict_ptr) {
        if (PyDict_GetItemString(*dict_ptr, "_jittor_torch_force_cpu")
                || PyDict_GetItemString(*dict_ptr, "_torch_data_owner"))
            return not_implemented();
        // The independent Tensor type keeps the same two facts in its object
        // state (`tensor_object_state.py`) behind properties; a `.data` view
        // there looked like a plain tensor, and a write through its index
        // never reached the tensor it was taken of.
        PyObject* state = PyDict_GetItemString(*dict_ptr, "_torch_object_state");
        if (state) {
            // An owner is a tensor, whose truth value is not a question to
            // ask: being there is the answer.
            PyObject* owner = PyObject_GetAttrString(state, "data_owner");
            if (!owner) { PyErr_Clear(); return not_implemented(); }
            bool viewed = owner != Py_None;
            Py_DECREF(owner);
            if (viewed) return not_implemented();
            PyObject* force_cpu = PyObject_GetAttrString(state, "force_cpu");
            if (!force_cpu) { PyErr_Clear(); return not_implemented(); }
            int forced = PyObject_IsTrue(force_cpu);
            Py_DECREF(force_cpu);
            if (forced != 0) { PyErr_Clear(); return not_implemented(); }
        }
    }
    vector<unique_ptr<VarHolder>> holders;
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    VarSlices slices = from_py_object<VarSlices>(index, holders);
    VarSlices recorded(slices);
    auto* out = new VarHolder(make_getitem(var, move(slices)));
    out->set_view_of(holder, move(recorded));
    return to_py_object<VarHolder*>(out);
}

namespace {
auto make_clamp_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
auto make_clamp_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
auto make_clamp_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
auto make_clamp_clone = op_constructor<VarPtr, Var*>("clone");

bool clamp_float_dtype(NanoString dtype) {
    return dtype == ns_float16 || dtype == ns_bfloat16 || dtype == ns_float32
        || dtype == ns_float64;
}

bool clamp_int_dtype(NanoString dtype) {
    return dtype == ns_int8 || dtype == ns_int16 || dtype == ns_int32 || dtype == ns_int64
        || dtype == ns_uint8 || dtype == ns_uint16 || dtype == ns_uint32 || dtype == ns_uint64;
}

// A Python number as the generated bindings receive one for a Var argument
// (`from_py_object<VarHolder*>(obj, holder)`): the scalar array, marked as a
// Python number.
VarPtr clamp_number(PyObject* value) {
    unique_ptr<VarHolder> holder(jit_op_maker::array__(value));
    mark_python_number(holder.get());
    return VarPtr(holder->var);
}

// `VarHolder::stop_grad`, on a Var no holder owns yet.
void clamp_stop_grad(Var* var) {
    if (var->autograd_alias) var->autograd_alias->own_requires_grad = false;
    var->set_stop_grad();
    var->set_flag(VarFlags::_requires_grad_disabled, 0);
    var->set_flag(VarFlags::_first_order_only, 0);
}

// `jt.unary(bound, dtype).stop_grad()`.
VarPtr clamp_bound(PyObject* bound, NanoString dtype) {
    VarPtr number = clamp_number(bound);
    VarPtr result = make_clamp_unary(number.ptr, dtype);
    clamp_stop_grad(result.ptr);
    return result;
}

// `prepare_bound` in `_clamp_cpu`, for a Python int or float bound.
VarPtr clamp_prepare(VarPtr& value, PyObject* bound) {
    if (clamp_float_dtype(value->dtype()))
        return clamp_bound(bound, value->dtype());
    if (PyFloat_CheckExact(bound)) {
        value = make_clamp_unary(value.ptr, ns_float32);
        return clamp_bound(bound, ns_float32);
    }
    return clamp_bound(bound, value->dtype());
}

// `select_bound` in `_clamp_cpu`.
VarPtr clamp_select(Var* value, Var* bound, bool lower) {
    VarPtr keep = make_clamp_binary(value, bound, lower ? ns_greater_equal : ns_less_equal);
    VarPtr result = make_clamp_ternary(keep.ptr, value, bound);
    if (clamp_float_dtype(value->dtype())) {
        VarPtr nan_value = make_clamp_clone(value);
        clamp_stop_grad(nan_value.ptr);
        VarPtr is_nan = make_clamp_binary(value, value, ns_not_equal);
        result = make_clamp_ternary(is_nan.ptr, nan_value.ptr, result.ptr);
    }
    return result;
}

bool clamp_scalar(PyObject* value) {
    if (value == Py_None || PyFloat_CheckExact(value)) return true;
    if (!PyLong_CheckExact(value)) return false;
    // The bindings read an int into an int32 (`ArrayOp(PyObject*)`); one that
    // does not fit is the Python path's to convert and report.
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (overflow || (v == -1 && PyErr_Occurred())) { PyErr_Clear(); return false; }
    return v >= std::numeric_limits<int32>::min() && v <= std::numeric_limits<int32>::max();
}
} // namespace

namespace {
auto make_where = op_constructor<vector<VarPtr>, Var*, NanoString>("where");

// An index Var `_fast_getitem` hands to the native getitem as it is: one
// that is not a mask.
bool plain_index_var(PyObject* item) {
    if (!PyObject_TypeCheck(item, &PyjtVarHolder.ht_type)) return false;
    NanoString dtype = GET_RAW_PTR(VarHolder, item)->var->dtype();
    return dtype != ns_bool && dtype != ns_uint8;
}

// `x[index]` for the index forms `_fast_getitem` (ops/indexing.py) passes to
// the native getitem because they hold index Vars: a lone bool mask, as its
// `where()` coordinates; a lone index Var; a tuple of ints, slices, None,
// Ellipsis and index Vars, at least one of them a Var. Each is a gather, so
// no view is recorded. The ops are the ones that path builds -- `where`, its
// coordinates submitted as their conversion to Python submits them, then
// `getitem` -- without the four Python frames, the dtype-name lookups and the
// coordinate tuple in between: a mask-driven loop indexes by a mask
// thousands of times per second.
//
// NotImplemented, and the Python path decides, for anything else: a uint8
// mask, a mask inside a tuple, a 0-d mask, a lone index of a Var subclass
// (whose `_jittor_constant_index_value` the Python path reads), or with a
// getitem -- or, for a mask, a where -- kernel registered.
PyObject* var_index_getitem(PyObject* self, PyObject* index) {
    if (kernel_op_registered("tensor.getitem")) return not_implemented();
    Var* x = GET_RAW_PTR(VarHolder, self)->var;
    if (PyObject_TypeCheck(index, &PyjtVarHolder.ht_type)) {
        if (Py_TYPE(index) != &PyjtVarHolder.ht_type) return not_implemented();
        Var* cond = GET_RAW_PTR(VarHolder, index)->var;
        NanoString dtype = cond->dtype();
        if (dtype == ns_uint8) return not_implemented();
        if (dtype != ns_bool) {
            // `_maybe_constant_index_gather` answers a lone 1-d index whose
            // Var carries a constant value, set as an instance attribute.
            if (cond->shape.size() == 1 && x->shape.size() >= 1) {
                PyObject** dict_ptr = _PyObject_GetDictPtr(index);
                if (dict_ptr && *dict_ptr
                        && PyDict_GetItemString(*dict_ptr, "_jittor_constant_index_value"))
                    return not_implemented();
            }
            PyTensorFrontendScope scope(self, nullptr, 0, false);
            VarSlices slices(1);
            slices.slices[0].set_var(cond);
            return to_py_object<VarHolder*>(new VarHolder(make_getitem(x, move(slices))));
        }
        if (cond->shape.size() == 0 || kernel_op_registered("tensor.where"))
            return not_implemented();
        // `cond.where()`, as `jittor_core.ops.where(cond)` builds and returns
        // it: the binding reads its Var argument through `grad_input`.
        cond = GET_RAW_PTR(VarHolder, index)->grad_input();
        vector<unique_ptr<VarHolder>> coords;
        {
            PyTensorFrontendScope scope(nullptr, &index, 1, false);
            for (auto& out : make_where(cond, ns_int64))
                coords.emplace_back(new VarHolder(move(out)));
        }
        for (auto& coord : coords) schedule_pending_from_python(coord.get());
        PyTensorFrontendScope scope(self, nullptr, 0, false);
        VarSlices slices(coords.size());
        for (size_t i = 0; i < coords.size(); i++) slices.slices[i].set_var(coords[i]->var);
        return to_py_object<VarHolder*>(new VarHolder(make_getitem(x, move(slices))));
    }
    if (!PyTuple_CheckExact(index)) return not_implemented();
    Py_ssize_t n = PyTuple_GET_SIZE(index);
    bool has_var = false;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyTuple_GET_ITEM(index, i);
        if (PySlice_Check(item)) {
            // `_plain_slice`: exact ints or None, nothing the full path
            // would convert first.
            auto* sl = (PySliceObject*)item;
            for (PyObject* part : {sl->start, sl->stop, sl->step})
                if (part != Py_None && !PyLong_CheckExact(part)) return not_implemented();
        } else if (plain_index_var(item)) {
            has_var = true;
        } else if (!(PyLong_CheckExact(item) || item == Py_None || item == Py_Ellipsis)) {
            return not_implemented();
        }
    }
    if (!has_var) return not_implemented();
    vector<unique_ptr<VarHolder>> holders;
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    VarSlices slices = from_py_object<VarSlices>(index, holders);
    return to_py_object<VarHolder*>(new VarHolder(make_getitem(x, move(slices))));
}
} // namespace

PyObject* jt_basic_getitem(PyObject* self, PyObject* index, bool gather) {
    if (!PyObject_TypeCheck(self, &PyjtVarHolder.ht_type)) return not_implemented();
    GET_RAW_PTR(VarHolder, self)->grad_input();
    if (gather && Py_TYPE(self) == &PyjtVarHolder.ht_type
            && (PyObject_TypeCheck(index, &PyjtVarHolder.ht_type) || PyTuple_CheckExact(index))) {
        PyObject* result = var_index_getitem(self, index);
        if (result != Py_NotImplemented) return result;
        Py_DECREF(result);
    }
    return fast_getitem(self, index);
}

namespace {
auto make_setitem_op = op_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>("setitem");
auto make_ternary_op = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
auto make_broadcast_to_op = op_constructor<VarPtr, Var*, NanoVector, NanoVector>("broadcast_to");

// An index `var_index_getitem` or the basic path would take as it is:
// ints, plain slices, None, Ellipsis and index Vars, alone or in a tuple.
bool settable_item(PyObject* item) {
    if (PyLong_CheckExact(item) || item == Py_None || item == Py_Ellipsis) return true;
    if (PySlice_Check(item)) {
        auto* sl = (PySliceObject*)item;
        for (PyObject* part : {sl->start, sl->stop, sl->step})
            if (part != Py_None && !PyLong_CheckExact(part)) return false;
        return true;
    }
    return plain_index_var(item);
}

bool exact_number(PyObject* value) {
    return PyLong_CheckExact(value) || PyFloat_CheckExact(value) || PyBool_Check(value);
}

// `x.assign(result)`, and `x` back to Python as the assignment returns it.
PyObject* assign_result(PyObject* self, VarPtr&& result) {
    unique_ptr<VarHolder> holder(new VarHolder(move(result)));
    // What returning the result to Python would have done with it.
    schedule_pending_from_python(holder.get());
    GET_RAW_PTR(VarHolder, self)->assign(holder.get());
    Py_INCREF(self);
    return self;
}
} // namespace

PyObject* jt_setitem(PyObject* self, PyObject* index, PyObject* value) {
    if (Py_TYPE(self) != &PyjtVarHolder.ht_type) return not_implemented();
    // Exact Vars only, for the value and a lone index: the bindings this
    // stands in for take their frontend type from a frontend tensor among
    // their direct arguments.
    bool value_var = Py_TYPE(value) == &PyjtVarHolder.ht_type;
    if (!value_var && !exact_number(value)) return not_implemented();
    if (kernel_op_registered("tensor.setitem")) return not_implemented();
    auto* holder = GET_RAW_PTR(VarHolder, self);
    if (holder->var->dtype() == ns_complex64) return not_implemented();
    // `_acl_assignment_value` converts the value for an ACL target.
    {
        auto backend = query_dispatch_context({holder->var}).backend;
        if (backend == "acl" || backend == "acl_legacy") return not_implemented();
    }
    bool mask = false;
    if (PyObject_TypeCheck(index, &PyjtVarHolder.ht_type)) {
        NanoString dtype = GET_RAW_PTR(VarHolder, index)->var->dtype();
        if (dtype == ns_uint8 || Py_TYPE(index) != &PyjtVarHolder.ht_type) return not_implemented();
        mask = dtype == ns_bool;
        if (mask && (GET_RAW_PTR(VarHolder, index)->var->shape.size() == 0
                     || kernel_op_registered("tensor.where")))
            return not_implemented();
    } else if (PyTuple_CheckExact(index)) {
        Py_ssize_t n = PyTuple_GET_SIZE(index);
        for (Py_ssize_t i = 0; i < n; i++)
            if (!settable_item(PyTuple_GET_ITEM(index, i))) return not_implemented();
    } else if (!settable_item(index)) {
        return not_implemented();
    }

    PyTensorFrontendScope scope(self, nullptr, 0, false);
    if (mask) {
        Var* x = holder->var;
        Var* cond = GET_RAW_PTR(VarHolder, index)->var;
        if (cond->shape == x->shape) {
            // `x.assign(mask.ternary(jt.array(value).broadcast(x.shape), x))`,
            // and for a one-element Var value `jt.broadcast(value, x.shape)`.
            VarPtr filled;
            if (!value_var) {
                unique_ptr<VarHolder> number(jit_op_maker::array__(value));
                filled = make_broadcast_to_op(number->var, x->shape, NanoVector());
            } else {
                auto* v = GET_RAW_PTR(VarHolder, value);
                if (v->var->shape.size() == 1 && v->var->shape[0] == 1)
                    filled = make_broadcast_to_op(v->grad_input(), x->shape, NanoVector());
            }
            if (filled) {
                Var* c = GET_RAW_PTR(VarHolder, index)->grad_input();
                return assign_result(self, make_ternary_op(c, filled.ptr, holder->grad_input()));
            }
        }
    }
    // `x.setitem(slices, value)` -- `var_setitem`, with a mask as its
    // `where()` coordinates -- through the generated binding.
    vector<unique_ptr<VarHolder>> holders;
    VarSlices slices;
    if (mask) {
        Var* cond = GET_RAW_PTR(VarHolder, index)->grad_input();
        {
            PyTensorFrontendScope where_scope(nullptr, &index, 1, false);
            for (auto& out : make_where(cond, ns_int64))
                holders.emplace_back(new VarHolder(move(out)));
        }
        for (auto& coord : holders) schedule_pending_from_python(coord.get());
        slices = VarSlices(holders.size());
        for (size_t i = 0; i < holders.size(); i++) slices.slices[i].set_var(holders[i]->var);
    } else {
        slices = from_py_object<VarSlices>(index, holders);
    }
    unique_ptr<VarHolder> number;
    VarHolder* y = from_py_object<VarHolder*>(value, number);
    return assign_result(self,
        make_setitem_op(holder->grad_input(), move(slices), y->grad_input(), ns_void));
}

PyObject* jt_concat(PyObject* tensors, int64 dim) {
    if (PyList_CheckExact(tensors) || PyTuple_CheckExact(tensors)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(tensors);
        PyObject** items = PySequence_Fast_ITEMS(tensors);
        for (Py_ssize_t i = 0; i < n; i++)
            if (PyObject_TypeCheck(items[i], &PyjtVarHolder.ht_type))
                GET_RAW_PTR(VarHolder, items[i])->grad_input();
    }
    return fast_cat(tensors, dim);
}

PyObject* fast_clamp(PyObject* self, PyObject* min_v, PyObject* max_v) {
    if (Py_TYPE(self) != &PyjtVarHolder.ht_type) return not_implemented();
    if (!clamp_scalar(min_v) || !clamp_scalar(max_v)) return not_implemented();
    if (min_v == Py_None && max_v == Py_None) return not_implemented();
    // Read as the bindings `_clamp_cpu` calls would read it.
    Var* x = GET_RAW_PTR(VarHolder, self)->grad_input();
    if (x->num <= 0) return not_implemented();
    NanoString dtype = x->dtype();
    bool is_float = clamp_float_dtype(dtype);
    if (!is_float && !clamp_int_dtype(dtype)) return not_implemented();
    if (kernel_op_registered("clamp.scalar")) return not_implemented();

    PyTensorFrontendScope scope(self, nullptr, 0, false);
    if (!is_float && PyLong_CheckExact(min_v) && PyLong_CheckExact(max_v)) {
        int ordered = PyObject_RichCompareBool(min_v, max_v, Py_LE);
        if (ordered < 0) { PyErr_Clear(); return not_implemented(); }
        if (ordered) {
            // The integer fast path: maximum then minimum at x's dtype.
            VarPtr low = clamp_bound(min_v, dtype);
            VarPtr high = clamp_bound(max_v, dtype);
            VarPtr result = make_clamp_binary(x, low.ptr, ns_maximum);
            result = make_clamp_binary(result.ptr, high.ptr, ns_minimum);
            return to_py_object<VarHolder*>(new VarHolder(move(result)));
        }
    }
    VarPtr value(x);
    VarPtr low, high;
    if (min_v != Py_None) low = clamp_prepare(value, min_v);
    if (max_v != Py_None) high = clamp_prepare(value, max_v);
    if (low) value = clamp_select(value.ptr, low.ptr, true);
    if (high) value = clamp_select(value.ptr, high.ptr, false);
    return to_py_object<VarHolder*>(new VarHolder(move(value)));
}

namespace {
enum BinaryCode { B_ADD, B_RADD, B_SUB, B_RSUB, B_MUL, B_RMUL, B_TRUEDIV, B_RTRUEDIV, B_COUNT };
binaryfunc binary_slots[B_COUNT] = {};
bool binary_reflected[B_COUNT] = {};
PyObject* mark_cpu_like_fn = nullptr;
// Whether a scalar divisor widens the division, as `_true_division` does on
// every backend but ACL (which has no float64 arithmetic).
bool widen_division = false;
auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
auto make_array = op_constructor<VarPtr, const void*, NanoVector, NanoString>("array");
auto make_reshape = op_constructor<VarPtr, Var*, NanoVector>("reshape");
auto make_transpose = op_constructor<VarPtr, Var*, NanoVector>("transpose");

bool is_var(PyObject* value) {
    return PyObject_TypeCheck(value, &PyjtVarHolder.ht_type);
}

bool is_floating(NanoString dtype) {
    return dtype == ns_float16 || dtype == ns_bfloat16 || dtype == ns_float32
        || dtype == ns_float64;
}

bool is_uint(NanoString dtype) {
    return dtype == ns_uint8 || dtype == ns_uint16 || dtype == ns_uint32
        || dtype == ns_uint64;
}

PyObject* none() {
    Py_INCREF(Py_None);
    return Py_None;
}

// `_mark_cpu_like`: a result without an explicit placement may still have to
// inherit a CPU residency hint from an operand, which the Python side keeps.
PyObject* mark_like(PyObject* out, PyObject* a, PyObject* b) {
    if (!is_var(out) || GET_RAW_PTR(VarHolder, out)->var->placement.explicit_backend)
        return out;
    PyObjHolder owned(out);
    return PyObject_CallFunctionObjArgs(mark_cpu_like_fn, out, a, b, nullptr);
}

// The new holder as a frontend tensor of `self`'s kind.
PyObject* wrap_like(PyObject* self, VarHolder* holder) {
    unique_ptr<VarHolder> owned(holder);
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    return to_py_object<VarHolder*>(owned.release());
}
} // namespace

void compat_fast_bind_binary(PyObject* natives, PyObject* mark_cpu_like, bool widen_scalar_division) {
    widen_division = widen_scalar_division;
    if (!PyTuple_Check(natives) || PyTuple_GET_SIZE(natives) != B_COUNT)
        throw std::runtime_error("_compat_fast_bind_binary needs one native per operator");
    for (int i = 0; i < B_COUNT; i++) {
        PyObject* native = PyTuple_GET_ITEM(natives, i);
        binary_slots[i] = nullptr;
        if (!Py_IS_TYPE(native, &PyWrapperDescr_Type)) continue;
        auto* descr = (PyWrapperDescrObject*)native;
        const char* name = descr->d_base->name;
        binary_slots[i] = (binaryfunc)descr->d_wrapped;
        binary_reflected[i] = name[0] == '_' && name[1] == '_' && name[2] == 'r';
    }
    Py_XINCREF(mark_cpu_like);
    Py_XDECREF(mark_cpu_like_fn);
    mark_cpu_like_fn = mark_cpu_like;
}

namespace {
// `_true_division` for a floating tensor over a Python float, op for op: the
// quotient is taken in float32 for half precision and float64 for float32 --
// the divisor a 0-d array of that dtype -- and cast back. In Python it built
// those three operators through a `result_type`, two dtype promotions and
// three install-context lookups, 21 us a call; every diffusers ResnetBlock2D
// ends in one (`/ self.output_scale_factor`), 45 a DDPM UNet forward.
PyObject* scalar_division(PyObject* self, PyObject* other, NanoString own) {
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    PyObject* out;
    if (own == ns_float64) {
        out = binary_slots[B_TRUEDIV](self, other);
    } else {
        NanoString wide = own == ns_float32 ? ns_float64 : ns_float32;
        double value = PyFloat_AS_DOUBLE(other);
        float narrow = (float)value;
        Var* x = GET_RAW_PTR(VarHolder, self)->var;
        PyObjHolder a(wrap_like(self, new VarHolder(make_unary(x, wide))));
        PyObjHolder b(wrap_like(self, new VarHolder(make_array(
            wide == ns_float64 ? (const void*)&value : (const void*)&narrow, {}, wide))));
        // This is a Python literal, not a trainable tensor operand. Native
        // floating ArrayOp defaults must not create a new autograd leaf.
        GET_RAW_PTR(VarHolder, b.obj)->stop_grad();
        out = binary_slots[B_TRUEDIV](a.obj, b.obj);
    }
    if (!out || out == Py_NotImplemented || !is_var(out)) return out;
    Var* result = GET_RAW_PTR(VarHolder, out)->var;
    if (result->dtype() != own) {
        PyObjHolder uncast(out);
        out = wrap_like(self, new VarHolder(make_unary(result, own)));
    }
    return mark_like(out, self, other);
}
} // namespace

PyObject* fast_binary(PyObject* self, PyObject* other, int code) {
    if (code < 0 || code >= B_COUNT || !binary_slots[code] || !mark_cpu_like_fn
            || !is_var(self))
        return none();
    NanoString own = GET_RAW_PTR(VarHolder, self)->var->dtype();
    bool division = code >= B_TRUEDIV;
    bool scalar = false;
    if (is_var(other)) {
        if (GET_RAW_PTR(VarHolder, other)->var->dtype() != own) return none();
        if (division ? !(is_floating(own) || own.is_complex()) : is_uint(own)) return none();
    } else if (division) {
        // A float divisor of a floating tensor, as `_true_division` widens
        // it; anything else -- an int, a reflected division -- stays there.
        if (code != B_TRUEDIV || !widen_division || !PyFloat_CheckExact(other)
                || !is_floating(own))
            return none();
        return scalar_division(self, other, own);
    } else if (PyFloat_CheckExact(other)) {
        if (!is_floating(own)) return none();
        scalar = true;
    } else if (PyLong_CheckExact(other)) {
        if (!is_floating(own) && !(own.is_int() && !is_uint(own) && own != ns_bool)
                && own != ns_uint8)
            return none();
        scalar = true;
    } else {
        return none();
    }
    PyObject* out = binary_reflected[code]
        ? binary_slots[code](other, self) : binary_slots[code](self, other);
    if (!out || out == Py_NotImplemented || !is_var(out)) return out;
    if (scalar) {
        // A scalar of these kinds leaves the tensor's dtype, as torch promotes.
        Var* result = GET_RAW_PTR(VarHolder, out)->var;
        if (result->dtype() != own) {
            PyObjHolder uncast(out);
            out = wrap_like(self, new VarHolder(make_unary(result, own)));
        }
    }
    return mark_like(out, self, other);
}

namespace {
auto make_code = op_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&,
    vector<string>&&, string&&, string&&, vector<string>&&, string&&, DataMap&&, string&&>("code");

// The body below as one kernel, in the same order and precision: float32
// throughout, the input widened and the result narrowed as `.float32()` and
// `.cast()` would. Taken for the output of a code operator -- a Linear's, in
// a transformer's MLP -- where the elementwise chain has nothing on its input
// side to fuse with, under no_grad on CUDA: one operator to build and launch
// instead of five.
const string& gelu_source() {
    static const string source = R"(
    __global__ static void jt_gelu(const in0_type* __restrict__ x, out0_type* __restrict__ y,
                                   long long n) {
        long long stride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            float v = static_cast<float>(x[i]);
            y[i] = static_cast<out0_type>((0.5f * v) * (1.0f + erff(v * 0.7071067811865476f)));
        }
    }
    long long n = in0->num;
    long long blocks = (n + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    jt_gelu<<<(unsigned)blocks, 256>>>(in0_p, out0_p, n);
    )";
    return source;
}
} // namespace

PyObject* fast_gelu(PyObject* x) {
    if (!is_var(x) || !binary_slots[B_MUL] || !binary_slots[B_RMUL] || !binary_slots[B_RADD]
            || !mark_cpu_like_fn || kernel_op_registered("nn.gelu"))
        return none();
    Var* var = GET_RAW_PTR(VarHolder, x)->var;
    NanoString own = var->dtype();
    if (!is_floating(own)) return none();
    bool low = own == ns_float16 || own == ns_bfloat16;
    Op* producer = var->input();
    // A CUDA kernel source: only where the accelerator is CUDA (`use_cuda` is
    // also set on ACL).
    if (own != ns_float64 && !amp_reg && runtime_flag_use_cuda()
            && accelerator_backend_id() == BackendId::Cuda && producer
            && producer->is_op(op_ids::code()) && var->num > 0 && var->is_contiguous()
            && (no_grad || var->is_stop_grad())) {
        PyTensorFrontendScope scope(x, nullptr, 0, false);
        unique_ptr<VarHolder> out(new VarHolder(make_code(var->shape, own, {var},
            "", {}, "", string(gelu_source()), {}, "", {}, "")));
        out->stop_grad();
        return to_py_object<VarHolder*>(out.release());
    }
    static PyObject* half = PyFloat_FromDouble(0.5);
    static PyObject* one = PyFloat_FromDouble(1.0);
    static PyObject* inv_sqrt2 = PyFloat_FromDouble(0.7071067811865476);
    // Owned references, any of which may be null (a step that declined).
    struct Ref {
        PyObject* obj = nullptr;
        ~Ref() { Py_XDECREF(obj); }
        // A declined step (None) becomes null; an error stays an error.
        bool take(PyObject* out) {
            obj = out;
            if (out && out != Py_None && is_var(out)) return true;
            Py_XDECREF(out);
            obj = nullptr;
            return false;
        }
    } compute, scaled, inner, erf, shifted, out;
    auto bail = []() { return PyErr_Occurred() ? nullptr : none(); };
    if (low) {
        compute.obj = wrap_like(x, new VarHolder(
            make_unary(GET_RAW_PTR(VarHolder, x)->var, ns_float32)));
    } else {
        Py_INCREF(x);
        compute.obj = x;
    }
    if (!compute.obj) return nullptr;
    if (!scaled.take(fast_binary(compute.obj, half, B_RMUL))) return bail();
    if (!inner.take(fast_binary(compute.obj, inv_sqrt2, B_MUL))) return bail();
    erf.obj = wrap_like(inner.obj, new VarHolder(
        make_unary(GET_RAW_PTR(VarHolder, inner.obj)->var, ns_erf)));
    if (!erf.obj) return nullptr;
    if (!shifted.take(fast_binary(erf.obj, one, B_RADD))) return bail();
    if (!out.take(fast_binary(scaled.obj, shifted.obj, B_MUL))) return bail();
    if (!low) {
        PyObject* result = out.obj;
        out.obj = nullptr;
        return result;
    }
    return wrap_like(out.obj, new VarHolder(
        make_unary(GET_RAW_PTR(VarHolder, out.obj)->var, own)));
}

namespace {
bool exact_int(PyObject* value, int64& out) {
    if (!PyLong_CheckExact(value)) return false;
    int overflow = 0;
    out = PyLong_AsLongLongAndOverflow(value, &overflow);
    return !overflow && !(out == -1 && PyErr_Occurred());
}

PyObject* storage_view(PyObject* self, const NanoVector& shape) {
    auto* holder = GET_RAW_PTR(VarHolder, self);
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    unique_ptr<VarHolder> out(new VarHolder(make_reshape(holder->grad_input(), shape)));
    out->set_storage_view_of(holder, false);
    return to_py_object<VarHolder*>(out.release());
}
} // namespace

PyObject* fast_view(PyObject* self, PyObject* shape) {
    if (!is_var(self) || !PyTuple_Check(shape)) return none();
    PyObject* items = shape;
    if (PyTuple_GET_SIZE(shape) == 1) {
        PyObject* first = PyTuple_GET_ITEM(shape, 0);
        if (PyTuple_CheckExact(first) || PyList_CheckExact(first)) items = first;
    }
    PyObjHolder seq(PySequence_Fast(items, "shape"));
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq.obj);
    if (n == 0 || n > 10) return none();
    PyObject** values = PySequence_Fast_ITEMS(seq.obj);
    NanoVector target;
    int inferred = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        int64 size;
        if (!exact_int(values[i], size)) { PyErr_Clear(); return none(); }
        if (size < -1) return none();
        if (size == -1 && inferred++) return none();
        target.push_back(size);
    }
    Var* var = GET_RAW_PTR(VarHolder, self)->var;
    if (!var->is_contiguous()) return none();
    return storage_view(self, target);
}

PyObject* fast_unsqueeze(PyObject* self, int64 dim) {
    if (!is_var(self)) return none();
    Var* var = GET_RAW_PTR(VarHolder, self)->var;
    int ndim = var->shape.size();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim || ndim >= 10 || !var->is_contiguous()) return none();
    NanoVector target;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) target.push_back(1);
        target.push_back(var->shape[i]);
    }
    if (dim == ndim) target.push_back(1);
    return storage_view(self, target);
}

namespace {
// `jittor.transpose(self, axes)` for a valid permutation.
PyObject* transpose_by(PyObject* self, VarHolder* holder, NanoVector axes) {
    int ndim = holder->var->shape.size();
    if (kernel_op_registered("tensor.transpose")) return none();
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    // A transpose of a transpose is one transpose of the source, and none at
    // all when the two cancel; see `jittor.transpose`.
    unique_ptr<VarHolder> source;
    NanoVector prior = holder->transpose_view_axes();
    if (prior.size()) {
        source.reset(holder->transpose_view_source());
    } else {
        prior = holder->producer_transpose_axes();
        if (prior.size()) source.reset(new VarHolder(holder->var->input()->inputs().front()));
    }
    VarHolder* base = holder;
    if (source && (int)prior.size() == ndim) {
        NanoVector composed;
        bool identity = true;
        for (int i = 0; i < ndim; i++) {
            composed.push_back(prior[axes[i]]);
            identity = identity && composed[i] == i;
        }
        if (identity) return to_py_object<VarHolder*>(source.release());
        base = source.get();
        axes = composed;
    }
    unique_ptr<VarHolder> out(new VarHolder(make_transpose(base->var, axes)));
    out->set_transpose_view_of(base, axes);
    return to_py_object<VarHolder*>(out.release());
}
} // namespace

PyObject* fast_transpose(PyObject* self, int64 dim0, int64 dim1) {
    if (!is_var(self)) return none();
    auto* holder = GET_RAW_PTR(VarHolder, self);
    int ndim = holder->var->shape.size();
    if (ndim == 0) return none();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) return none();
    NanoVector axes;
    for (int i = 0; i < ndim; i++) axes.push_back(i == dim0 ? dim1 : i == dim1 ? dim0 : i);
    return transpose_by(self, holder, axes);
}

PyObject* fast_permute(PyObject* self, PyObject* axes) {
    if (!is_var(self) || !(PyTuple_Check(axes) || PyList_Check(axes))) return none();
    auto* holder = GET_RAW_PTR(VarHolder, self);
    int ndim = holder->var->shape.size();
    if (ndim == 0 || PySequence_Fast_GET_SIZE(axes) != ndim) return none();
    PyObject** items = PySequence_Fast_ITEMS(axes);
    NanoVector order;
    uint64 seen = 0;
    for (int i = 0; i < ndim; i++) {
        if (!PyLong_CheckExact(items[i])) return none();
        long value = PyLong_AsLong(items[i]);
        if (value < 0 || value >= ndim || (seen >> value & 1)) { PyErr_Clear(); return none(); }
        seen |= uint64(1) << value;
        order.push_back(value);
    }
    return transpose_by(self, holder, order);
}

} // namespace jittor
