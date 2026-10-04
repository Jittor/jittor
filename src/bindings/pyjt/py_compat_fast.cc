#include "bindings/pyjt/py_compat_fast.h"
#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "core/var_slices.h"
#include "bindings/pyjt/py_kernel_select.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "runtime/backend.h"
#include <stdexcept>
#include <descrobject.h>

namespace jittor {

namespace {
// Per frontend type: its dtype table, and the entries already looked up by
// native name. A type is never freed while referenced here.
struct DtypeTable {
    PyObject* table = nullptr;
    std::unordered_map<string, PyObject*> by_name;
};
std::unordered_map<PyTypeObject*, DtypeTable> dtype_tables;
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
        Py_INCREF(type);
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

PyObject* fast_gelu(PyObject* x) {
    if (!is_var(x) || !binary_slots[B_MUL] || !binary_slots[B_RMUL] || !binary_slots[B_RADD]
            || !mark_cpu_like_fn || kernel_op_registered("nn.gelu"))
        return none();
    NanoString own = GET_RAW_PTR(VarHolder, x)->var->dtype();
    if (!is_floating(own)) return none();
    bool low = own == ns_float16 || own == ns_bfloat16;
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
    unique_ptr<VarHolder> out(new VarHolder(make_reshape(holder->var, shape)));
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

PyObject* fast_transpose(PyObject* self, int64 dim0, int64 dim1) {
    if (!is_var(self)) return none();
    auto* holder = GET_RAW_PTR(VarHolder, self);
    int ndim = holder->var->shape.size();
    if (ndim == 0) return none();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) return none();
    if (kernel_op_registered("tensor.transpose")) return none();
    NanoVector axes;
    for (int i = 0; i < ndim; i++) axes.push_back(i == dim0 ? dim1 : i == dim1 ? dim0 : i);
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

} // namespace jittor
