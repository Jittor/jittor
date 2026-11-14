#include "bindings/pyjt/py_compat_fast.h"
#include "bindings/pyjt/py_converter.h"
#include "core/var_holder.h"
#include "core/grad.h"
#include "core/var_slices.h"
#include "bindings/pyjt/py_kernel_select.h"
#include "bindings/pyjt/py_tensor_frontend.h"
#include "bindings/pyjt/py_type_lifetime.h"
#include "ops/op_register.h"
#include "core/op.h"
#include "ops/composite/code_op.h"
#include "mem/allocator.h"
#include <cstring>
#include "runtime/device.h"
#include "runtime/backend.h"
#include "runtime/dispatch_context.h"
#include "runtime/graph_capture.h"
#include "runtime/tensor_placement.h"
#include <stdexcept>
#include <descrobject.h>
#include <unordered_set>

namespace jittor {

DECLARE_FLAG(bool, no_grad);
DECLARE_FLAG(int, amp_reg);

PyObject* grad_optional_prepare(VarHolder* loss,
    const vector<VarHolder*>& targets, bool retain_graph, bool create_graph) {
    loss->observe_alias_history();
    vector<Var*> inputs;
    inputs.reserve(targets.size());
    for (auto* target : targets) {
        target->observe_alias_history();
        inputs.push_back(target->var);
    }
    // Use the same differentiation and alias observation as grad_optional.
    // No forward, backward graph or gradient value survives between calls.
    auto grads = grad(loss->var, inputs, retain_graph, false);
    bool prepared = !graph_capture_active();
    for (size_t i = 0; i < grads.size(); ++i)
        if (grads[i] && grads[i]->dtype() != inputs[i]->dtype()) prepared = false;
    vector<Maybe<VarHolder>> holders;
    holders.reserve(grads.size());
    for (auto& value : grads) {
        if (!value) {
            holders.emplace_back(nullptr);
            continue;
        }
        if (prepared && !create_graph) {
            auto placement = value->placement;
            TensorPlacementScope scope(placement.explicit_backend
                ? placement : current_tensor_placement());
            value = detach(value);
        }
        auto* holder = new VarHolder(move(value));
        // Torch detach stops the output holder, including its alias policy;
        // stopping just the clone operator is insufficient.
        if (prepared && !create_graph) holder->stop_grad();
        holders.emplace_back(holder);
    }
    PyObjHolder result(to_py_object(holders));
    return PyTuple_Pack(2, result.obj, prepared ? Py_True : Py_False);
}

bool try_optimizer_copies(PyObject* targets, PyObject* values) {
    if (!no_grad || graph_capture_active() || !runtime_use_cuda()
            || accelerator_backend_id() != BackendId::Cuda
            || !PyList_CheckExact(targets) || !PyList_CheckExact(values))
        return false;
    auto count = PyList_GET_SIZE(targets);
    if (!count || count != PyList_GET_SIZE(values)) return false;
    // A backend copy can release the GIL. Caller-owned lists may then change;
    // immutable snapshots keep every holder alive throughout the whole batch.
    PyObjHolder target_owners(PyList_AsTuple(targets));
    PyObjHolder value_owners(PyList_AsTuple(values));
    vector<VarHolder*> dst, src;
    dst.reserve(count);
    src.reserve(count);
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* a = PyTuple_GET_ITEM(target_owners.obj, i);
        auto* b = PyTuple_GET_ITEM(value_owners.obj, i);
        if (!PyObject_TypeCheck(a, &PyjtVarHolder.ht_type)
                || !PyObject_TypeCheck(b, &PyjtVarHolder.ht_type)) return false;
        auto* d = GET_RAW_PTR(VarHolder, a);
        auto* s = GET_RAW_PTR(VarHolder, b);
        auto* dv = d->var;
        auto* sv = s->var;
        if (d->view || s->view || !dv->is_finished() || !sv->is_finished()
                || dv->flag(VarFlags::_kept) || sv->flag(VarFlags::_kept)
                || !dv->mem_ptr || !sv->mem_ptr || !dv->allocator || !sv->allocator
                || !dv->allocator->is_cuda() || !sv->allocator->is_cuda()
                || dv->device_id != sv->device_id || dv->dtype() != sv->dtype()
                || dv->shape != sv->shape || !dv->is_contiguous() || !sv->is_contiguous())
            return false;
        dst.push_back(d);
        src.push_back(s);
    }
    // Retain logical-alias refresh, saved-reader/version checks, source
    // synchronization and stream ordering in the existing copy primitives.
    for (Py_ssize_t i = 0; i < count; ++i) {
        if (!dst[i]->copy_view_into(src[i])) dst[i]->copy_into(src[i], true);
    }
    return true;
}

namespace {
bool canonical_optimizer_holder(PyObject* value, PyObject* types) {
    if (!PyTuple_CheckExact(types)) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(types); ++i)
        if ((PyObject*)Py_TYPE(value) == PyTuple_GET_ITEM(types, i))
            return PyObject_TypeCheck(value, &PyjtVarHolder.ht_type);
    return false;
}

PyObject* no_optimizer_metadata_batch() { Py_RETURN_NONE; }

void append_optimizer_index(PyObject* result, Py_ssize_t index) {
    PyObjHolder item(PyLong_FromSsize_t(index));
    if (!item.obj || PyList_Append(result, item.obj) < 0)
        throw std::runtime_error("could not publish optimizer metadata indices");
}
}

PyObject* try_optimizer_shape_changes(PyObject* params, PyObject* moments,
    PyObject* variances, PyObject* canonical_types) {
    if ((!PyList_CheckExact(params) && !PyTuple_CheckExact(params))
            || !PyList_CheckExact(moments)
            || !PyList_CheckExact(variances)) return no_optimizer_metadata_batch();
    auto parameter = [&](Py_ssize_t i) {
        return PyTuple_CheckExact(params) ? PyTuple_GET_ITEM(params, i)
                                         : PyList_GET_ITEM(params, i);
    };
    auto count = std::min(PyTuple_CheckExact(params) ? PyTuple_GET_SIZE(params)
                                                   : PyList_GET_SIZE(params),
        std::min(PyList_GET_SIZE(moments), PyList_GET_SIZE(variances)));
    // Validate before reading anything. A Python-owned shape must be observed
    // only by the original loop, including when it is in a later slot.
    for (Py_ssize_t i = 0; i < count; ++i) {
        if (!canonical_optimizer_holder(parameter(i), canonical_types)
                || !canonical_optimizer_holder(PyList_GET_ITEM(moments, i), canonical_types)
                || !canonical_optimizer_holder(PyList_GET_ITEM(variances, i), canonical_types))
            return no_optimizer_metadata_batch();
    }
    PyObjHolder result(PyList_New(0));
    for (Py_ssize_t i = 0; i < count; ++i) {
        const auto& shape = GET_RAW_PTR(VarHolder, parameter(i))->var->shape;
        if (shape != GET_RAW_PTR(VarHolder, PyList_GET_ITEM(moments, i))->var->shape
                || shape != GET_RAW_PTR(VarHolder, PyList_GET_ITEM(variances, i))->var->shape)
            append_optimizer_index(result.obj, i);
    }
    return result.release();
}


namespace {
bool optimizer_string_keys(PyObject* dict) {
    if (!PyDict_CheckExact(dict)) return false;
    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (PyDict_Next(dict, &position, &key, &value))
        if (!PyUnicode_CheckExact(key)) return false;
    return true;
}

PyObject* optimizer_type_field(PyObject* type, const char* name) {
    PyObjHolder key;
    key.obj = PyUnicode_FromString(name);
    return key.obj ? _PyType_Lookup((PyTypeObject*)type, key.obj) : nullptr;
}

bool optimizer_function_code(PyObject* function, PyObject* code) {
    return code == Py_None || (PyFunction_Check(function)
        && PyFunction_GET_CODE(function) == code);
}

bool optimizer_function_bindings(PyObject* function, PyObject* expected) {
    if (!function || !PyFunction_Check(function) || !PyTuple_CheckExact(expected)) return false;
    auto* globals = PyFunction_GET_GLOBALS(function);
    if (!optimizer_string_keys(globals)) return false;
    PyObjHolder builtins;
#if PY_VERSION_HEX >= 0x030A0000
    builtins.obj = PyObject_GetAttrString(function, "__builtins__");
    if (!builtins.obj) return false;
#else
    // Older interpreters have no function.__builtins__; their normal Python
    // path remains available rather than guessing the function's environment.
    return false;
#endif
    if (!optimizer_string_keys(builtins.obj)) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(expected); ++i) {
        auto* pair = PyTuple_GET_ITEM(expected, i);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 2) return false;
        auto* key = PyTuple_GET_ITEM(pair, 0);
        if (!PyUnicode_CheckExact(key)) return false;
        auto* value = PyDict_GetItemWithError(globals, key);
        if (!value) value = PyDict_GetItemWithError(builtins.obj, key);
        if (value != PyTuple_GET_ITEM(pair, 1)) return false;
    }
    return true;
}

bool optimizer_adam_protocol(PyObject* protocol, PyObject* types) {
    if (!PyTuple_CheckExact(protocol) || PyTuple_GET_SIZE(protocol) != 10
            || !PyTuple_CheckExact(types) || PyTuple_GET_SIZE(types) != 3)
        return false;
    auto* step_type = PyTuple_GET_ITEM(protocol, 0);
    auto* steps_type = PyTuple_GET_ITEM(protocol, 1);
    if (!PyType_Check(step_type) || !PyType_Check(steps_type)) return false;
    auto* contracts = PyTuple_GET_ITEM(protocol, 2);
    if (!PyTuple_CheckExact(contracts)) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(contracts); ++i) {
        auto* contract = PyTuple_GET_ITEM(contracts, i);
        if (!PyTuple_CheckExact(contract) || PyTuple_GET_SIZE(contract) != 4)
            return false;
        auto* owner = PyTuple_GET_ITEM(contract, 0);
        auto* name = PyTuple_GET_ITEM(contract, 1);
        auto* method = PyTuple_GET_ITEM(contract, 2);
        if (!PyType_Check(owner) || !PyUnicode_CheckExact(name)
                || _PyType_Lookup((PyTypeObject*)owner, name) != method
                || !optimizer_function_code(method, PyTuple_GET_ITEM(contract, 3)))
            return false;
    }
    if (optimizer_type_field(step_type, "value")
            || optimizer_type_field(step_type, "counter")) return false;
    auto* globals = PyTuple_GET_ITEM(protocol, 7);
    if (!optimizer_string_keys(globals)
            || PyDict_GetItemString(globals, "_AdamStep") != step_type
            || PyDict_GetItemString(globals, "_AdamSteps") != steps_type
            || PyDict_GetItemString(globals, "jt") != PyTuple_GET_ITEM(protocol, 8))
        return false;
    auto* runtime = PyTuple_GET_ITEM(protocol, 8);
    if (!PyModule_Check(runtime)
            || Py_TYPE(runtime)->tp_getattro != PyModule_Type.tp_getattro
            || PyModule_GetDict(runtime) == nullptr
            || PyDict_GetItemString(PyModule_GetDict(runtime), "Var") != PyTuple_GET_ITEM(types, 0)
            || Py_TYPE(PyTuple_GET_ITEM(types, 0)) != &PyType_Type)
        return false;
    auto* builtins = PyTuple_GET_ITEM(protocol, 9);
    auto* int_method = optimizer_type_field(step_type, "__int__");
    auto* set_method = optimizer_type_field(steps_type, "__setitem__");
    if (!optimizer_function_bindings(int_method, builtins)
            || !optimizer_function_bindings(set_method, builtins)) return false;
    auto* getter = PyTuple_GET_ITEM(protocol, 6);
    if (!PyTuple_CheckExact(getter) || PyTuple_GET_SIZE(getter) != 2
            || !optimizer_function_code(PyTuple_GET_ITEM(getter, 0),
                                        PyTuple_GET_ITEM(getter, 1))) return false;
    auto* function = PyTuple_GET_ITEM(getter, 0);
    if (!optimizer_function_bindings(function, builtins)) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(types); ++i) {
        auto* type = PyTuple_GET_ITEM(types, i);
        if (!PyType_Check(type)
                || Py_TYPE(type)->tp_getattro != PyType_Type.tp_getattro
                || optimizer_type_field(type, "shape") != PyTuple_GET_ITEM(protocol, 3)
                || optimizer_type_field(type, "__getattribute__") != PyTuple_GET_ITEM(protocol, 5))
            return false;
        auto* descriptor = optimizer_type_field(type, "requires_grad");
        if (descriptor == PyTuple_GET_ITEM(protocol, 4)) continue;
        if (!descriptor || Py_TYPE(descriptor) != &PyProperty_Type
                || optimizer_type_field(type, "_frontend_native_requires_grad")
                    != PyTuple_GET_ITEM(protocol, 4)) return false;
        PyObjHolder fget;
        fget.obj = PyObject_GetAttrString(descriptor, "fget");
        if (fget.obj != function) return false;
    }
    return !PyErr_Occurred();
}

PyObject* optimizer_adam_counter(PyObject* value, PyObject* step_type,
                                 bool allow_integer) {
    if (allow_integer && PyLong_CheckExact(value)) return value;
    if ((PyObject*)Py_TYPE(value) != step_type) return nullptr;
    auto** pointer = _PyObject_GetDictPtr(value);
    if (!pointer || !*pointer || !optimizer_string_keys(*pointer)
            || PyDict_GetItemString(*pointer, "set")) return nullptr;
    auto* counter = PyDict_GetItemString(*pointer, "counter");
    auto* number = PyDict_GetItemString(*pointer, "value");
    return counter == Py_None && number && PyLong_CheckExact(number) ? number : nullptr;
}
}

PyObject* try_optimizer_adam_entries(PyObject* group, PyObject* gradients,
    PyObject* steps, PyObject* types, PyObject* protocol, bool advance) {
    if (!runtime_use_cuda() || accelerator_backend_id() != BackendId::Cuda
            || graph_capture_active() || !optimizer_string_keys(group)
            || !PyList_CheckExact(gradients) || !PyList_Check(steps))
        return no_optimizer_metadata_batch();
    auto* params = PyDict_GetItemString(group, "params");
    auto* moments = PyDict_GetItemString(group, "m");
    auto* variances = PyDict_GetItemString(group, "values");
    if (!params || !moments || !variances
            || (!PyList_CheckExact(params) && !PyTuple_CheckExact(params))
            || !PyList_CheckExact(moments) || !PyList_CheckExact(variances))
        return no_optimizer_metadata_batch();
    auto count = PyTuple_CheckExact(params) ? PyTuple_GET_SIZE(params) : PyList_GET_SIZE(params);
    PyObjHolder result;
    result.obj = PyList_New(count);
    if (!result.obj) return nullptr;
    // Allocate tracked Python containers before validating the live metadata.
    // No GC-triggering allocation then interrupts native holder/counter reads.
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* item = advance ? PyLong_FromSsize_t(i) : PyTuple_New(5);
        if (!item) return nullptr;
        PyList_SET_ITEM(result.obj, i, item);
    }
    if (!optimizer_adam_protocol(protocol, types))
        return PyErr_Occurred() ? nullptr : no_optimizer_metadata_batch();
    if ((PyObject*)Py_TYPE(steps) != PyTuple_GET_ITEM(protocol, 1))
        return no_optimizer_metadata_batch();
    // Allocation above may have run a finalizer. Re-read every group container
    // and validate the entire call before observing a holder or changing steps.
    if (!optimizer_string_keys(group)) return no_optimizer_metadata_batch();
    params = PyDict_GetItemString(group, "params");
    moments = PyDict_GetItemString(group, "m");
    variances = PyDict_GetItemString(group, "values");
    if (!params || !moments || !variances
            || (!PyList_CheckExact(params) && !PyTuple_CheckExact(params))
            || !PyList_CheckExact(moments) || !PyList_CheckExact(variances)
            || (PyTuple_CheckExact(params) ? PyTuple_GET_SIZE(params) : PyList_GET_SIZE(params)) != count
            || PyList_GET_SIZE(gradients) != count || PyList_GET_SIZE(moments) != count
            || PyList_GET_SIZE(variances) != count || PyList_GET_SIZE(steps) < count)
        return no_optimizer_metadata_batch();
    auto parameter = [&](Py_ssize_t i) {
        return PyTuple_CheckExact(params) ? PyTuple_GET_ITEM(params, i) : PyList_GET_ITEM(params, i);
    };
    auto* step_type = PyTuple_GET_ITEM(protocol, 0);
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* gradient = PyList_GET_ITEM(gradients, i);
        if (!canonical_optimizer_holder(parameter(i), types)
                || (gradient != Py_None && !canonical_optimizer_holder(gradient, types))
                || !optimizer_adam_counter(PyList_GET_ITEM(steps, i), step_type, !advance))
            return no_optimizer_metadata_batch();
    }
    Py_ssize_t used = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* p = parameter(i);
        auto* gradient = PyList_GET_ITEM(gradients, i);
        auto* holder = GET_RAW_PTR(VarHolder, p);
        if (!holder->get_requires_grad() || gradient == Py_None
                || holder->var->shape != GET_RAW_PTR(VarHolder, gradient)->var->shape)
            continue;
        auto* counter = PyList_GET_ITEM(steps, i);
        auto* number = optimizer_adam_counter(counter, step_type, !advance);
        if (advance) {
            PyObjHolder one, next;
            one.obj = PyLong_FromLong(1);
            if (!one.obj) return nullptr;
            next.obj = PyNumber_Add(number, one.obj);
            if (!next.obj || PyDict_SetItemString(*_PyObject_GetDictPtr(counter), "value", next.obj) < 0)
                return nullptr;
            auto* index = PyList_GET_ITEM(result.obj, i);
            Py_INCREF(index);
            auto* previous = PyList_GET_ITEM(result.obj, used);
            PyList_SET_ITEM(result.obj, used, index);
            Py_DECREF(previous);
        } else {
            auto* row = PyList_GET_ITEM(result.obj, used);
            PyObject* values[] = {p, PyList_GET_ITEM(moments, i),
                PyList_GET_ITEM(variances, i), gradient, number};
            for (int j = 0; j < 5; ++j) {
                Py_INCREF(values[j]);
                PyTuple_SET_ITEM(row, j, values[j]);
            }
        }
        ++used;
    }
    if (PyList_SetSlice(result.obj, used, count, nullptr) < 0) return nullptr;
    return result.release();
}


namespace {
bool gradient_property_function(PyObject* function, PyObject* code,
    PyObject* closure, PyObject* protocol) {
    if (!PyFunction_Check(function) || PyFunction_GET_CODE(function) != code
            || !PyTuple_CheckExact(closure)) return false;
    auto* cells = PyFunction_GET_CLOSURE(function);
    if (!cells || PyTuple_GET_SIZE(cells) != PyTuple_GET_SIZE(closure)) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(cells); ++i) {
        auto* actual = PyCell_GET(PyTuple_GET_ITEM(cells, i));
        auto* expected = PyTuple_GET_ITEM(closure, i);
        if (actual == expected) continue;
        if (!actual || !PyUnicode_CheckExact(actual) || !PyUnicode_CheckExact(expected)
                || PyUnicode_Compare(actual, expected) != 0) return false;
    }
    auto* globals = PyFunction_GET_GLOBALS(function);
    return optimizer_string_keys(globals)
        && PyDict_GetItemString(globals, "get_tensor_object_state") == PyTuple_GET_ITEM(protocol, 6)
        && optimizer_function_bindings(function, PyTuple_GET_ITEM(protocol, 8));
}

bool gradient_publication_protocol(PyObject* types, PyObject* protocol) {
    if (!PyTuple_CheckExact(protocol) || PyTuple_GET_SIZE(protocol) != 11) return false;
    auto* state_type = PyTuple_GET_ITEM(protocol, 0);
    auto* getter = PyTuple_GET_ITEM(protocol, 6);
    if (!PyType_Check(state_type)
            || ((PyTypeObject*)state_type)->tp_getattro != PyObject_GenericGetAttr
            || ((PyTypeObject*)state_type)->tp_setattro != PyObject_GenericSetAttr
            || optimizer_type_field(state_type, "grad") != PyTuple_GET_ITEM(protocol, 1)
            || Py_TYPE(PyTuple_GET_ITEM(protocol, 1)) != &PyMemberDescr_Type
            || !optimizer_function_code(getter, PyTuple_GET_ITEM(protocol, 7))
            || !optimizer_function_bindings(getter, PyTuple_GET_ITEM(protocol, 8))) return false;
    auto* globals = PyFunction_GET_GLOBALS(getter);
    auto* key = PyDict_GetItemString(globals, "_STATE_KEY");
    if (PyDict_GetItemString(globals, "TensorObjectState") != state_type
            || !key || !PyUnicode_CheckExact(key)
            || PyUnicode_CompareWithASCIIString(key, "_torch_object_state") != 0) return false;
    // Read the same property and underlying holder dict as vars()/object's
    // setter. A subclass, replaced descriptor, closure or helper stays Python.
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(types); ++i) {
        auto* type = PyTuple_GET_ITEM(types, i);
        if (optimizer_type_field(type, "__dict__") != PyTuple_GET_ITEM(protocol, 9)
                || optimizer_type_field(type, "stop_grad") != PyTuple_GET_ITEM(protocol, 10))
            return false;
        // Legacy native Vars have no frontend state property; they cannot
        // participate in this batch, but remain a canonical gradient type.
        auto* descriptor = optimizer_type_field(type, "_torch_grad");
        if (!descriptor) continue;
        if (Py_TYPE(descriptor) != &PyProperty_Type) return false;
        PyObjHolder get(PyObject_GetAttrString(descriptor, "fget"));
        PyObjHolder set(PyObject_GetAttrString(descriptor, "fset"));
        if (!get.obj || !set.obj
                || !gradient_property_function(get.obj, PyTuple_GET_ITEM(protocol, 2),
                    PyTuple_GET_ITEM(protocol, 3), protocol)
                || !gradient_property_function(set.obj, PyTuple_GET_ITEM(protocol, 4),
                    PyTuple_GET_ITEM(protocol, 5), protocol)) return false;
    }
    return !PyErr_Occurred();
}
}

namespace {
bool optimizer_batch_policy(PyObject* protocol, PyObject* writer) {
    if (!PyTuple_CheckExact(protocol) || PyTuple_GET_SIZE(protocol) != 3
            || !PyFunction_Check(PyTuple_GET_ITEM(protocol, 0))) return false;
    auto* function = PyTuple_GET_ITEM(protocol, 0);
    return optimizer_function_code(function, PyTuple_GET_ITEM(protocol, 1))
        && optimizer_function_bindings(function, PyTuple_GET_ITEM(protocol, 2))
        && PyDict_GetItemString(PyFunction_GET_GLOBALS(function), "_optimizer_update") == writer;
}

bool optimizer_plain_type_dicts(PyTypeObject* type,
                               std::unordered_set<PyTypeObject*>& checked) {
    if (!type->tp_mro || !PyTuple_CheckExact(type->tp_mro)) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(type->tp_mro); ++i) {
        auto* base = PyTuple_GET_ITEM(type->tp_mro, i);
        if (!PyType_Check(base)) return false;
        auto* cls = (PyTypeObject*)base;
        if (checked.insert(cls).second
                && (!cls->tp_dict || !optimizer_string_keys(cls->tp_dict)))
            return false;
    }
    return true;
}

// Ordinary function lookup cannot run Python. A metaclass or descriptor can;
// leave those to the original getter rather than calling them speculatively.
int optimizer_plain_writer(PyTypeObject* type, PyObject* writer, PyObject* key,
                           std::unordered_set<PyTypeObject*>& checked) {
    auto* meta = Py_TYPE(type);
    if (meta->tp_getattro != PyType_Type.tp_getattro
            || !optimizer_plain_type_dicts(type, checked)
            || !optimizer_plain_type_dicts(meta, checked)
            || _PyType_Lookup(meta, key)) return -1;
    auto* found = _PyType_Lookup(type, key);
    if (found == writer) return 1;
    return !found || PyFunction_Check(found) ? 0 : -1;
}
}

PyObject* try_pack_optimizer_updates(PyObject* active, PyObject* updates,
    PyObject* types, PyObject* writer, PyObject* protocol) {
    auto sequence = [](PyObject* value) {
        return PyList_CheckExact(value) || PyTuple_CheckExact(value);
    };
    if (!sequence(active) || !sequence(updates) || !PyTuple_CheckExact(types)
            || !PyFunction_Check(writer)) Py_RETURN_NONE;
    auto size = [](PyObject* value) {
        return PyTuple_CheckExact(value) ? PyTuple_GET_SIZE(value) : PyList_GET_SIZE(value);
    };
    auto item = [](PyObject* value, Py_ssize_t i) {
        return PyTuple_CheckExact(value) ? PyTuple_GET_ITEM(value, i) : PyList_GET_ITEM(value, i);
    };
    auto count = size(active);
    if (count != size(updates) || count > PY_SSIZE_T_MAX / 3) Py_RETURN_NONE;
    PyObjHolder targets(PyList_New(count * 3)), values(PyList_New(count * 3));
    if (!targets.obj || !values.obj) return nullptr;
    PyObjHolder result(PyTuple_Pack(2, targets.obj, values.obj));
    if (!result.obj) return nullptr;
    PyObjHolder key(PyUnicode_FromString("_optimizer_update"));
    if (!key.obj) return nullptr;
    // Tracked allocations can run a finalizer. Read the original live lists
    // again afterwards; do not use an earlier parameter/update snapshot.
    if (size(active) != count || size(updates) != count
            || !optimizer_batch_policy(protocol, writer)) {
        if (PyErr_Occurred()) return nullptr;
        Py_RETURN_NONE;
    }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(types); ++i) {
        auto* type = PyTuple_GET_ITEM(types, i);
        if (!PyType_Check(type)
                || Py_TYPE(type)->tp_richcompare != PyType_Type.tp_richcompare)
            Py_RETURN_NONE;
    }
    std::unordered_set<PyTypeObject*> checked, writers;
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* row = item(active, i);
        auto* triple = item(updates, i);
        if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 5
                || !PyTuple_CheckExact(triple) || PyTuple_GET_SIZE(triple) != 3)
            Py_RETURN_NONE;
        for (int j = 0; j < 3; ++j) {
            auto* target = PyTuple_GET_ITEM(row, j);
            auto* value = PyTuple_GET_ITEM(triple, j);
            if (!canonical_optimizer_holder(target, types)
                    || !canonical_optimizer_holder(value, types)) Py_RETURN_NONE;
            auto* type = Py_TYPE(target);
            if (writers.insert(type).second
                    && optimizer_plain_writer(type, writer, key.obj, checked) != 1) {
                if (PyErr_Occurred()) return nullptr;
                Py_RETURN_NONE;
            }
            Py_INCREF(target);
            PyList_SET_ITEM(targets.obj, i * 3 + j, target);
            Py_INCREF(value);
            PyList_SET_ITEM(values.obj, i * 3 + j, value);
        }
    }
    return result.release();
}

PyObject* try_optimizer_writers(PyObject* targets, PyObject* writer, PyObject* protocol) {
    if ((!PyList_CheckExact(targets) && !PyTuple_CheckExact(targets))
            || !PyFunction_Check(writer)) Py_RETURN_NONE;
    PyObjHolder key(PyUnicode_FromString("_optimizer_update"));
    if (!key.obj) return nullptr;
    if (!optimizer_batch_policy(protocol, writer)) {
        if (PyErr_Occurred()) return nullptr;
        Py_RETURN_NONE;
    }
    std::unordered_set<PyTypeObject*> checked, writers;
    auto count = PyTuple_CheckExact(targets) ? PyTuple_GET_SIZE(targets) : PyList_GET_SIZE(targets);
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto* target = PyTuple_CheckExact(targets)
            ? PyTuple_GET_ITEM(targets, i) : PyList_GET_ITEM(targets, i);
        auto* type = Py_TYPE(target);
        if (!writers.insert(type).second) continue;
        auto matches = optimizer_plain_writer(type, writer, key.obj, checked);
        if (PyErr_Occurred()) return nullptr;
        if (matches < 0) Py_RETURN_NONE;
        if (!matches) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

bool try_publish_fresh_gradients(PyObject* groups, PyObject* gradients,
    PyObject* filled, PyObject* types, PyObject* metadata_protocol,
    PyObject* state_protocol) {
    if (graph_capture_active() || !PyList_CheckExact(groups)
            || !PyDict_CheckExact(gradients) || Py_TYPE(filled) != &PySet_Type
            || PySet_Size(filled) != 0) return false;
    // Allocate the only tracked container before checking live metadata.
    // The remaining validation and slot writes invoke no Python callbacks or
    // GC-triggering allocations, and never release the GIL.
    PyObjHolder keys(PyList_New(0));
    if (!keys.obj) throw std::runtime_error("could not allocate gradient identities");
    if (!optimizer_adam_protocol(metadata_protocol, types)
            || !gradient_publication_protocol(types, state_protocol)) return false;
    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (PyDict_Next(gradients, &position, &key, &value))
        if (!PyLong_CheckExact(key)) return false;
    struct Row { PyObject *parameter, *gradient, *state, *slots, *key; Py_ssize_t index; };
    vector<Row> rows;
    std::unordered_set<PyObject*> seen;
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(groups); ++j) {
        auto* group = PyList_GET_ITEM(groups, j);
        if (!optimizer_string_keys(group)) return false;
        auto* params = PyDict_GetItemString(group, "params");
        auto* slots = PyDict_GetItemString(group, "grads");
        if (!params || !PyList_CheckExact(params) || !slots || !PyList_CheckExact(slots)
                || PyList_GET_SIZE(params) != PyList_GET_SIZE(slots)) return false;
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(params); ++i) {
            auto* p = PyList_GET_ITEM(params, i);
            if (!canonical_optimizer_holder(p, types) || !seen.insert(p).second
                    || PyList_GET_ITEM(slots, i) != Py_None
                    || !optimizer_type_field((PyObject*)Py_TYPE(p), "_torch_grad")) return false;
            auto* holder = GET_RAW_PTR(VarHolder, p);
            if (holder->view || !holder->get_requires_grad()) return false;
            PyObjHolder identity(PyLong_FromVoidPtr(p));
            if (!identity.obj || PyList_Append(keys.obj, identity.obj) < 0)
                throw std::runtime_error("could not allocate gradient identity");
            auto* g = PyDict_GetItemWithError(gradients, identity.obj);
            if (!g || !canonical_optimizer_holder(g, types)) return false;
            auto* gradient = GET_RAW_PTR(VarHolder, g);
            if (gradient->view || !gradient->var->is_stop_grad()
                    || holder->var->shape != gradient->var->shape
                    || holder->var->dtype() != gradient->var->dtype()) return false;
            auto** attrs = _PyObject_GetDictPtr(p);
            if (!attrs || !*attrs || !optimizer_string_keys(*attrs)) return false;
            auto* state = PyDict_GetItemString(*attrs, "_torch_object_state");
            if (!state || (PyObject*)Py_TYPE(state) != PyTuple_GET_ITEM(state_protocol, 0)) return false;
            PyObjHolder previous(PyObject_GetAttrString(state, "grad"));
            if (!previous.obj) {
                if (!PyErr_ExceptionMatches(PyExc_AttributeError))
                    throw std::runtime_error("could not inspect gradient slot");
                PyErr_Clear();
                return false;
            }
            if (previous.obj != Py_None) return false;
            rows.push_back({p, g, state, slots, identity.obj, i});
        }
    }
    for (const auto& row : rows) {
        // stop_grad also resets alias/first-order policy on an already stopped
        // gradient. Keep that operation rather than assuming it is a no-op.
        GET_RAW_PTR(VarHolder, row.gradient)->stop_grad();
        Py_INCREF(row.gradient);
        if (PyList_SetItem(row.slots, row.index, row.gradient) < 0
                || PyObject_SetAttrString(row.state, "grad", row.gradient) < 0
                || PySet_Add(filled, row.key) < 0)
            throw std::runtime_error("could not publish fresh gradient");
    }
    return true;
}


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
PyObject* concat_fallback_fn = nullptr;
PyObject* concat_selector_state = nullptr;
PyObject* getitem_selector_state = nullptr;
auto make_empty = op_constructor<VarPtr, NanoVector, NanoString>("empty");
auto make_setitem = op_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>("setitem");
auto make_getitem = op_constructor<VarPtr, Var*, VarSlices&&>("getitem");
auto make_cat_cast = op_constructor<VarPtr, Var*, NanoString>("unary");

struct ConcatAmpScope {
    int saved = amp_reg;
    ConcatAmpScope() { amp_reg |= amp_keep_reduce; }
    ~ConcatAmpScope() { amp_reg = saved; }
};

bool concat_selector_unchanged() {
    if (!concat_fallback_fn || !concat_selector_state) return false;
    PyObject* owner = PyTuple_GET_ITEM(concat_selector_state, 0);
    PyObject* dispatch = PyTuple_GET_ITEM(concat_selector_state, 2);
    return PyDict_GetItemString(owner, "select_kernel") == PyTuple_GET_ITEM(concat_selector_state, 1)
        && PyDict_GetItemString(dispatch, "dispatch_context") == PyTuple_GET_ITEM(concat_selector_state, 3)
        && PyDict_GetItemString(dispatch, "_NATIVE_SELECT") == PyTuple_GET_ITEM(concat_selector_state, 4);
}

bool basic_item(PyObject* item) {
    return PyLong_CheckExact(item) || PySlice_Check(item) || item == Py_None
        || item == Py_Ellipsis;
}

bool getitem_selector_unchanged() {
    if (!getitem_selector_state) return false;
    PyObject* owner = PyTuple_GET_ITEM(getitem_selector_state, 0);
    PyObject* dispatch = PyTuple_GET_ITEM(getitem_selector_state, 2);
    return PyDict_GetItemString(owner, "try_dispatch") == PyTuple_GET_ITEM(getitem_selector_state, 1)
        && PyDict_GetItemString(dispatch, "select_kernel") == PyTuple_GET_ITEM(getitem_selector_state, 3)
        && PyDict_GetItemString(dispatch, "dispatch_context") == PyTuple_GET_ITEM(getitem_selector_state, 4)
        && PyDict_GetItemString(dispatch, "_NATIVE_SELECT") == PyTuple_GET_ITEM(getitem_selector_state, 5);
}

// 1 for the new provider path's static subset, 0 for the Python path, -1 for
// a Python error. Exact ints exclude bool and arbitrary __index__ callbacks.
int positive_basic_item(PyObject* item) {
    if (PyLong_CheckExact(item) || item == Py_None || item == Py_Ellipsis) return 1;
    if (!PySlice_Check(item)) return 0;
    auto* slice = reinterpret_cast<PySliceObject*>(item);
    for (PyObject* bound : {slice->start, slice->stop, slice->step})
        if (bound != Py_None && !PyLong_CheckExact(bound)) return 0;
    if (slice->step == Py_None) return 1;
    int overflow = 0;
    long long step = PyLong_AsLongLongAndOverflow(slice->step, &overflow);
    if (PyErr_Occurred()) return -1;
    return overflow > 0 || (overflow == 0 && step > 0);
}

PyObject* not_implemented() {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}
} // namespace

void compat_fast_bind(PyObject* tracing, PyObject* concat_fallback, PyObject* selector_state,
                      PyObject* getitem_state) {
    if (concat_fallback == Py_None) concat_fallback = nullptr;
    if (selector_state == Py_None) selector_state = nullptr;
    if (getitem_state == Py_None) getitem_state = nullptr;
    if (concat_fallback && !PyCallable_Check(concat_fallback))
        throw std::invalid_argument("concat continuation must be callable");
    if (selector_state && (!PyTuple_CheckExact(selector_state)
            || PyTuple_GET_SIZE(selector_state) != 5
            || !PyDict_Check(PyTuple_GET_ITEM(selector_state, 0))
            || !PyDict_Check(PyTuple_GET_ITEM(selector_state, 2))))
        throw std::invalid_argument("invalid concat selector state");
    if (getitem_state && (!PyTuple_CheckExact(getitem_state)
            || PyTuple_GET_SIZE(getitem_state) != 6
            || !PyDict_Check(PyTuple_GET_ITEM(getitem_state, 0))
            || !PyDict_Check(PyTuple_GET_ITEM(getitem_state, 2))))
        throw std::invalid_argument("invalid getitem selector state");
    Py_XINCREF(tracing);
    Py_XINCREF(concat_fallback);
    Py_XINCREF(selector_state);
    Py_XINCREF(getitem_state);
    Py_XDECREF(tracing_fn);
    Py_XDECREF(concat_fallback_fn);
    Py_XDECREF(concat_selector_state);
    Py_XDECREF(getitem_selector_state);
    tracing_fn = tracing;
    concat_fallback_fn = concat_fallback;
    concat_selector_state = selector_state;
    getitem_selector_state = getitem_state;
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
    const bool registered = kernel_op_registered("tensor.concat");
    if ((registered || concat_selector_state) && !concat_selector_unchanged())
        return not_implemented();
    ConcatAmpScope amp_scope;
    // Even uniform inputs may need promotion under AMP. The Python path owns
    // those casts, including the cast back to torch.cat's common input dtype.
    if (binary_dtype_infer(ns_add, dtype, dtype) != dtype) return not_implemented();
    PyTensorFrontendScope scope(nullptr, items, n, false);
    if (registered) {
        // torch.cat normalizes to a list before selection; concat then passes
        // a tuple of the promoted inputs to its provider. Keep that observable
        // distinction for custom supports predicates and implementations.
        PyObjHolder normalized(PySequence_List(tensors));
        PyObjHolder axis(PyLong_FromLongLong(dim));
        PyObjHolder select_args(PyTuple_Pack(2, normalized.obj, axis.obj));
        PyObjHolder name(PyUnicode_FromString("tensor.concat"));
        // A provider can rebind the fast paths while it runs. Keep this call's
        // continuation alive independently of the binding's next value.
        Py_INCREF(concat_fallback_fn);
        PyObjHolder continuation(concat_fallback_fn);
        PyObject* selected = kernel_select(name.obj, select_args.obj, nullptr);
        if (!selected) return nullptr;
        PyObjHolder kernel(selected);
        if (kernel.obj != Py_None) {
            PyObjHolder provider_inputs(PySequence_Tuple(normalized.obj));
            PyObjHolder provider_args(PyTuple_Pack(2, provider_inputs.obj, axis.obj));
            PyObject* result = PyObject_CallObject(kernel.obj, provider_args.obj);
            if (!result) return nullptr;
            PyObjHolder output(result);
            if (result != Py_None) {
                if (!PyObject_TypeCheck(result, &PyjtVarHolder.ht_type)) {
                    PyErr_SetString(PyExc_TypeError, "concat provider must return a Var or None");
                    return nullptr;
                }
                Var* value = GET_RAW_PTR(VarHolder, result)->var;
                if (value->dtype() != dtype)
                    return to_py_object<VarHolder*>(new VarHolder(make_cat_cast(value, dtype)));
                return output.release();
            }
        }
        PyObjHolder dtype_name(PyUnicode_FromString(dtype.to_cstring()));
        PyObjHolder fallback_args(PyTuple_Pack(3, normalized.obj, axis.obj, dtype_name.obj));
        return PyObject_CallObject(continuation.obj, fallback_args.obj);
    }
    PyObject* capture = PyObject_CallObject(tracing_fn, nullptr);
    if (!capture) return nullptr;
    PyObjHolder capturing(capture);
    int active = PyObject_IsTrue(capture);
    if (active < 0) return nullptr;
    if (active) return not_implemented();

    // The frontend and placement the generated bindings of `empty` and
    // `setitem` would each have entered.
    NanoVector out_shape;
    for (int d = 0; d < ndim; d++) out_shape.push_back(d == dim ? total : shape[d]);
    VarPtr out = make_empty(out_shape, dtype);
    // The scratch allocation is not a differentiable input to concatenation.
    // Only the tensors copied into it may contribute gradient edges.
    out->set_stop_grad();
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
    const bool registered = kernel_op_registered("tensor.getitem");
    if ((registered || getitem_selector_state) && !getitem_selector_unchanged())
        return not_implemented();
    if (registered) {
        if (!no_grad) return not_implemented();
        if (PyTuple_CheckExact(index)) {
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(index); ++i) {
                int accepted = positive_basic_item(PyTuple_GET_ITEM(index, i));
                if (accepted < 0) return nullptr;
                if (!accepted) return not_implemented();
            }
        } else {
            int accepted = positive_basic_item(index);
            if (accepted < 0) return nullptr;
            if (!accepted) return not_implemented();
        }
    }
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
    const bool cascade_integer = PyLong_CheckExact(index)
        || (PyTuple_CheckExact(index) && PyTuple_GET_SIZE(index) == 1
            && PyLong_CheckExact(PyTuple_GET_ITEM(index, 0)));
    if (registered && !cascade_integer) {
        // _dispatch_slices creates a fresh tuple for tuple indices. The items
        // need no conversion in this subset, but supports sees that same form.
        PyObjHolder normalized;
        if (PyTuple_CheckExact(index)) {
            normalized.assign(PyTuple_New(PyTuple_GET_SIZE(index)));
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(index); ++i) {
                PyObject* item = PyTuple_GET_ITEM(index, i);
                Py_INCREF(item);
                PyTuple_SET_ITEM(normalized.obj, i, item);
            }
        } else {
            Py_INCREF(index);
            normalized.assign(index);
        }
        PyObjHolder name(PyUnicode_FromString("tensor.getitem"));
        PyObjHolder call_args(PyTuple_Pack(3, self, normalized.obj, Py_None));
        PyObject* selected = kernel_select(name.obj, call_args.obj, nullptr);
        if (!selected) return nullptr;
        PyObjHolder kernel(selected);
        if (selected != Py_None) {
            PyObject* result = PyObject_CallObject(selected, call_args.obj);
            if (!result) return nullptr;
            PyObjHolder output(result);
            if (result != Py_None) {
                if (PyObject_TypeCheck(result, &PyjtVarHolder.ht_type)) {
                    auto* out = GET_RAW_PTR(VarHolder, result);
                    if (!out->is_view()) {
                        VarSlices recorded = from_py_object<VarSlices>(index, holders);
                        out->set_view_of(holder, move(recorded));
                    }
                }
                return output.release();
            }
        }
        // The old Python continuation reaches x.getitem(), which can retry
        // the same provider. This subset resumes at the native op instead,
        // giving a declining custom provider exactly one call.
        var = holder->var;
    }
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
                // Torch casts the assigned scalar to the destination dtype;
                // `ternary` would instead promote the whole tensor (int32 x
                // with 0.25 became float32). The Python path performs that
                // cast, so decline here and let it own the assignment.
                if (number->var->dtype() != x->dtype()) return not_implemented();
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
// The (dims_mask, keepdims_mask) constructor: unlike `reduce(x, op, dims,
// keepdims)` it keeps a bool input's dtype for the logical reductions.
auto make_reduce_mask = op_constructor<VarPtr, Var*, NanoString, uint, uint>("reduce");
} // namespace

PyObject* logical_reduce(PyObject* self, PyObject* dims_obj, bool keepdims, bool is_and) {
    // Frontend tensors share the native payload. Infer their result type in
    // the same scope as the general reduction binding below.
    if (!PyObject_TypeCheck(self, &PyjtVarHolder.ht_type)) return not_implemented();
    // Read as the generated `any_` binding would read it.
    Var* x = GET_RAW_PTR(VarHolder, self)->grad_input();
    if (x->dtype() != ns_bool) return not_implemented();
    // The placement and precision the generated binding would build under.
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    // ROCm has no narrow atomics (cuda_atomic.h), and the other backends
    // have not been asked to reduce into a bool.
    BackendId backend = construction_target_backend(x);
    if (backend != BackendId::Cpu && backend != BackendId::Cuda) return not_implemented();
    // `any_` takes one int or a sequence of them; anything else is left to its
    // overload resolution, and an out-of-range dim to its error.
    NanoVector dims;
    if (PyLong_CheckExact(dims_obj)) {
        int overflow = 0;
        long long dim = PyLong_AsLongLongAndOverflow(dims_obj, &overflow);
        if (overflow || (dim == -1 && PyErr_Occurred())) {
            PyErr_Clear();
            return not_implemented();
        }
        dims.push_back(dim);
    } else if (Py_TYPE(dims_obj) == &PyjtNanoVector) {
        dims = from_py_object<NanoVector>(dims_obj);
    } else if (PyTuple_CheckExact(dims_obj) || PyList_CheckExact(dims_obj)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(dims_obj);
        PyObject** items = PySequence_Fast_ITEMS(dims_obj);
        for (Py_ssize_t i = 0; i < n; i++)
            if (!PyLong_CheckExact(items[i])) return not_implemented();
        dims = from_py_object<NanoVector>(dims_obj);
    } else {
        return not_implemented();
    }
    // The mask `ReduceOp(x, op, dims, keepdims)` derives.
    int64 xdim = x->shape.size();
    uint reduce_mask = 0;
    if (!dims.size()) {
        reduce_mask = xdim >= 32 ? ~0u : (1u << xdim) - 1;
    } else {
        for (auto dim : dims) {
            if (dim < 0) dim += xdim;
            if (dim < 0 || dim >= xdim) return not_implemented();
            reduce_mask |= 1u << dim;
        }
    }
    VarPtr y = make_reduce_mask(x, is_and ? ns_logical_and : ns_logical_or,
                                reduce_mask, keepdims ? ~0u : 0u);
    return to_py_object<VarHolder*>(new VarHolder(move(y)));
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
        // float32 divides in float32: that is what torch does with a Python
        // float, measured -- `x / 0.28209479177387814` over 20000 float32 `x`
        // matches `x / float32(scale)` bit for bit, and differs from a float64
        // route on 12615 of them. Only half precision widens, to its float32
        // opmath. The float64 route this replaces was also the largest single
        // excess of the elementwise class on a 1/64-rate double part.
        const bool widen = own != ns_float32;
        double value = PyFloat_AS_DOUBLE(other);
        float narrow = (float)value;
        Var* x = GET_RAW_PTR(VarHolder, self)->var;
        PyObjHolder a;
        PyObject* left = self;
        if (widen) {
            a.assign(wrap_like(self, new VarHolder(make_unary(x, ns_float32))));
            left = a.obj;
        }
        PyObjHolder b(wrap_like(self, new VarHolder(make_array(
            (const void*)&narrow, {}, ns_float32))));
        // This is a Python literal, not a trainable tensor operand. Native
        // floating ArrayOp defaults must not create a new autograd leaf.
        GET_RAW_PTR(VarHolder, b.obj)->stop_grad();
        out = binary_slots[B_TRUEDIV](left, b.obj);
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
        // A float divisor of a floating tensor, built as `_true_division`
        // builds it; anything else -- an int, a reflected division -- stays
        // there.
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

bool try_ordered_unaliased_inplace(PyObject* self, PyObject* value) {
    if (!no_grad || graph_capture_active() || !runtime_use_cuda()
            || accelerator_backend_id() != BackendId::Cuda
            || !is_var(self) || !is_var(value)) return false;
    PyTensorFrontendScope frontend(self, nullptr, 0, false);
    // Stay under the generated binding's graph lock, as assign does. This
    // builds new dependencies; it neither enters nor drains the executor.
    auto* target = GET_RAW_PTR(VarHolder, self);
    Var* a = target->var;
    Var* b = GET_RAW_PTR(VarHolder, value)->var;
    NanoString dtype = a->dtype();
    // WriteBack orders and copies storage without floating-point arithmetic.
    // Integer updates need the same alias/version guards, not a global drain.
    if (dtype != ns_float32 && dtype != ns_float16 && dtype != ns_bfloat16
            && dtype != ns_int32 && dtype != ns_int64) return false;
    if (a == b || !a->mem_ptr || !a->is_finished() || !a->allocator
            || !a->allocator->is_cuda() || target->view || target->views
            || a->share_src || a->liveness.forward.count() != 1
            || a->capture_owners || a->flag(VarFlags::_kept) || b->flag(VarFlags::_kept)
            || !a->is_stop_grad() || !b->is_stop_grad()
            || !a->is_contiguous() || !b->is_contiguous()
            || a->shape != b->shape || a->dtype() != b->dtype()
            || a->placement != b->placement || a->device_id != b->device_id)
        return false;
    if (a->placement.explicit_backend && a->placement.device.backend != BackendId::Cuda)
        return false;
    // Foreign/DLPack allocations may still have consumers in another runtime.
    // Admit only storage from this device's current ordinary allocator stack.
    if (a->allocator != get_allocator(Device{BackendId::Cuda, a->device_id}, false))
        return false;
    Op* update = b->input();
    if (!update || !update->is_op(op_ids::binary())) return false;
    bool consumes_target = false;
    for (Var* input : update->inputs())
        consumes_target = consumes_target || input == a;
    if (!consumes_target) return false;
    // An unexecuted native view has no physical share ring yet. Future reads
    // through that alias would not depend on the replacement holder.
    for (Op* reader : a->outputs())
        for (Var* output : reader->outputs())
            if (output->share_src == a) return false;
    auto version = a->peek_storage_version();
    if (version && (version.use_count() != 2
            || version->externally_visible.load(std::memory_order_relaxed))) return false;
    if (a->autograd_alias && (a->autograd_alias.use_count() != 1
            || a->autograd_alias->version.use_count() != 1)) return false;
    {
        // The async executor links allocated aliases under this mutex.
        // Pending aliases were excluded above before their allocation.
        std::lock_guard<std::mutex> guard(share_ring_mutex());
        if (a->share_next) return false;
    }
    static auto make_write_back = op_constructor<vector<VarPtr>, vector<Var*>&&, vector<Var*>&&>("write_back");
    auto written = make_write_back(vector<Var*>{a}, vector<Var*>{b});
    VarHolder result(move(written[0]));
    // WriteBackOp orders earlier readers and shares the destination's storage
    // and mutation token. assign advances that token exactly once.
    target->assign(&result);
    return true;
}

bool try_defer_unaliased_inplace(PyObject* self, PyObject* value) {
    if (!no_grad || graph_capture_active() || !runtime_use_cuda()
            || accelerator_backend_id() != BackendId::Cuda
            || !is_var(self) || !is_var(value)) return false;
    PyTensorFrontendScope frontend(self, nullptr, 0, false);
    // The generated binding holds GraphEntryScope, as assign does. This
    // only constructs/rebinds nodes; entering the executor here would release
    // that graph lock and allow queued work to allocate the pending result.
    auto* target = GET_RAW_PTR(VarHolder, self);
    auto* replacement = GET_RAW_PTR(VarHolder, value);
    Var* a = target->var;
    Var* b = replacement->var;
    NanoString dtype = a->dtype();
    if (dtype != ns_float32 && dtype != ns_float16 && dtype != ns_bfloat16) return false;
    if (a == b || a->mem_ptr || a->is_finished() || target->view || target->views
            || a->share_src || a->share_next || a->liveness.forward.count() != 1
            || !a->is_stop_grad() || !b->is_stop_grad() || !a->is_contiguous()
            || a->shape != b->shape || a->dtype() != b->dtype()
            || a->placement != b->placement || a->device_id != b->device_id)
        return false;
    if (a->placement.explicit_backend && a->placement.device.backend != BackendId::Cuda)
        return false;
    // These providers allocate a fresh result, rather than a view or an
    // output aliased with an input. Broader producers retain the store path.
    Op* producer = a->input();
    if (!producer || !((has_op("cublas_matmul")
                && producer->is_op(get_op_id("cublas_matmul")))
            || (has_op("cublas_batched_matmul")
                && producer->is_op(get_op_id("cublas_batched_matmul"))))) return false;
    Op* update = b->input();
    if (!update || !update->is_op(op_ids::binary())) return false;
    int readers = 0;
    for (auto* consumer : a->outputs())
        if (consumer != update || ++readers > 1) return false;
    if (readers != 1) return false;
    auto version = a->peek_storage_version();
    if (version && version.use_count() != 2) return false;
    if (a->autograd_alias && (a->autograd_alias.use_count() != 1
            || a->autograd_alias->version.use_count() != 1)) return false;
    // Copy also covers copy_(rhs) when the caller still owns rhs: assigning
    // rhs itself would make the two tensors share mutable storage.
    static auto make_copy = op_constructor<VarPtr, Var*>("copy");
    VarHolder copied(make_copy(b));
    if (!version) version = a->storage_version_token();
    target->assign(&copied);
    std::atomic_store(&target->var->storage_mutation_version, version);
    return true;
}

PyObject* fast_contiguous(PyObject* self, PyObject* tensor_type) {
    if (!is_var(self) || !PyType_Check(tensor_type)) return none();
    auto* holder = GET_RAW_PTR(VarHolder, self);
    TensorPlacement placement = holder->var->placement;
    if (!placement.explicit_backend || (placement.device.backend != BackendId::Cpu
            && placement.device.backend != BackendId::Cuda)) return none();
    // As tensor_frontend(..., like=self): the owner selects the result type
    // and AD policy, while the input outranks any ambient construction device.
    PyTensorFrontendScope frontend(reinterpret_cast<PyTypeObject*>(tensor_type));
    TensorPlacementScope input_placement(placement);
    return to_py_object<VarHolder*>(new VarHolder(contiguous_storage(holder->grad_input())));
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

PyObject* fast_storage_transpose(PyObject* self, PyObject* dims) {
    if (!is_var(self) || !PyTuple_Check(dims) || PyTuple_GET_SIZE(dims) != 2)
        return none();
    int64 first, second;
    if (!exact_int(PyTuple_GET_ITEM(dims, 0), first)
            || !exact_int(PyTuple_GET_ITEM(dims, 1), second)) {
        PyErr_Clear();
        return none();
    }
    auto* holder = GET_RAW_PTR(VarHolder, self);
    int ndim = holder->var->shape.size();
    if (!ndim || first < -ndim || first >= ndim || second < -ndim || second >= ndim)
        return none();
    if (first < 0) first += ndim;
    if (second < 0) second += ndim;
    NanoVector axes;
    for (int i = 0; i < ndim; ++i)
        axes.push_back(i == first ? second : i == second ? first : i);
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    unique_ptr<VarHolder> out(holder->storage_permute(axes));
    out->set_transpose_view_of(holder, axes);
    return to_py_object<VarHolder*>(out.release());
}

namespace {
// `jittor.transpose(self, axes)` for a valid permutation.
PyObject* transpose_by(PyObject* self, VarHolder* holder, NanoVector axes) {
    int ndim = holder->var->shape.size();
    if (kernel_op_registered("tensor.transpose")) return none();
    PyTensorFrontendScope scope(self, nullptr, 0, false);
    // A transpose of a transpose is one transpose of the source, and none at
    // all when the two cancel; see `jittor.transpose`. Not when the
    // intermediate tracks a gradient: the result would no longer depend on
    // it, and `jt.grad(loss, intermediate)` came back zero.
    unique_ptr<VarHolder> source;
    NanoVector prior;
    if (no_grad || holder->var->is_stop_grad()) {
        prior = holder->transpose_view_axes();
        if (prior.size()) {
            source.reset(holder->transpose_view_source());
        } else {
            prior = holder->producer_transpose_axes();
            if (prior.size()) source.reset(new VarHolder(holder->var->input()->inputs().front()));
        }
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
