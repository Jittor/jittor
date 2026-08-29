// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guoye Yang <498731903@qq.com>
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <Python.h>
#include <frameobject.h>
#include "pyjt/py_obj_holder.h"
#include "pyjt/py_converter.h"
#include "pybind/py_var_tracer.h"
#include "utils/str_utils.h"
#include "op.h"
#include "var.h"
#include "fused_op.h"

namespace jittor {

DEFINE_FLAG(int, trace_py_var, 0, "Trace py stack max depth for debug.");
DEFINE_FLAG(int, trace_var_data, 0, "Trace py stack max depth for debug.");
Op* trace_grad_op = nullptr;

TraceData trace_data;
int64 cnt = 0;

static PyObject* my_import(const char* module_name, const char* attr) {
    // LOGir << module_name << attr;
    PyObjHolder a(PyImport_ImportModule(module_name));
    PyObjHolder b(PyObject_GetAttrString(a.obj, attr));
    // LOGir << "Done";
    return b.obj;
}

// CPython 3.11 made PyFrameObject and PyCodeObject opaque: f_code, f_back and
// f_localsplus are no longer struct fields, and co_varnames / co_cellvars /
// co_freevars moved behind accessors that only exist from 3.11 on. The helpers
// below read the same information through the attribute protocol, which has
// behaved identically on every supported version, so this tracer works on 3.7
// and newer releases from one code path.

// A reference holder that tolerates nullptr. PyObjHolder treats a null pointer
// as a fatal Python error, but walking a stack legitimately reaches the bottom,
// and a frame legitimately may not expose a locals mapping.
struct PyRef {
    PyObject* obj;
    PyRef() : obj(nullptr) {}
    explicit PyRef(PyObject* o) : obj(o) {}
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
    inline void reset(PyObject* o = nullptr) {
        PyObject* old = obj;
        obj = o;
        Py_XDECREF(old);
    }
    inline ~PyRef() { Py_XDECREF(obj); }
};

// New reference to the frame's code object, or nullptr.
static PyObject* frame_code(PyFrameObject* f) {
    if (!f) return nullptr;
    #if PY_VERSION_HEX >= 0x03090000
    return (PyObject*)PyFrame_GetCode(f);
    #else
    PyObject* co = (PyObject*)f->f_code;
    Py_XINCREF(co);
    return co;
    #endif
}

// New reference to the calling frame, or nullptr at the bottom of the stack.
static PyFrameObject* frame_back(PyFrameObject* f) {
    if (!f) return nullptr;
    #if PY_VERSION_HEX >= 0x03090000
    return PyFrame_GetBack(f);
    #else
    PyFrameObject* back = f->f_back;
    Py_XINCREF(back);
    return back;
    #endif
}

// New reference to an attribute of the frame's code object, or nullptr.
static PyObject* code_attr(PyFrameObject* f, const char* name) {
    PyRef co(frame_code(f));
    if (!co.obj) return nullptr;
    PyObject* value = PyObject_GetAttrString(co.obj, name);
    if (!value) PyErr_Clear();
    return value;
}

// New reference to the frame's local variables as a mapping. Reading the
// ``f_locals`` attribute performs the fast-locals snapshot on every version,
// including the ones where PyFrame_FastToLocalsWithError has been removed.
static PyObject* frame_locals(PyFrameObject* f) {
    if (!f) return nullptr;
    PyObject* locals = PyObject_GetAttrString((PyObject*)f, "f_locals");
    if (!locals) PyErr_Clear();
    return locals;
}

// Python 3.13 promoted optional attribute lookup to the public C API and no
// longer declares the private spelling used by older releases.
static int object_lookup_optional_attr(PyObject* obj, PyObject* name, PyObject** result) {
    #if PY_VERSION_HEX >= 0x030d0000
    return PyObject_GetOptionalAttr(obj, name, result);
    #else
    return _PyObject_LookupAttr(obj, name, result);
    #endif
}

// The value bound to the frame's first parameter -- ``self`` for a method --
// as a borrowed reference into ``locals``, or nullptr.
static PyObject* frame_first_argument(PyFrameObject* f, PyObject* locals) {
    if (!locals || !PyDict_Check(locals)) return nullptr;
    PyRef names(code_attr(f, "co_varnames"));
    if (!names.obj || !PyTuple_Check(names.obj) || PyTuple_GET_SIZE(names.obj) == 0)
        return nullptr;
    PyObject* first = PyTuple_GET_ITEM(names.obj, 0);
    PyObject* value = PyDict_GetItem(locals, first);   // borrowed
    if (!value) PyErr_Clear();
    return value;
}

static PyObject* find_obj_name(PyFrameObject* f, PyObject* obj, const char* default_name="_model") {
    PyRef locals(frame_locals(f));
    if (locals.obj && PyDict_Check(locals.obj)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(locals.obj, &pos, &key, &value)) {
            if (value == obj) {
                Py_INCREF(key);
                return key;
            }
        }
    }
    return PyUnicode_FromString(default_name);
}

static string to_string(PyObject* obj) {
    Py_ssize_t size;
    const char* s = PyUnicode_AsUTF8AndSize(obj, &size);
    return string(s, size);
}

static string code_string(PyFrameObject* f, const char* name) {
    PyRef value(code_attr(f, name));
    if (!value.obj || !PyUnicode_Check(value.obj)) return string();
    return to_string(value.obj);
}

static vector<Stack> get_stack_info() {
    // cnt ++;
    // if (cnt % 100 != 0) return {};
    vector<Stack> stacks;
    static auto getframe = my_import("sys", "_getframe");
    static auto jt_module = my_import("jittor", "Module");
    static auto jt_optimizer = my_import("jittor.optim", "Optimizer");
    static auto fill_module_name = my_import("jittor.utils.tracer", "fill_module_name");
    static auto _trace_name = PyUnicode_FromString("_trace_name");

    PyObjHolder ret(PyObject_CallFunctionObjArgs(getframe, nullptr));

    // ``PyFrame_GetBack`` hands out a new reference on every version that has
    // it, so the walk owns each frame and releases them together at the end.
    vector<PyFrameObject*> owned;
    for (PyFrameObject* frame = frame_back((PyFrameObject*)ret.obj);
         frame != nullptr;
         frame = frame_back(frame))
        owned.push_back(frame);
    int n = (int)owned.size() + 1;
    STACK_ALLOC(PyFrameObject*, frames, n);
    frames[n-1] = (PyFrameObject*)ret.obj;
    for (int k = 0; k < (int)owned.size(); k++)
        frames[n-2-k] = owned[k];
    struct FrameRelease {
        vector<PyFrameObject*>& owned;
        ~FrameRelease() { for (auto* f : owned) Py_XDECREF(f); }
    } release{owned};
    PyRef prev_obj_ref;
    PyObject* prev_obj = nullptr;
    if (trace_py_var >= 3) {
        // trace raw stack
        // auto start = std::max(0, n-5);
        auto start = 0;
        for (int i=start; i<n; i++) {
            auto f = frames[i];
            auto filename = code_string(f, "co_filename");
            auto lineno = (int)PyFrame_GetLineNumber(f);
            stacks.emplace_back(Stack{
                filename+":"+S(lineno), 
                code_string(f, "co_name"),
                filename,
                lineno});
        }
        return stacks;
    }
    for (int i=0; i<n; i++) {
        auto f = frames[i];
        {
            PyRef locals(frame_locals(f));
            auto borrowed = frame_first_argument(f, locals.obj);
            if (borrowed == prev_obj) continue;
            if (borrowed == nullptr) {
                // a plain function has no bound first argument
                prev_obj_ref.reset();
                prev_obj = nullptr;
                continue;
            }
            // ``borrowed`` points into ``locals``; own it so it stays valid
            // here and so the next iteration's identity check cannot compare
            // against a freed address.
            Py_INCREF(borrowed);
            prev_obj_ref.reset(borrowed);
            auto obj = borrowed;
            prev_obj = obj;
            auto tp_mro = obj->ob_type->tp_mro;
            auto base_type = PyTuple_GET_ITEM(tp_mro, Py_SIZE(tp_mro)-2);
            auto prev_f = i? frames[i-1] : f;
            if (base_type == jt_optimizer) {
                string init_name = string(obj->ob_type->tp_name) + "_init";
                PyRef caller((PyObject*)frame_back(f));
                PyObjHolder ret(find_obj_name((PyFrameObject*)caller.obj, obj,
                                              init_name.c_str()));
                stacks.emplace_back(Stack{
                    to_string(ret.obj), 
                    string(obj->ob_type->tp_name),
                    code_string(prev_f, "co_filename"),
                    (int)PyFrame_GetLineNumber(prev_f)});
                break;
            }
            if (base_type != jt_module)
                continue;
            PyObjHolder ret;
            auto lookup_result = object_lookup_optional_attr(obj, _trace_name, &ret.obj);
            if (lookup_result < 0)
                LOGf << "Failed to look up Module._trace_name";
            string scope_name;
            if (!ret.obj) {
                // find base name
                auto co_name = code_string(f, "co_name");
                if (co_name == "__init__") {
                    scope_name = string(obj->ob_type->tp_name) + "_init";
                } else
                if (co_name == "__call__") {
                    if (i) {
                        PyRef caller((PyObject*)frame_back(f));
                        ret.assign(find_obj_name((PyFrameObject*)caller.obj, obj));
                        scope_name = to_string(ret.obj);
                    } else {
                        ret.assign(PyUnicode_FromString("_model"));
                        scope_name = "_model";
                    }
                    PyObjHolder _(PyObject_CallFunctionObjArgs(
                        fill_module_name, obj, ret.obj, nullptr));
                }
            } else {
                scope_name = to_string(ret.obj);
            }
            stacks.emplace_back(Stack{
                move(scope_name), 
                string(obj->ob_type->tp_name),
                code_string(prev_f, "co_filename"),
                (int)PyFrame_GetLineNumber(prev_f)});
        }
    }
    if (stacks.size() == 0) {
        auto m = std::min(3,n);
        for (int i=0; i<m; i++) {
            auto f = frames[n-m+i];
            auto s = code_string(f, "co_filename");
            auto num = (int)PyFrame_GetLineNumber(f);
            stacks.emplace_back(Stack{
                s+":"+S(num), 
                "",
                s,
                num});
        }
    }
    return stacks;
}

template<class T>
string get_str(T* t, int64 num) {
    string s = "";
    for (int64 i=0; i<num; i++) {
        s += S(t[i]);
        if (i != num-1)
            s += ',';
    }
    return s;
} 

static inline string get_var_data_str(Var* v) {
    if (v->dtype() == ns_int8)
        return get_str(v->ptr<int8>(), v->num);
    if (v->dtype() == ns_int16)
        return get_str(v->ptr<int16>(), v->num);
    if (v->dtype() == ns_int32)
        return get_str(v->ptr<int32>(), v->num);
    if (v->dtype() == ns_int64)
        return get_str(v->ptr<int64>(), v->num);


    if (v->dtype() == ns_uint8)
        return get_str(v->ptr<uint8>(), v->num);
    if (v->dtype() == ns_uint16)
        return get_str(v->ptr<uint16>(), v->num);
    if (v->dtype() == ns_uint32)
        return get_str(v->ptr<uint32>(), v->num);
    if (v->dtype() == ns_uint64)
        return get_str(v->ptr<uint64>(), v->num);

    if (v->dtype() == ns_float32)
        return get_str(v->ptr<float32>(), v->num);
    if (v->dtype() == ns_float64)
        return get_str(v->ptr<float64>(), v->num);
    return "";
}

void TraceData::record_node(Node* node, bool record_stack) {
    if (get_thread_name().size()) return;
    NodeData data;
    data.id = node_data_cnt++;
    id_map[node] = data.id;
    if (trace_py_var) {
        if (record_stack) {
            if (trace_grad_op) {
                auto iter = trace_data.id_map.find(trace_grad_op);
                data.stacks.emplace_back(Stack{"grad", "Grad", "", 0});
                if (iter != trace_data.id_map.end()) {
                    data.attrs["grad_op_id"] = S(iter->second);
                    auto& prev_stack = trace_data.node_data[iter->second].stacks;
                    for (auto& s : prev_stack)
                        data.stacks.push_back(s);
                }
            } else
                data.stacks = get_stack_info();
        }
    } else {
    }
    data.attrs["__id"] = S(node->id);
    data.attrs["is_var"] = node->is_var() ? "1" : "0";
    data.attrs["name"] = "unname";
    node_data[data.id] = move(data);
}

static int64 get_node_id(Node* node) {
    auto iter = trace_data.id_map.find(node);
    if (iter != trace_data.id_map.end())
        return iter->second;
    trace_data.record_node(node, false);
    return trace_data.node_data_cnt - 1;
}

void TraceData::release_node(Node* node) {
    if (get_thread_name().size()) return;
    auto iter = trace_data.id_map.find(node);
    if (iter == trace_data.id_map.end())
        return;
    auto node_id = iter->second;
    id_map.erase(node);
    if (trace_py_var < 2 || execute_op_info.size() > 100000) {
        node_data.erase(node_id);
    }
}

void TraceData::record_exe_node(Node* node) {
    auto node_id = get_node_id(node);
    auto& data = node_data[node_id];
    auto name_iter = data.attrs.find("name");
    if (data.inputs.size() != node->inputs().size() || data.attrs.size() == 0 || name_iter == data.attrs.end() || name_iter->second == "unname") {
        data.inputs.clear();
        data.inputs.reserve(node->inputs().size());
        for (auto i : node->inputs()) {
            auto iid = get_node_id(i);
            data.inputs.push_back(iid);
            node_data[iid].outputs.push_back(node_id);
        }
        if (node->is_var()) {
            auto v = node->var();
            std::stringstream ss;
            ss << v->shape;
            data.attrs["shape"] = ss.str();
            data.attrs["ndim"] = S(v->shape.size());
            data.attrs["dtype"] = v->dtype().to_cstring();
            data.attrs["dsize"] = S(v->dtype().dsize());
            data.attrs["name"] = v->name.c_str();
            data.attrs["is_var"] = "1";
            if (trace_var_data && v->mem_ptr)
                data.attrs["data"] = get_var_data_str(v);
        } else {
            auto op = node->op();
            data.attrs["name"] = op->name_ex();
            data.attrs["is_var"] = "0";
            // TODO: add other op attrs
        }
    }
}

void TraceData::record_op(Op* op) {
    record_exe_node(op);
    for (auto o : op->outputs())
        record_exe_node(o);
}

void TraceData::record_execution(Op* op, bool is_fused_op, JK& jk) {
    if (execute_op_info.size() > 100000) return;
    ExecuteOpInfo& einfo = execute_op_info[execute_op_info_cnt++];
    if (is_fused_op) {
        FusedOp* fop = (FusedOp*)op;
        for (auto op : fop->ops) {
            record_op(op);
            einfo.fused_ops.push_back(get_node_id(op));
        }
    } else {
        record_op(op);
        einfo.fused_ops.push_back(get_node_id(op));
    }
    op->do_prepare(jk);
    if (jk.empty()) return;
    const char* jit_key = jk.to_cstring();
    auto iter = jit_key_mapper.find(jit_key);
    if (iter == jit_key_mapper.end())
        einfo.jit_key = jit_key;
    else
        einfo.jit_key = iter->second;
    jit_key_map[einfo.jit_key].push_back(execute_op_info_cnt-1);
    einfo.file_path = Op::get_filename_from_jit_key(jk.to_cstring(), ".cc");
}

template<class T>
static void fill_dict(PyObject* dict, T key, PyObject* value) {
    PyObjHolder k(to_py_object<T>(key));
    PyObjHolder v(value);
    PyDict_SetItem(dict, k.obj, value);
}

// template<>
// PyObject* to_py_object(const Stack& stack) {
//     return nullptr;
// }

DEF_IS(Stack, PyObject*) to_py_object(const T& a) {
    PyObjHolder dict(PyDict_New());
    fill_dict(dict.obj, string("name"), to_py_object<string>(a.module_name));
    fill_dict(dict.obj, string("type"), to_py_object<string>(a.module_type));
    fill_dict(dict.obj, string("file_path"), to_py_object<string>(a.file_path));
    fill_dict(dict.obj, string("lineno"), to_py_object<int64>(a.lineno));
    return dict.release();
}

PyObject* dump_trace_data() {
    PyObjHolder dict(PyDict_New());
    PyObjHolder node_data(PyDict_New());
    PyObjHolder execute_op_info(PyDict_New());
    for (auto& kv : trace_data.node_data) {
        if (kv.second.attrs.size() == 0)
            continue;
        auto name_iter = kv.second.attrs.find("name");
        // if don't have name, this node is not executed
        if (name_iter == kv.second.attrs.end() || name_iter->second == "unname")
            continue;
        PyObjHolder dict(PyDict_New());
        fill_dict(dict.obj, string("id"), to_py_object(kv.second.id));
        fill_dict(dict.obj, string("inputs"), to_py_object(kv.second.inputs));
        fill_dict(dict.obj, string("outputs"), to_py_object(kv.second.outputs));
        fill_dict(dict.obj, string("stacks"), to_py_object(kv.second.stacks));
        fill_dict(dict.obj, string("attrs"), to_py_object(kv.second.attrs));
        fill_dict(node_data.obj, kv.first, dict.release());
    }
    for (auto& kv : trace_data.execute_op_info) {
        PyObjHolder dict(PyDict_New());
        fill_dict(dict.obj, string("fused_ops"), to_py_object(kv.second.fused_ops));
        fill_dict(dict.obj, string("jit_key"), to_py_object<string>(kv.second.jit_key));
        fill_dict(dict.obj, string("file_path"), to_py_object<string>(kv.second.file_path));
        fill_dict(dict.obj, string("attrs"), to_py_object(kv.second.attrs));
        fill_dict(execute_op_info.obj, kv.first, dict.release());
    }
    fill_dict(dict.obj, string("node_data"), node_data.release());
    fill_dict(dict.obj, string("execute_op_info"), execute_op_info.release());
    return dict.release();
}

void clear_trace_data() {
    trace_data.execute_op_info.clear();
    trace_data.jit_key_map.clear();
    trace_data.id_map.clear();
    trace_data.node_data.clear();
}

string _get_stack_info(Node* node, const char* change_line) {
    string stack_info = "";
    auto iter = trace_data.id_map.find(node);
    if (iter == trace_data.id_map.end())
        return stack_info;
    auto node_id = iter->second;
    auto iter2 = trace_data.node_data.find(node_id);
    if (iter2 == trace_data.node_data.end())
        return stack_info;
    for (auto& stack : iter2->second.stacks) {
        stack_info += stack.module_name;
        stack_info += '(';
        stack_info += stack.module_type;
        stack_info += ')';
        stack_info += " -> ";
        stack_info += change_line;
    }
    return stack_info;
}

void print_node_trace(const Node* node, std::ostream& os) {
    os << _get_stack_info((Node*)node, "\n");
}

vector<Stack> get_node_trace(Node* node) {
    auto iter = trace_data.id_map.find(node);
    if (iter == trace_data.id_map.end())
        return vector<Stack>();
    auto node_id = iter->second;
    auto iter2 = trace_data.node_data.find(node_id);
    if (iter2 == trace_data.node_data.end())
        return vector<Stack>();
    return iter2->second.stacks;
}


} // jittor
