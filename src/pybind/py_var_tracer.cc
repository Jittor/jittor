// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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
#include "misc/str_utils.h"
#include "op.h"
#include "var.h"
#include "fused_op.h"

namespace jittor {

DEFINE_FLAG(int, trace_py_var, 0, "Trace py stack max depth for debug.");
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

static PyObject* find_obj_name(PyFrameObject* f, PyObject* obj, const char* default_name="_model") {
    auto co = f->f_code;
    auto map = co->co_varnames;

    auto fast = f->f_localsplus;
    auto j = PyTuple_GET_SIZE(map);
    if (j > co->co_nlocals)
        j = co->co_nlocals;
    if (co->co_nlocals) {
        for (int i=0; i<j; i++) {
            if (fast[i] == obj) {
                auto s = PyTuple_GET_ITEM(map, i);
                Py_INCREF(s);
                return s;
            }
        }
    }
    auto ncells = PyTuple_GET_SIZE(co->co_cellvars);
    auto nfreevars = PyTuple_GET_SIZE(co->co_freevars);
    if (ncells || nfreevars) {
        for (int i=0; i<ncells; i++) {
            if (fast[i+co->co_nlocals] == obj) {
                auto s = PyTuple_GET_ITEM(co->co_cellvars, i);
                Py_INCREF(s);
                return s;
            }
        }
        for (int i=0; i<nfreevars; i++) {
            if (fast[i+co->co_nlocals+ncells] == obj) {
                auto s = PyTuple_GET_ITEM(co->co_freevars, i);
                Py_INCREF(s);
                return s;
            }
        }
    }
    // LOGw << "not found name" << map << co->co_cellvars << co->co_freevars << (PyObject*)f;
    return PyUnicode_FromString(default_name);
}

static string to_string(PyObject* obj) {
    Py_ssize_t size;
    const char* s = PyUnicode_AsUTF8AndSize(obj, &size);
    return string(s, size);
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

    auto frame = (PyFrameObject*)ret.obj;
    int n=0;
    while (frame) n++, frame = frame->f_back;
    PyFrameObject* frames[n];
    frame = (PyFrameObject*)ret.obj;
    int i=n;
    while (i) frames[--i] = frame, frame = frame->f_back;
    PyObject* prev_obj = nullptr;
    if (trace_py_var >= 3) {
        // trace raw stack
        // auto start = std::max(0, n-5);
        auto start = 0;
        for (int i=start; i<n; i++) {
            auto f = frames[i];
            auto filename = to_string(f->f_code->co_filename);
            auto lineno = (int)PyFrame_GetLineNumber(f);
            stacks.emplace_back(Stack{
                filename+":"+S(lineno), 
                to_string(f->f_code->co_name),
                filename,
                lineno});
        }
        return stacks;
    }
    for (int i=0; i<n; i++) {
        auto f = frames[i];
        if (Py_SIZE(f->f_code->co_varnames)) {
            auto fast = f->f_localsplus;
            auto obj = fast[0];
            if (obj == prev_obj) continue;
            prev_obj = obj;
            if (obj == nullptr)
                // normal function first argument is null
                continue;
            auto tp_mro = obj->ob_type->tp_mro;
            auto base_type = PyTuple_GET_ITEM(tp_mro, Py_SIZE(tp_mro)-2);
            auto prev_f = i? frames[i-1] : f;
            if (base_type == jt_optimizer) {
                string init_name = string(obj->ob_type->tp_name) + "_init";
                PyObjHolder ret(find_obj_name(f->f_back, obj, init_name.c_str()));
                stacks.emplace_back(Stack{
                    to_string(ret.obj), 
                    string(obj->ob_type->tp_name),
                    to_string(prev_f->f_code->co_filename),
                    (int)PyFrame_GetLineNumber(prev_f)});
                break;
            }
            if (base_type != jt_module)
                continue;
            PyObjHolder ret;
            _PyObject_LookupAttr(obj, _trace_name, &ret.obj);
            string scope_name;
            if (!ret.obj) {
                // find base name
                auto co_name = to_string(f->f_code->co_name);
                if (co_name == "__init__") {
                    scope_name = string(obj->ob_type->tp_name) + "_init";
                } else
                if (co_name == "__call__") {
                    if (i) {
                        ret.assign(find_obj_name(f->f_back, obj));
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
                to_string(prev_f->f_code->co_filename),
                (int)PyFrame_GetLineNumber(prev_f)});
        }
    }
    if (stacks.size() == 0) {
        auto m = std::min(3,n);
        for (int i=0; i<m; i++) {
            auto f = frames[n-m+i];
            auto s = to_string(f->f_code->co_filename);
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

void TraceData::record_node(Node* node, bool record_stack) {
    if (thread_name.size()) return;
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
    if (node->__id())
        data.attrs["__id"] = S(node->__id());
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
    if (thread_name.size()) return;
    auto iter = trace_data.id_map.find(node);
    if (iter == trace_data.id_map.end())
        return;
    auto node_id = iter->second;
    id_map.erase(node);
    if (trace_py_var < 2) {
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

string _get_stack_info(Node* node) {
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
    }
    return stack_info;
}

void print_node_trace(const Node* node, std::ostream& os) {
    os << _get_stack_info((Node*)node);
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
