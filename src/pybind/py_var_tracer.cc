// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind/py_var_tracer.h"
#include "misc/str_utils.h"
#include "op.h"
#include "var.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace jittor {

DEFINE_FLAG(int, trace_py_var, 0, "Trace py stack max depth for debug.");

unordered_map<const Node*, string> trace_data;

void __registe_node_trace(Node* node) {
    auto py_stacks = 
        py::module::import("traceback")
        .attr("extract_stack")(nullptr, trace_py_var);
    auto len = py_stacks.attr("__len__")().cast<int>();
    string info;
    for (int i=0; i<len; i++) {
        auto py_stack = py_stacks.attr("__getitem__")(i);
        auto filename = py_stack.attr("filename").cast<string>();
        if (len==1)
            info += split(filename, "/").back();
        else {
            info += "\n        ";
            info += filename;
        }
        info += ':';
        info +=  py_stack.attr("name").cast<string>();
        info += ':';
        info +=  S(py_stack.attr("lineno").cast<int>());
        info += ':';
        info +=  py_stack.attr("line").cast<string>();
    }
    trace_data[node] = info;
}

void __unregiste_node_trace(Node* node) {
    trace_data.erase(node);
}

void __registe_node_trace_grad(Node* g, Node* node, int x_index) {
    if (!g) return;
    string& gname = trace_data.at(g);
    string name = "grad(";
    if (startswith(gname, "grad("))
        return;
    if (!node->is_var()) {
        name += node->op()->name_ex();
        name += ':';
        name += S(x_index);
    }
    name += ":" + gname;
    name += "):";
    name += trace_data.at(node);
    gname = name;
    std::function<void(Node*)> dfs = [&] (Node* node) {
        for (Node* i : node->inputs()) {
            string& iname = trace_data[i];
            if (iname.find("__init__.py:grad:") != string::npos && !startswith(iname, "grad(")) {
                iname = name;
                dfs(i);
            }
        }
    };
    dfs(g);
}

void __print_node_trace(const Node* node, std::ostream& os) {
    if (trace_data.count(node))
        os << '{' << trace_data.at(node) << '}';
}

} // jittor
