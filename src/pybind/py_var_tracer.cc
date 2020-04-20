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

DEFINE_FLAG(int, trace_py_var, 0, "Trace py stack for debug.");

unordered_map<const Node*, string> trace_data;

void __registe_node_trace(Node* node) {
    auto py_stack = 
        py::module::import("traceback")
        .attr("extract_stack")(nullptr, 1).attr("__getitem__")(0);
    auto filename = py_stack.attr("filename").cast<string>();
    auto basename = split(filename, "/").back();
    basename += ':';
    basename +=  py_stack.attr("name").cast<string>();
    basename += ':';
    basename +=  S(py_stack.attr("lineno").cast<int>());
    basename += ':';
    basename +=  py_stack.attr("line").cast<string>();
    trace_data[node] = basename;
}

void __unregiste_node_trace(Node* node) {
    trace_data.erase(node);
}

void __registe_node_trace_grad(Node* g, Node* node, int x_index) {
    if (!g) return;
    string name = "grad(";
    if (!node->is_var()) {
        name += node->op()->name_ex();
        name += ':';
        name += S(x_index);
    }
    name += "):";
    name += trace_data.at(node);
    trace_data[g] = name;
    std::function<void(Node*)> dfs = [&] (Node* node) {
        for (Node* i : node->inputs()) {
            string& iname = trace_data[i];
            if (startswith(iname, "__init__.py:grad:")) {
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
