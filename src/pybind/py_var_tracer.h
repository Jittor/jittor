// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

DECLARE_FLAG(int, trace_py_var);


#ifdef NODE_MEMCHECK
void __registe_node_trace(Node* node);
void __unregiste_node_trace(Node* node);
void __registe_node_trace_grad(Node* g, Node* node, int x_index);
void __print_node_trace(const Node* node, std::ostream& os);

inline void registe_node_trace(Node* node) 
    { if (trace_py_var) __registe_node_trace(node); }
inline void unregiste_node_trace(Node* node) 
    { if (trace_py_var) __unregiste_node_trace(node); }
inline void registe_node_trace_grad(Node* g, Node* node, int x_index)
    { if (trace_py_var) __registe_node_trace_grad(g, node, x_index); }
inline void print_node_trace(const Node* node, std::ostream& os)
    { if (trace_py_var) __print_node_trace(node, os); }
#else
inline void registe_node_trace(Node* node) {}
inline void unregiste_node_trace(Node* node) {}
inline void registe_node_trace_grad(Node* g, Node* node, int x_index) {}
inline void print_node_trace(const Node* node, std::ostream& os) {}
#endif

} // jittor
