// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

DECLARE_FLAG(int, trace_py_var);
extern Op* trace_grad_op;
struct JitKey;

struct Stack {
    string module_name;
    string module_type;
    string file_path;
    int lineno;
};

struct NodeData {
    int64 id;
    vector<int64> inputs;
    vector<int64> outputs;
    vector<Stack> stacks;
    /*
        if is var, then contain:
            is_var: 1
            shape: [a,b,c,d]
            ndim: x
            dtype: floatxx
            dsize: 4 or 8
            name: xxx
        if is op, then contain:
            is_var: 0
            name: xxx
            other op attr
     */
    unordered_map<string,string> attrs;
};

struct ExecuteOpInfo {
    vector<int64> fused_ops;
    string jit_key;
    string file_path;
    unordered_map<string,string> attrs;
};

struct TraceData {
    int64 node_data_cnt;
    int64 execute_op_info_cnt;
    unordered_map<int64, NodeData> node_data;
    unordered_map<int64, ExecuteOpInfo> execute_op_info;
    // jit_key map to id of execute_op_info
    unordered_map<string, vector<int64>> jit_key_map;
    unordered_map<Node*, int64> id_map;

    void record_node(Node* node, bool record_stack=true);
    void release_node(Node*);
    void record_op(Op* op);
    void record_exe_node(Node* node);
    void record_execution(Op* op, bool is_fused_op, JitKey& jk);
};

extern TraceData trace_data;

void print_node_trace(const Node* node, std::ostream& os);
vector<Stack> get_node_trace(Node* node);
} // jittor
