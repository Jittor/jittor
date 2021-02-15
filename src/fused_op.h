// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "opt/var_relay.h"

namespace jittor {

struct VarInfo {
    Var* var;
    // 0: input, 1: intermediate, 2: output
    int type;
};
std::ostream& operator<<(std::ostream& os, const VarInfo& vi);

struct FusedOpContext {
    VarRelayManager vrm;
    jit_op_entry_t entry;
    unordered_map<Node*, int> node_id;
    void setup(FusedOp* fop);
};

extern string_view_map<FusedOpContext*> jit_fused_ops;

struct FusedOp final : Op {
    vector<Op*> ops;
    // edges: [[i,j,k,l], ...] represents opi.output(j) == opk.input(i)
    vector<std::tuple<uint,uint,uint,uint>> edges;
    vector<VarInfo> vars;
    loop_options_t loop_options_merged, loop_options_tuned;
    loop_options_t* loop_options, * loop_options_origin;
    loop_options_t& get_loop_options_tuned();
    FusedOpContext* context;

    int get_node_id(Node* node);
    int has(Node* node);
    void update_ops();
    FusedOp();
    FusedOp(const FusedOp& other);
    ~FusedOp();

    int get_loop_option(const string& key, const int& _default=0);
    void add_loop_option_candidate(const string& key, int x);
    void update_jit_key();
    
    const char* name() const override { return "fused"; }
    void statistics(uint64_t& in, uint64_t& out, uint64_t& compute) override;
    void infer_shape() override;
    void do_jit_prepare(JK& jk) override;
    void do_prepare(JK& jk) override;
    void do_run_after_prepare(JK& jk) override;
    void do_run() override;
#ifdef JIT
    void jit_run();
#endif
};

}