// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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

EXTERN_LIB string_view_map<FusedOpContext*> jit_fused_ops;

struct FusedOp final : Op {
    vector<Op*> ops;
    // edges: [[i,j,k,l], ...] represents opi.output(j) == opk.input(l)
    vector<std::tuple<uint,uint,uint,uint>> edges;
    vector<VarInfo> vars;
    loop_options_t loop_options_merged, loop_options_tuned;
    loop_options_t* loop_options, * loop_options_origin;
    loop_options_t& get_loop_options_tuned();
    FusedOpContext* context;

    // The batch's fusion verdict, borrowed from the run_sync frame that built
    // this group: 1 the var has to stay in memory, 0 it may be fused away,
    // indexed by Node::batch_index. It used to arrive as bit 0 of the same
    // Node::custom_data int update_ops() packs its own var indices into, which
    // meant the executor and this op wrote to one field and neither could say
    // "not classified".
    const vector<int>* batch_var_fused = nullptr;
    // The stamp of that batch; a node's batch_index means something only while
    // Node::batch_stamp matches it.
    int64 batch_stamp_wanted = 0;

    // A var the batch classified as fusable may be fused away. A var that is
    // not in the batch -- a multi-output op only one of whose outputs this
    // execution needs -- has no verdict, and the only safe answer there is
    // "keep it": materialising a var that could have been fused away costs
    // memory, dropping one that could not costs the value. The bit-packed
    // field had no way to say "not classified"; it returned whatever bit 0
    // happened to hold from an earlier batch (the commented-out block in
    // load_fused_op is somebody meeting the same case, with a comment that
    // contradicts its own code).
    inline bool var_stays_in_memory(Node* v) const {
        if (!batch_var_fused) return false;
        if (v->batch_stamp != batch_stamp_wanted) return true;
        return (*batch_var_fused)[v->batch_index_at(batch_stamp_wanted)] == 1;
    }

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