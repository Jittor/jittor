// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "codegen/opt/var_relay.h"

namespace jittor {

struct VarInfo {
    Var* var;
    // 0: input, 1: intermediate, 2: output
    int type;
};
std::ostream& operator<<(std::ostream& os, const VarInfo& vi);

// Everything a compiled fused kernel needs that outlives the FusedOp it was
// compiled from. It is kept in the cache below and reused by every later batch
// whose fusion produces the same jit key, so it must not refer to the FusedOp
// that happened to be current when it was built: `setup` copies the numbering
// out, and `VarRelayManager` no longer keeps a `FusedOp*` at all (it used to,
// pointing at a `run_sync` stack frame or at a compile-worker temporary that
// was already gone -- see opt/var_relay.h).
struct FusedOpContext {
    VarRelayManager vrm;
    jit_op_entry_t entry;
    // Node -> index within the fusion this context was compiled from. Read
    // during code generation only; the pointers are not dereferenced and mean
    // nothing after that fusion is gone.
    unordered_map<Node*, int> node_id;
    void setup(FusedOp* fop);
};

// The compiled fused kernels, by jit key. Bounded (utils/jit_cache_map.h) and
// owning: two keys map to one context -- the key the fusion prepares and the
// key its tuned kernel was compiled under -- so dropping either of them must
// not free it, and dropping both must. It was a raw `FusedOpContext*` that
// nothing ever freed.
EXTERN_LIB jit_cache_map<shared_ptr<FusedOpContext>> jit_fused_ops;

struct FusedOp final : Op {
    vector<Op*> ops;
    // Stable numbering owned by this fusion across load/update, relay and
    // code-generation stages. This replaces Node::custom_data bit-packing.
    unordered_map<Op*, uint> op_index;
    unordered_map<Var*, uint> var_index;
    // edges: [[i,j,k,l], ...] represents opi.output(j) == opk.input(l)
    vector<std::tuple<uint,uint,uint,uint>> edges;
    vector<VarInfo> vars;
    loop_options_t loop_options_merged, loop_options_tuned;
    loop_options_t* loop_options, * loop_options_origin;
    loop_options_t& get_loop_options_tuned();
    // The generated kernel reads `context->vrm...` by that name, so this stays
    // a raw pointer. `context_owner` holds the cache's reference for as long
    // as this op exists, so an eviction cannot free the context underneath a
    // running kernel; it is null when the context is not the cache's --
    // src/tests/test_op_relay.cc puts one on the stack.
    FusedOpContext* context;
    shared_ptr<FusedOpContext> context_owner;

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
    void prepare_fused_key(JK& jk);
    void execute_fused_prepared(JK& jk);
#ifdef JIT
    void jit_run();
#endif
};

}
