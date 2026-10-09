// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/op.h"
#include "core/var.h"
#include "codegen/opt/var_relay.h"
#include "ops/op_register.h"
#include "core/fused_op.h"
#include "core/graph.h"
#include "core/executor.h"

namespace jittor {

static auto make_binary_op = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_broadcast_to_op = op_constructor<VarPtr, Var*, NanoVector, NanoVector>("broadcast_to");
static auto make_reduce = op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");
static auto make_array = op_constructor<VarPtr, const void*, NanoVector, NanoString>("array");

JIT_TEST(op_register) {
    VarPtr a({10,10,1}, "float32");
    VarPtr b({1,10,10}, "float32");
    auto c = make_binary_op(a, b, ns_add);
    CHECK(c->size==1000*4);
    CHECK(c->input()->is_op(op_ids::binary()));
    CHECK(c->input()->ns == ns_add);
}

JIT_TEST(nested_run_sync_restores_outer_epoch) {
    float lhs[] = {1.f, 2.f, 3.f, 4.f};
    float rhs[] = {5.f, 6.f, 7.f, 8.f};
    auto a = make_array(lhs, {4}, ns_float32);
    auto b = make_array(rhs, {4}, ns_float32);
    auto c = make_binary_op(a, b, ns_add);

    vector<Node*> graph{c.ptr};
    bfs_backward(graph, [](Node*) { return true; });
    TraversalEpoch outer("outer_around_run_sync");
    for (Node* node : graph) outer.mark(node);

    runtime_executor().run_sync({c.ptr}, false);
    for (Node* node : graph) CHECK(outer.marked(node));
}

// Mark vars the batch says have to stay in memory, the way run_sync's
// var_fused does: by position in the batch, not by a bit on the node.
static void fop_cannot_fuse(vector<int>& var_fused, int64 batch_stamp,
                            std::initializer_list<Var*> vars) {
    for (Var* v : vars)
        var_fused[((Node*)v)->batch_index_at(batch_stamp)] = 1;
}

JIT_TEST(fused_op_relay_matmul) {
    JK& jk = get_jk();
    VarPtr a({10,10}, "float32");
    VarPtr b({10,10}, "float32");
    auto aa = make_broadcast_to_op(a, {10,10,10}, {2});
    auto bb = make_broadcast_to_op(b, {10,10,10}, {0});
    auto c = make_binary_op(aa, bb, ns_add);
    auto d = make_reduce(c, ns_add, 1, false);
    vector<Node*> s({d->node()}), q;
    vector<Op*> ops;
    bfs_backward(s, q, [&](Node *node) -> bool {
        if (!node->is_var()) ops.push_back(node->op());
        return true;
    });
    // "a, b, d have to stay in memory" used to be written as bit 0 of each
    // node's custom_data -- cleared for the whole batch above, then set for
    // the three -- into the same field update_ops() packs its own indices
    // into. It is the batch's verdict vector now, indexed the way the executor
    // indexes it, so stamp the batch the way run_sync does.
    TraversalEpoch batch_epoch("op_relay_batch");
    int64 batch_stamp = batch_epoch.stamp;
    for (uint i=0; i<q.size(); i++) q[i]->set_batch_index(batch_stamp, i);
    vector<int> var_fused(q.size(), 0);
    CHECKop(q.size(),==,10);
    CHECKop(ops.size(),==,4);
    for (auto op : ops) op->do_jit_prepare(jk);
    FusedOp fop;
    // A stack context, so `fop.context_owner` stays null: the cache is not
    // involved here.
    FusedOpContext context;
    fop.context = &context;
    for (uint i=0; i<ops.size(); i++)
        fop.ops.push_back(ops.at(ops.size()-i-1));
    fop.batch_var_fused = &var_fused;
    fop.batch_stamp_wanted = batch_stamp;
    fop_cannot_fuse(var_fused, batch_stamp, {a.ptr, b.ptr, d.ptr});
    fop.update_ops();
    context.setup(&fop);
    if (!has_op("mkl_matmul")) return;
    auto make_matmul = get_op_info("mkl_matmul")
        .get_constructor<VarPtr, Var*, Var*, bool, bool>();
    auto rvar = make_matmul(a, b, 0, 0);

    fop.context->vrm.add_relay_group(&fop, {{rvar, d}});
    CHECKop(context.vrm.relay_groups[0].removed_input_vars.size(),==,2);
    auto is_op_relayed = context.vrm.get_op_relay_info(&fop, {1});
    for (auto v : is_op_relayed) CHECK(v.first==0 && v.second==0);

    // test2
    for (uint i=0; i<var_fused.size(); i++) var_fused[i] = 0;
    fop_cannot_fuse(var_fused, batch_stamp, {a.ptr, b.ptr, d.ptr});
    // broadcast(a) can not fused
    fop_cannot_fuse(var_fused, batch_stamp, {fop.vars[1].var});
    fop.update_ops();
    context.setup(&fop);
    is_op_relayed = context.vrm.get_op_relay_info(&fop, {1});
    vector<pair<int,int>> ans{{-1,-1},{0,0},{0,0},{0,0}};
    CHECKop(is_op_relayed,==,ans);
    auto& oprc = context.vrm.relay_groups[0].oprcs[0];
    CHECKop(oprc.op,==,rvar->input());
    // matmul op.x --> a, op.y --> b, op.z --> d
    CHECK(oprc.relayed_members[0]==fop.var_index.at(a.ptr));
    CHECK(oprc.relayed_members[1]==fop.var_index.at(b.ptr));
    CHECK(oprc.relayed_members[2]==fop.var_index.at(d.ptr));
    auto src = context.vrm.get_relay_src(0,0);
    CHECK(src.find("@relay_op") != string::npos);
    CHECK(src.find("set_var_member") != string::npos);
    CHECK(src.find("run_registered") != string::npos);

    // BroadcastTo is now a storage view (OpType::other). The production
    // fuser keeps such views outside fused kernels. Compiling this manually
    // assembled graph would bypass that boundary and reject its view source;
    // the registered matmul and convolution rows test actual execution.
    CHECK(aa->input()->type() == OpType::other);
    CHECK(bb->input()->type() == OpType::other);
}

} // jittor
