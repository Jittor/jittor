// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "op.h"
#include "var.h"
#include "opt/var_relay.h"
#include "ops/op_register.h"
#include "fused_op.h"
#include "graph.h"
#include "op_compiler.h"
#include "mem/allocator.h"

namespace jittor {

static auto make_binary_op = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_broadcast_to_op = get_op_info("broadcast_to")
    .get_constructor<VarPtr, Var*, NanoVector, NanoVector>();
static auto make_reduce = get_op_info("reduce")
    .get_constructor<VarPtr, Var*, NanoString, NanoVector, bool>();

JIT_TEST(op_register) {
    VarPtr a({10,10,1}, "float32");
    VarPtr b({1,10,10}, "float32");
    auto c = make_binary_op(a, b, ns_add);
    CHECK(c->size==1000*4);
    CHECK(c->input()->name_ex()=="binary.add");
}

JIT_TEST(fused_op_relay_matmul) {
    VarPtr a({10,10}, "float32");
    VarPtr b({10,10}, "float32");
    auto aa = make_broadcast_to_op(a, {10,10,10}, {2});
    auto bb = make_broadcast_to_op(b, {10,10,10}, {0});
    auto c = make_binary_op(aa, bb, ns_add);
    auto d = make_reduce(c, ns_add, 1, false);
    vector<Node*> s({d->node()}), q;
    vector<Op*> ops;
    bfs_backward(s, q, [&](Node *node) -> bool {
        node->custom_data=0;
        if (!node->is_var()) ops.push_back(node->op());
        return true;
    });
    CHECKop(q.size(),==,10);
    CHECKop(ops.size(),==,4);
    for (auto op : ops) op->do_jit_prepare(jk);
    FusedOp fop;
    FusedOpContext context;
    fop.context = &context;
    context.vrm.set_fused_op(&fop);
    for (uint i=0; i<ops.size(); i++)
        fop.ops.push_back(ops.at(ops.size()-i-1));
    // a, b, d can not fuse
    a->custom_data = b->custom_data = d->custom_data = 1;
    fop.update_ops();
    context.setup(&fop);
    if (!has_op("mkl_matmul")) return;
    auto make_matmul = get_op_info("mkl_matmul")
        .get_constructor<VarPtr, Var*, Var*, bool, bool>();
    auto rvar = make_matmul(a, b, 0, 0);

    fop.context->vrm.add_relay_group({{rvar, d}});
    CHECKop(context.vrm.relay_groups[0].removed_input_vars.size(),==,2);
    auto is_op_relayed = context.vrm.get_op_relay_info({1});
    for (auto v : is_op_relayed) CHECK(v.first==0 && v.second==0);

    // test2
    for (Node* node : q) node->custom_data = 0;
    // a, b, d can not fuse
    a->custom_data = b->custom_data = d->custom_data = 1;
    // broadcast(a) can not fused
    fop.vars[1].var->custom_data = 1;
    fop.update_ops();
    context.setup(&fop);
    is_op_relayed = context.vrm.get_op_relay_info({1});
    vector<pair<int,int>> ans{{-1,-1},{0,0},{0,0},{0,0}};
    CHECKop(is_op_relayed,==,ans);
    auto& oprc = context.vrm.relay_groups[0].oprcs[0];
    CHECKop(oprc.op,==,rvar->input());
    // matmul op.x --> a, op.y --> b, op.z --> d
    CHECK(oprc.relayed_members[0]==(a->custom_data>>2));
    CHECK(oprc.relayed_members[1]==(b->custom_data>>2));
    CHECK(oprc.relayed_members[2]==(d->custom_data>>2));
    auto src = context.vrm.get_relay_src(0,0);

    auto& loop_options = fop.get_loop_options_tuned();
    loop_options["relay0"] = 1;
    OpCompiler oc(&fop);

    auto allocator = get_allocator();
    for (auto& v : fop.vars)
        if (v.type!=1) v.var->alloc(allocator);
    auto entry = oc.compile("[OP:_fused_op_relay_matmul]", oc.src);
    for (uint i=0; i<a->num; i++)
        a->ptr<float>()[i] = b->ptr<float>()[i] = 1;
    entry(&fop);
    for (uint i=0; i<a->num; i++)
        CHECK(d->ptr<float>()[i]==10);
}

} // jittor
