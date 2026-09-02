// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "var.h"

namespace jittor {

struct OpRelayContext {
    Op* op;
    // j=relayed_members[i] represents: op's i-th member is relay to fused_op's j-th vars
    vector<int> relayed_members;
    // name and byte offset of each Var* member of op, copied from the op
    // registry when the relay group is built.
    //
    // The generated kernel names the member it wants to set and this resolves
    // the name here.  It must not carry the offset instead: the kernel source
    // is written to the JIT cache and reused, while the offset belongs to a
    // struct layout that changes whenever an op gains a member or the compiler
    // ABI changes -- and the jit key says nothing about either.  A kernel that
    // outlives the layout it was generated against writes a Var* into the
    // wrong member and nothing reports it.
    vector<pair<string, uint64>> var_members;
    // set op's Var* member `name` to v; fails if op has no such member
    void set_var_member(const char* name, Var* v);
};

struct VarRelayGroup {
    // pair<VarPtr, uint64>: VarPtr: relay source, uint64: relay target var id in fused_op
    vector<pair<VarPtr, uint64>> relayed_pairs;
    vector<Var*> removed_input_vars;
    // nodes of relay source
    vector<Node*> nodes;
    vector<OpRelayContext> oprcs;
    VarRelayGroup();
    VarRelayGroup(const VarRelayGroup&) = delete;
    VarRelayGroup(VarRelayGroup&&);
    ~VarRelayGroup();
};

struct VarRelayManager {
    FusedOp* fop = nullptr;
    vector<VarRelayGroup> relay_groups;
    
    void set_fused_op(FusedOp* fop) {this->fop=fop;}
    /* add_relay_group: add relay group into current fused_op
        group: list of pair of source and target vars
        return: relay group id
     */
    int add_relay_group(const vector<pair<Var*, Var*>>& group);
    /* get_op_relay_info
        relay_switches: switches control the on or off of each relay
        return: relay group id and op id
     */
    vector<pair<int,int>> get_op_relay_info(const vector<bool>& relay_switches);

    string get_relay_src(int group_id, int op_id);
};

} // jittor
