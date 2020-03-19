// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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
