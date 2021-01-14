// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <unordered_set>
#include <sstream>
#include "ops/op_register.h"
#include "opt/var_relay.h"
#include "fused_op.h"
#include "graph.h"

namespace jittor {


VarRelayGroup::VarRelayGroup() {
}

VarRelayGroup::VarRelayGroup(VarRelayGroup&& other) {
    relayed_pairs = move(other.relayed_pairs);
    removed_input_vars = move(other.removed_input_vars);
    nodes = move(other.nodes);
    oprcs = move(other.oprcs);
}

VarRelayGroup::~VarRelayGroup() {
    for (Node* node : nodes)
        if (node->is_var())
            Var::number_of_lived_vars++;
        else
            Op::number_of_lived_ops++;
}

int VarRelayManager::add_relay_group(const vector<pair<Var*, Var*>>& group) {
    for (auto& g : relay_groups)
        for (auto& p : g.relayed_pairs)
            for (auto& p2 : group)
            if (p.second == (fop->get_node_id(p2.second))) {
                LOGvvvv << "Var allready relayed" << p2.second;
                return -1;
            }
    relay_groups.emplace_back();
    auto& relay_group = relay_groups.back();
    relay_group.relayed_pairs.reserve(group.size());
    for (const auto& p : group) {
        relay_group.relayed_pairs.push_back({p.first, fop->get_node_id(p.second)});
        ASSERTop(p.first->size,==,p.second->size);
    }
    
    // break the input link between relay and target
    std::unordered_set<Node*> fnodes;
    fnodes.reserve(fop->ops.size()+fop->vars.size());
    for (auto& op : fop->ops) fnodes.insert(op);
    for (auto& v : fop->vars) fnodes.insert(v.var); 
    vector<Node*>& q = relay_group.nodes;
    vector<Node*> s;
    for (const auto& p : group) {
        s.push_back(p.first->node());
        ASSERT(!fnodes.count(p.first)) << "Relayed source should not in fused_op";
    }
    bfs_backward(s, q, [&](Node *node) -> bool {
        return !fnodes.count(node);
    });
    // currently, we only support single op relay
    ASSERT(q.size()==2*group.size());
    for (Node* node : q) {
        node->__release();
        if (node->is_var())
            continue;
        Op* op = node->op();
        op->do_jit_prepare(jk);
        list<Node*> new_inputs;
        int removed = 0;
        for (Var* v : op->inputs())
            if (!fnodes.count(v))
                new_inputs.push_back(v->node());
            else {
                removed++;
                relay_group.removed_input_vars.push_back(v);
            }
        if (removed) {
            op->set_inputs(move(new_inputs));
            LOGvvv << "Remove" << removed << "inputs from" << op;
        }
    }
    // generate OpRelayContext
    relay_group.oprcs.resize(relay_group.relayed_pairs.size());
    for (uint i=0; i<relay_group.oprcs.size(); i++) {
        auto& oprc = relay_group.oprcs[i];
        auto& p = relay_group.relayed_pairs[i];
        oprc.op = p.first->input();
        auto op_info = get_op_info(oprc.op->name());
        oprc.relayed_members.resize(op_info.var_members.size());
        for (uint i=0; i<op_info.var_members.size(); i++) {
            auto offset = op_info.var_members[i].second;
            Var* v = GET_VAR_MEMBER(oprc.op, offset);
            if (v==p.first.ptr)
                oprc.relayed_members[i] = p.second;
            else if (v == nullptr)
                oprc.relayed_members[i] = -1;
            else {
                ASSERT(fnodes.count(v));
                oprc.relayed_members[i] = fop->get_node_id(v);
            }
            LOGvvvv << "Relay op" << oprc.op->name() >>".">>
                op_info.var_members[i].first << "-->" <<
                oprc.relayed_members[i];
        }
    }
    return (int)relay_groups.size()-1;
}

vector<pair<int,int>> VarRelayManager::get_op_relay_info(const vector<bool>& relay_switches) {
    ASSERT(relay_switches.size()==relay_groups.size());
    auto num = fop->ops.size()+fop->vars.size();
    auto node_id = [&](Node* node) -> int {
        if (node->is_var()) return fop->get_node_id(node);
        return fop->get_node_id(node) + fop->vars.size();
    };
    vector<int> deps(num);
    // pair: first: group_id, second: relayed_pair id
    vector<pair<int,int>> relay_source(num, {-1,-1});
    vector<bool> is_relayed(num);
    for (uint i=0; i<relay_groups.size(); i++) {
        if (!relay_switches[i]) continue;
        uint j=0;
        for (const auto& p : relay_groups[i].relayed_pairs) {
            is_relayed[p.second] = 1;
            relay_source[p.second] = {i, j};
            j++;
        }
    }
    vector<Node*> nodes(num);
    for (auto v : fop->vars) {
        auto vid = node_id(v.var);
        nodes[vid] = v.var;
        // if is input, continue
        if (v.type==0) continue;
        // add input op dependency
        deps[node_id(v.var->input())]++;
        // if var is relayed
        if (is_relayed[vid]) continue;
        // if is output, add dependency
        if (v.type==2)
            deps[vid]++;
    }
    for (auto op : fop->ops) {
        nodes[node_id(op)] = op;
        for (auto var : op->inputs()) {
            deps[node_id(var)]++;
        }
    }
    vector<uint> q;
    q.reserve(num);
    for (uint i=0; i<num; i++)
        if (!deps[i]) q.push_back(i);
    for (uint i=0; i<q.size(); i++) {
        auto nid = q[i];
        Node* node = nodes[nid];
        // if is input of fused_op, continue;
        if (node->is_var() && fop->vars[nid].type==0)
            continue;
        for (auto i : node->_inputs) {
            auto nnid = node_id(i.node);
            deps[nnid]--;
            if (!deps[nnid]) {
                q.push_back(nnid);
                relay_source[nnid] = relay_source[nid];
            }
        }
    }
    relay_source.erase(relay_source.begin(), relay_source.begin()+fop->vars.size());
    return relay_source;
}

string VarRelayManager::get_relay_src(int group_id, int op_id) {
    auto& oprc = relay_groups[group_id].oprcs[op_id];
    Op* op = oprc.op;
    string name = op->name();
    auto op_info = get_op_info(name);
    string name2 = Op::op_name_to_file_name(name);
    string name3 = Op::file_name_to_class_name(name2);
    std::stringstream ss;
    string relay_op_name = "rop_"+S(group_id)+"_"+S(op_id);
    ss << "\n    // @relay_op\n";
    ss << "    Op* "<<relay_op_name<<
        " = context->vrm.relay_groups["<<group_id<<"].oprcs["<<op_id<<"].op;\n";
    for (uint i=0; i<oprc.relayed_members.size(); i++) {
        int j = oprc.relayed_members[i];
        auto offset = op_info.var_members[i].second;
        ss << "    GET_VAR_MEMBER("<<relay_op_name<<
            ", "<<offset<<") = vars["<<j<<"].var;\n";
    }
    ss << "    "<<relay_op_name<<"->do_run();\n";
    LOGvvv << "get_relay_src\n" << ss.str();
    return ss.str();
}

} // jittor
