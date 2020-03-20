// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "graph.h"
#include "var_holder.h"
#include "var.h"

namespace jittor {

DEFINE_FLAG(int, check_graph, 0, "Unify graph sanity check.");

extern unordered_map<void*, int64> lived_nodes;

template <typename T>
string ss_convert(T x) {
    std::stringstream ss;
    ss << x;
    return ss.str();
}

void do_graph_check() {
    vector<Node*> queue;
    unordered_map<Node*,int> visited;
    for (auto& vh : VarHolder::hold_vars) {
        if (0==visited[vh->var]++)
            queue.push_back(vh->var);
    }
    LOGvv << "Check hold_vars size" << queue.size();
    int vhsize = queue.size();
    for (auto* node : queue) {
        ASSERTop(node->forward_liveness,>,0);
        ASSERTop(node->backward_liveness,>,0);
    }
    for (uint i=0; i<queue.size(); i++) {
        auto* node = queue[i];
        for (auto* i : node->inputs()) {
            if (visited.count(i)) continue;
            visited[i] = 0;
            queue.push_back(i);
        }
    }
    LOGvv << "Check all var size" << queue.size();
    for (int i=0; i<(int)queue.size(); i++) {
        auto* node = queue[i];
        LOGvvvv << "Check node" << i << node;
        int f=0, b=0, p=0;
        if (i<vhsize) {
            f+=visited.at(node), b+=visited.at(node);
        }
        for (auto* i : node->inputs()) {
            if (i->is_stop_grad()) continue;
            if (!i->forward_liveness) continue;
            f ++;
        }
        for (auto* o : node->outputs()) {
            if (o->backward_liveness)
                b ++;
            if (o->pending_liveness && !o->is_finished())
                p++;
        }
        if (f>0 && b>0 && !node->is_finished()) p++;
        if (f!=node->forward_liveness || b!=node->backward_liveness || p!=node->pending_liveness) {
            LOGf << "ERROR" << node << '\n' 
                << f << b << p << i << '\n' 
                << node->inputs() << '\n' 
                << node->outputs();
            continue;
        }
    }
    for (auto& kv : lived_nodes) {
        if (!kv.second) continue;
        auto* node = (Node*) kv.first;
        if (!visited.count(node) && node->tflag != -1) {
            if (node->is_var() && node->_inputs.size())
                continue;
            LOGf << "ERROR dnode" << (void*)node << kv.second << node;
        }
    }
}

DumpGraphs dump_all_graphs() {
    vector<Node*> queue;
    auto t = ++Node::tflag_count;
    for (auto& vh : VarHolder::hold_vars)
        if (vh->var->tflag != t) {
            vh->var->tflag = t;
            queue.push_back(vh->var);
        }
    bfs_both(queue, [](Node*){return true;});
    DumpGraphs graphs;
    for (uint i=0; i<queue.size(); i++)
        queue[i]->custom_data = i;
    for (Node* node : queue) {
        graphs.nodes_info.emplace_back(ss_convert(node));
        
        graphs.inputs.emplace_back();
        auto& inputs = graphs.inputs.back();
        inputs.reserve(node->_inputs.size());
        for (auto i : node->_inputs)
            inputs.push_back(i.node->custom_data);

        graphs.outputs.emplace_back();
        auto& outputs = graphs.outputs.back();
        outputs.reserve(node->_outputs.size());
        for (auto o : node->_outputs)
            outputs.push_back(o.node->custom_data);
    }
    return graphs;
}

} // jittor