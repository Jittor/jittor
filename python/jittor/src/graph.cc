// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include <algorithm>
#include "graph.h"
#include "misc/node_index.h"
#include "var_holder.h"
#include "var.h"

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, check_graph, 0, "Unify graph sanity check.");

// do_graph_check's dangling-node sweep walks lived_nodes, which used to be
// filled only under `#ifdef NODE_MEMCHECK`. In every build anyone ships that
// made check_graph=1 sweep an empty table and report success -- half of the
// only cross-check of the three liveness counters was a no-op exactly where
// it was supposed to be watching. Registering nodes is what costs (a hash
// insert per node), not the check, so the registry follows the flag instead
// of the build type. Nodes made before the flag went on are not in it, which
// costs coverage but can never produce a false report.
void setter_check_graph(const int& old_value, const int& value) {
    node_track_lived = value != 0;
}


template <typename T>
string ss_convert(T x) {
    std::stringstream ss;
    ss << x;
    return ss.str();
}

int64 do_graph_check() {
    vector<Node*> queue;
    unordered_map<Node*,int> visited;
    for (auto& vh : runtime_holder_state().holders()) {
        if (0==visited[vh->var]++)
            queue.push_back(vh->var);
    }
    LOGvv << "Check hold_vars size" << queue.size();
    int vhsize = queue.size();
    for (auto* node : queue) {
        // Forward liveness may legitimately be zero for a held, finished var.
        ASSERTop(node->liveness.backward.count(),>,0);
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
            f+=visited.at(node), b+=visited.at(node), p+=visited.at(node);
        }
        for (auto* i : node->inputs()) {
            if (i->is_stop_grad()) continue;
            if (!i->liveness.forward.active()) continue;
            f ++;
        }
        for (auto* o : node->outputs()) {
            if (o->liveness.backward.active())
                b ++;
            if (o->liveness.pending.active() && !o->is_finished())
                p++;
        }
        // if (f>0 && b>0 && !node->is_finished()) p++;
        node->liveness.assert_expected(f, b, p, node);
    }
    int64 swept = 0;
    for (auto& kv : lived_nodes) {
        if (!kv.second) continue;
        auto* node = (Node*) kv.first;
        swept++;
        if (!visited.count(node) && !node->flags.get(NodeFlags::_released)) {
            if (node->is_var() && node->_inputs.size())
                continue;
            LOGf << "ERROR dnode" << (void*)node << kv.second << node;
        }
    }
    // Say which half ran. Reporting "all clear" while half the check swept an
    // empty table is the original defect; a check that cannot run has to be
    // heard, not assumed.
    if (swept == 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOGw << "graph check: liveness verified over" << (int64)queue.size()
                << "nodes, but the dangling-node half swept nothing -- no node"
                << "is registered. Nodes register while check_graph is on, or"
                << "always in a NODE_MEMCHECK build; nodes made before it was"
                << "switched on are not covered.";
        }
    } else
        LOGvv << "graph check: liveness over" << (int64)queue.size()
            << "nodes, dangling-node sweep over" << swept << "registered nodes";
    return swept;
}

DumpGraphs dump_all_graphs() {
    DumpGraphs graphs;
    vector<Node*> queue;
    unordered_set<Node*> seeds;
    for (auto& vh : runtime_holder_state().holders())
        if (seeds.insert(vh->var).second) {
            queue.push_back(vh->var);
            graphs.hold_vars.emplace_back(ss_convert(vh->var));
        }
    bfs_both(queue, [](Node*){return true;});
    std::sort(queue.begin(), queue.end(),
        [](Node* a, Node* b) { return a->id < b->id;});
    // Position of each node in `queue`, kept here rather than in
    // Node::custom_data: dump_all_graphs is a debug entry point and can be
    // called at any moment, including from a breakpoint inside an execution
    // that is keeping its own numbering in that slot.
    NodeIndex index;
    index.reset(queue.size());
    for (uint i=0; i<queue.size(); i++)
        index[queue[i]] = i;
    for (Node* node : queue) {
        graphs.nodes_info.emplace_back(ss_convert(node));
        
        graphs.inputs.emplace_back();
        auto& inputs = graphs.inputs.back();
        inputs.reserve(node->_inputs.size());
        for (auto i : node->_inputs)
            inputs.push_back(index.get(i.node));

        graphs.outputs.emplace_back();
        auto& outputs = graphs.outputs.back();
        outputs.reserve(node->_outputs.size());
        for (auto o : node->_outputs)
            outputs.push_back(index.get(o.node));
    }
    return graphs;
}

void clean_graph() {
    vector<Node*> queue;
    unordered_set<Node*> seeds;
    for (auto& vh : runtime_holder_state().holders())
        if (seeds.insert(vh->var).second) {
            queue.push_back(vh->var);
        }
    bfs_both(queue, [](Node*){return true;});
    unordered_set<Node*> held;
    held.reserve(runtime_holder_state().holders().size());
    for (auto& vh : runtime_holder_state().holders())
        held.insert(vh->var);
    SetupFreeBuffer setup_free_buffer;
    for (auto node : queue) {
        if (!held.count(node)) {
            node->set_stop_grad();
        }
    }
}

void check_circle(Node* s) {
    vector<Node*> q = {s};
    vector<int> fa = {-1};
    unordered_set<Node*> visited = {s};
    for (int i=0; i<q.size(); i++) {
        auto n = q[i];
        for (auto o : n->outputs()) {
            if (o == s) {
                LOGe << "Found circle:";
                int j=i;
                vector<Node*> nodes{o};
                while (j) {
                    nodes.push_back(q[j]);
                    j = fa[j];
                }
                for (int i=0; i<nodes.size(); i++) {
                    auto n = nodes[i];
                    auto out = nodes[(i-1+nodes.size())%nodes.size()];
                    auto in = nodes[(i+1)%nodes.size()];
                    int in_id=0, out_id=0;
                    for (auto ii : n->inputs()) {
                        if (ii == in) break;
                        in_id ++;
                    }
                    for (auto oo : n->outputs()) {
                        if (oo == out) break;
                        out_id ++;
                    }
                    LOGe << n << "in:" >> in_id >> '/' >> n->inputs().size() << "out:" >> out_id >> '/' >> n->outputs().size();
                }
                LOGf << "found circle";
            }
            if (!visited.count(o)) {
                visited.emplace(o);
                q.push_back(o);
                fa.push_back(i);
            }
        }
    }

}

} // jittor
