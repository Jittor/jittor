// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator.h"
#include "node.h"
#include "op.h"
#include "var.h"
#include <algorithm>
#include <functional>

namespace jittor {

int64 tflag_count = 0;
int64 nt = 0;
// See graph.cc: check_graph turns this on so that the dangling-node half of
// do_graph_check has something to sweep in a build without NODE_MEMCHECK.
int node_track_lived = 0;
unordered_map<void*, int64> lived_nodes;
unordered_map<int64, Node*> lived_nodes_id;
int64 total_node = 0;
vector<Node*> free_buffer;
NodeLifecycleObserver* node_lifecycle_observer = nullptr;

NodeLifecycleObserver* set_node_lifecycle_observer(NodeLifecycleObserver* observer) {
    NodeLifecycleObserver* previous = node_lifecycle_observer;
    node_lifecycle_observer = observer;
    return previous;
}

extern void free_var(Var* v);
extern void free_var_mem(Var* v);

// ---------------------------------------------------------------------------
// The three liveness counters
// ---------------------------------------------------------------------------
// Node carries three reference counts (see node.h for what contributes to
// each). Ownership changes propagate along graph edges, and a propagation step
// can create further propagation steps -- releasing forward liveness of a var
// can release the backward liveness of the same var, which frees it, which
// releases the liveness of all its inputs, and so on.
//
// Doing that recursively would blow the stack on long chains, so every step is
// pushed onto this queue instead and the queue is drained iteratively at the
// few entry points that are allowed to observe a consistent graph
// (own/release_both_liveness, finish_pending_liveness, release_inputs,
// set_inputs, add_inputs, set_stop_grad).
//
// Invariants:
//   * A callback running out of the queue may append to the queue. It must
//     therefore never hold a reference or iterator into it across the call.
//   * The queue is drained to completion by whoever started the drain, so
//     `liveness_queue_front` is 0 whenever a drain begins. It is a file static
//     rather than a local so that the "already draining" state is visible to
//     the whole file.
//   * Nodes deleted during a drain go to `free_buffer` and are destroyed by
//     the outermost SetupFreeBuffer, never in the middle of the drain.
//
// NOTE: one deliberate deviation from the shipped object. It stored these as
// `void (*)(Node*)`, casting `&Node::release_forward_liveness` -- a pointer to
// member -- to a plain function pointer and calling `op(node)`. That cast is
// undefined behaviour that happens to work on the Itanium ABI. Every operation
// in the queue is a non-virtual member of Node taking no arguments, so a real
// pointer to member expresses the same thing and `(node->*op)()` compiles to the
// same call. No symbol changes: the queue is a file static.
typedef void (Node::*liveness_op_t)();
static vector<pair<Node*, liveness_op_t>> liveness_queue;
static size_t liveness_queue_front = 0;

// Only used for logging: turns one of the six propagation steps back into a
// readable name.
static const char* liveness_op_name(liveness_op_t func) {
    if (func == &Node::release_forward_liveness) return "release_forward_liveness";
    if (func == &Node::release_backward_liveness) return "release_backward_liveness";
    if (func == &Node::release_pending_liveness) return "release_pending_liveness";
    if (func == &Node::own_forward_liveness) return "own_forward_liveness";
    if (func == &Node::own_backward_liveness) return "own_backward_liveness";
    if (func == &Node::own_pending_liveness) return "own_pending_liveness";
    return "unknown";
}

// Run every pending propagation step, including the ones the steps themselves
// append. `caller` only names the entry point in the log.
static void run_liveness_queue(const char* caller) {
    LOGvvvv << "run liveness queue from" << caller << "size" << liveness_queue.size();
    for (; liveness_queue_front < liveness_queue.size();) {
        // Copy the entry out before calling: the call may append to the
        // queue and reallocate it, invalidating any reference into it.
        Node* node = liveness_queue[liveness_queue_front].first;
        liveness_op_t op = liveness_queue[liveness_queue_front].second;
        liveness_queue_front++;
        LOGvvvv << "liveness" << liveness_op_name(op) << (void*)node;
        (node->*op)();
    }
    liveness_queue.clear();
    liveness_queue_front = 0;
}

// The cold half of Node::batch_index_at (node.h): a batch index read while the
// node carries a different batch's stamp is somebody else's number, which is
// exactly what the shared `custom_data` used to hand out silently.
void Node::batch_index_mismatch(int64 stamp) const {
    LOGf << "batch_index of" << this << "was written by batch" << batch_stamp
        << "but read as batch" << stamp >> ". A traversal is reading another"
        << "traversal's numbering.";
}

void Node::free() {
    CHECK_EXIST;
    // already scheduled for deletion in this free_buffer round
    if (tflag == nt) return;
    // A var that still has an input op and is either alive forward or not yet
    // finished is going to be recomputed or written; it is not garbage.
    if (is_var() && _inputs.size() && (forward_liveness || !is_finished())) {
        return;
    }
    tflag = nt;
    free_buffer.push_back(this);
    for (auto in : _inputs) {
        in.node->_outputs.erase(in.back);
        if (backward_liveness) {
            liveness_queue.emplace_back(in.node, &Node::release_backward_liveness);
        }
        if (pending_liveness && !is_finished())
            liveness_queue.emplace_back(in.node, &Node::release_pending_liveness);
    }
    _inputs.clear();
    for (auto out : _outputs) {
        out.node->_inputs.erase(out.back);
        if (!is_stop_grad()) {
            if (forward_liveness)
                liveness_queue.emplace_back(out.node, &Node::release_forward_liveness);
        }
        // an output var that nothing needs backward has just lost its only
        // producer, so it can go with us
        if (out.node->is_var() && out.node->backward_liveness == 0) out.node->free();
    }
    _outputs.clear();
    if (is_var()) free_var((Var*)this);
}

void Node::__release() {
    if (is_var())
        Var::number_of_lived_vars--;
    else
        Op::number_of_lived_ops--;
    tflag = -1;
}

// Empty in the shipped object too, not lost in the restoration. The memory
// checker it belongs to is `exist()` plus the `lived_nodes` maps above, all of
// which compile to nothing unless NODE_MEMCHECK is defined (node.h); this is the
// out-of-line half and it has no body on either side of that #ifdef. The single
// caller is Op::do_jit_prepare in op.cc. Left as it was found: what it should assert
// under NODE_MEMCHECK is not recoverable from the object.
void Node::memcheck_all_exist() const {
}

void Node::own_pending_liveness() {
    CHECK_EXIST;
    pending_liveness++;
    // p2: an unfinished node with pending liveness keeps its inputs pending
    if (pending_liveness == 1 && !is_finished())
        for (auto* in : inputs())
            liveness_queue.emplace_back(in, &Node::own_pending_liveness);
}

void Node::release_pending_liveness() {
    CHECK_EXIST;
    pending_liveness--;
    if (!pending_liveness && !is_finished()) {
        for (auto* in : inputs())
            liveness_queue.emplace_back(in, &Node::release_pending_liveness);
    }
    // Nothing is waiting to compute from this var any more. Its memory can go
    // even though the var itself stays alive, unless backward still needs it.
    if (pending_liveness == 0 && is_var()) {
        // _needed_by_backward is a Var flag, so it takes a Var* to read -- the
        // is_var() test above is now what makes the read well typed, not a
        // convention.
        Var* v = (Var*)this;
        if (v->mem_ptr != nullptr && v->flag(VarFlags::_needed_by_backward) == 0)
            free_var_mem(v);
    }
}

void Node::release_forward_liveness() {
    CHECK_EXIST;
    forward_liveness--;
    if (!forward_liveness) {
        // Snapshot the outputs: the propagation below can erase edges, and on
        // the second loop we may enqueue an operation on ourselves.
        int n = outputs().size(), i = 0;
        STACK_ALLOC(Node*, outs, n);
        for (auto* out : outputs()) {
            outs[i++] = out;
        }
        // f3: outputs lose the forward liveness we contributed
        if (!is_stop_grad()) {
            for (int i = 0; i < n; i++) {
                auto out = outs[i];
                liveness_queue.emplace_back(out, &Node::release_forward_liveness);
            }
        }
        // b3: a finished output var can no longer produce a gradient for us,
        // so the backward liveness it contributed goes away too
        if (backward_liveness) {
            for (int i = 0; i < n; i++) {
                auto out = outs[i];
                if (out->is_var() && out->is_finished()) {
                    if (out->is_stop_grad()) continue;
                    liveness_queue.emplace_back(this, &Node::release_backward_liveness);
                }
            }
        }
    }
}

void Node::own_forward_liveness() {
    CHECK_EXIST;
    forward_liveness++;
    if (forward_liveness == 1) {
        if (!is_stop_grad())
            for (auto* out : outputs())
                liveness_queue.emplace_back(out, &Node::own_forward_liveness);
    }
}

void Node::release_backward_liveness() {
    CHECK_EXIST;
    backward_liveness--;
    if (!backward_liveness) {
        int n = inputs().size(), i = 0;
        STACK_ALLOC(Node*, is, n);
        for (auto* in : inputs()) {
            is[i++] = in;
        }
        for (int j = 0; j < n; j++) {
            auto in = is[j];
            // a finished var whose input is already forward-dead cannot be
            // recomputed, so it never contributed backward liveness
            if (in->forward_liveness == 0 && is_finished() && is_var()) continue;
            if (is_finished() && is_stop_grad()) continue;
            liveness_queue.emplace_back(in, &Node::release_backward_liveness);
        }
        LOGvvvv << "Free backward_liveness=0" << this;
        free();
    }
}

void Node::own_backward_liveness() {
    CHECK_EXIST;
    backward_liveness++;
    if (backward_liveness == 1) {
        if (!is_finished() || !is_stop_grad())
            for (auto* in : inputs()) {
                liveness_queue.emplace_back(in, &Node::own_backward_liveness);
            }
    }
}

void Node::own_both_liveness() {
    CHECK_EXIST;
    liveness_queue.emplace_back(this, &Node::own_forward_liveness);
    liveness_queue.emplace_back(this, &Node::own_backward_liveness);
    liveness_queue.emplace_back(this, &Node::own_pending_liveness);
    run_liveness_queue("own_both_liveness");
}

void Node::release_both_liveness() {
    CHECK_EXIST;
    SetupFreeBuffer setup_free_buffer;
    liveness_queue.emplace_back(this, &Node::release_forward_liveness);
    liveness_queue.emplace_back(this, &Node::release_backward_liveness);
    liveness_queue.emplace_back(this, &Node::release_pending_liveness);
    run_liveness_queue("release_both_liveness");
}

void Node::finish_pending_liveness() {
    CHECK_EXIST;
    if (is_finished()) return;
    SetupFreeBuffer setup_free_buffer;
    flags.set(NodeFlags::_finished);
    // p1 no longer holds once we are finished
    if (pending_liveness)
        for (auto* in : inputs()) {
            liveness_queue.emplace_back(in, &Node::release_pending_liveness);
        }
    if (is_var() || is_stop_grad()) {
        int n = inputs().size(), i = 0;
        STACK_ALLOC(Node*, is, n);
        for (auto* in : inputs()) {
            is[i++] = in;
        }
        for (int j = 0; j < n; j++) {
            auto in = is[j];
            if (in->forward_liveness == 0 || is_stop_grad()) {
                liveness_queue.emplace_back(in, &Node::release_backward_liveness);
            }
        }
    }
    run_liveness_queue("finish_pending_liveness");
}

void Node::release_inputs() {
    CHECK_EXIST;
    if (!_inputs.size()) return;
    SetupFreeBuffer setup_free_buffer;
    for (auto in : _inputs) {
        if (!in.node->is_stop_grad() && in.node->forward_liveness)
            liveness_queue.emplace_back(this, &Node::release_forward_liveness);
        in.node->_outputs.erase(in.back);
        if (backward_liveness) {
            liveness_queue.emplace_back(in.node, &Node::release_backward_liveness);
        }
        if (pending_liveness)
            liveness_queue.emplace_back(in.node, &Node::release_pending_liveness);
    }
    _inputs.clear();
    run_liveness_queue("release_inputs");
}

void Node::set_inputs(list<Node*> nodes) {
    CHECK_EXIST;
    LOGvvvv << "Set inputs of" << this << "to" << nodes;
    ASSERT(!is_finished());
    // Take the new liveness before dropping the old edges, so that a node that
    // appears in both the old and the new input list never drops to zero.
    for (Node* node : nodes) {
        if (!node->is_stop_grad() && node->forward_liveness)
            liveness_queue.emplace_back(this, &Node::own_forward_liveness);
        if (backward_liveness) {
            liveness_queue.emplace_back(node, &Node::own_backward_liveness);
        }
        if (pending_liveness)
            liveness_queue.emplace_back(node, &Node::own_pending_liveness);
    }
    run_liveness_queue("set_inputs");
    release_inputs();
    bool is_var = this->is_var();
    auto iter = nodes.begin();
    for (size_t i = 0; i < nodes.size(); i++, iter++) {
        Node* node = *iter;
        _inputs.emplace_back(node);
        // For an op the output index is the argument position; for a var it is
        // the position in the producer's output list.
        node->_outputs.emplace_back(this, is_var ? node->_outputs.size() : i);
        _inputs.back().back = std::prev(node->_outputs.end());
        node->_outputs.back().back = std::prev(_inputs.end());
    }
}

void Node::add_inputs(const vector<Node*>& nodes) {
    CHECK_EXIST;
    LOGvvvv << "add inputs" << nodes << "to" << this;
    ASSERT(!is_finished());
    for (Node* node : nodes) {
        if (!node->is_stop_grad() && node->forward_liveness)
            liveness_queue.emplace_back(this, &Node::own_forward_liveness);
        if (backward_liveness) {
            liveness_queue.emplace_back(node, &Node::own_backward_liveness);
        }
        if (pending_liveness)
            liveness_queue.emplace_back(node, &Node::own_pending_liveness);
    }
    run_liveness_queue("add_inputs");
    bool is_var = this->is_var();
    auto iter = nodes.begin();
    uint n_old_inputs = _inputs.size();
    for (size_t i = 0; i < nodes.size(); i++, iter++) {
        Node* node = *iter;
        _inputs.emplace_back(node);
        node->_outputs.emplace_back(this, is_var ? node->_outputs.size() : i + n_old_inputs);
        _inputs.back().back = std::prev(node->_outputs.end());
        node->_outputs.back().back = std::prev(_inputs.end());
    }
}

void Node::add_inputs(const vector<Var*>& nodes) {
    add_inputs((const vector<Node*>&)nodes);
}

void Node::set_stop_grad() {
    CHECK_EXIST;
    if (is_stop_grad()) return;
    SetupFreeBuffer setup_free_buffer;
    flags.set(NodeFlags::_stop_grad, 1);
    int had_backward_liveness = backward_liveness;
    int n = inputs().size(), i = 0;
    STACK_ALLOC(Node*, is, n);
    for (auto* in : inputs()) {
        is[i++] = in;
    }
    // f3 stops propagating through a stop_grad node
    if (forward_liveness)
        for (Node* out : outputs()) {
            liveness_queue.emplace_back(out, &Node::release_forward_liveness);
        }
    if (had_backward_liveness) {
        for (int j = 0; j < n; j++) {
            auto in = is[j];
            if (in->forward_liveness == 0 && is_var() && is_finished()) {
                continue;
            }
            if (!is_finished()) continue;
            liveness_queue.emplace_back(in, &Node::release_backward_liveness);
        }
    }
    run_liveness_queue("set_stop_grad");
}

std::ostream& operator<<(std::ostream& os, const Node* node) {
    return node->is_var() ? os << (const Var*)node : os << (const Op*)node;
}

} // jittor
