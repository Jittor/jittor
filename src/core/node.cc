// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mem/allocator.h"
#include "core/node.h"
#include "core/op.h"
#include "core/var.h"
#include <algorithm>
#include <cstdlib>
#include <functional>

namespace jittor {

// TEMP DIAGNOSTIC: see Node::erase_output. Cached because the erase path runs
// for every edge of every node the liveness drain frees.
static bool h3_fuse_dump_on() {
    static const bool on = getenv("H3_FUSE_DUMP") != nullptr;
    return on;
}

int free_buffer_depth = 0;
// See graph.cc: check_graph turns this on so that the dangling-node half of
// do_graph_check has something to sweep in a build without NODE_MEMCHECK.
int node_track_lived = 0;
unordered_map<void*, int64> lived_nodes;
unordered_map<int64, Node*> lived_nodes_id;
std::atomic<int64> total_node{0};
NodeLifecycleObserver* node_lifecycle_observer = nullptr;

// Kept alive for the whole process, like graph_mutation_mutex below: it is
// appended to from static destructors. `Node::free` runs from the liveness
// drain, and that drain is reached at exit through the compiled-fused-op
// cache's destructor (`~VarRelayGroup` -> `~VarPtr` -> `release_both_liveness`),
// which runs in whatever order the linker picked relative to a namespace-scope
// vector. ASAN caught the append into the already-destroyed buffer; glibc
// reported the same event later as a "corrupted double-linked list". The
// process is exiting, so there is nothing to reclaim.
vector<Node*>& free_buffer() {
    static auto* buffer = new vector<Node*>();
    return *buffer;
}

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
// Never destroyed, for the same reason as free_buffer above: an exit-time
// static destructor (the compiled-fused-op cache's) reaches
// release_both_liveness, which appends here. A namespace-scope vector would
// already be gone by then. ASAN named this exact write as a heap-use-after-free
// in Node::release_both_liveness <- ~VarRelayGroup; that is the corruption
// glibc surfaced afterwards as "corrupted double-linked list".
static vector<pair<Node*, liveness_op_t>>& liveness_queue =
    *new vector<pair<Node*, liveness_op_t>>();
static size_t liveness_queue_front = 0;

// Leaked on purpose: this is taken from an atexit handler and from static
// destructors, where a function-local static may already be gone.
std::recursive_mutex& graph_mutation_mutex() {
    static std::recursive_mutex* mutex = new std::recursive_mutex();
    return *mutex;
}

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
    // The drain owns the queue -- it clears it on the way out -- so two threads
    // draining at once would empty each other's work and run each other's
    // callbacks on nodes neither of them owns.
    std::lock_guard<std::recursive_mutex> guard(graph_mutation_mutex());
    LOGvvvv << "run liveness queue from" << caller << "size" << liveness_queue.size();
    // A step can throw: the counters assert their own invariants and `free`
    // reaches the allocator. Leaving the queue half-drained would make the
    // *next* drain resume at a stale front index, over entries naming nodes
    // that the unwinding caller has already let go -- a use-after-free
    // attributed to whoever drains next. The drain owns the queue, so it
    // hands it back empty on either path.
    struct Reset {
        ~Reset() {
            liveness_queue.clear();
            liveness_queue_front = 0;
        }
    } reset;
    for (; liveness_queue_front < liveness_queue.size();) {
        // Copy the entry out before calling: the call may append to the
        // queue and reallocate it, invalidating any reference into it.
        Node* node = liveness_queue[liveness_queue_front].first;
        liveness_op_t op = liveness_queue[liveness_queue_front].second;
        liveness_queue_front++;
        LOGvvvv << "liveness" << liveness_op_name(op) << (void*)node;
        (node->*op)();
    }
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
    // Same lock as the drain: this appends to `liveness_queue` and erases this
    // node from its neighbours' edge lists, and those neighbours may belong to
    // another worker's relay.
    std::lock_guard<std::recursive_mutex> guard(graph_mutation_mutex());
    // already scheduled for deletion in this free_buffer round
    if (flags.get(NodeFlags::_queued_for_free)) return;
    // A var that still has an input op and is either alive forward or not yet
    // finished is going to be recomputed or written; it is not garbage.
    if (is_var() && _inputs.size() && (liveness.forward.active() || !is_finished())) {
        return;
    }
    // NOTE: an op with a live or unfinished output reaches this point, and
    // clearing the edges below is what takes its segment out from under a
    // batch the executor is still planning (`FusedOp::update_ops` classifies a
    // segment by walking `op->outputs()`). Returning early here is NOT the fix:
    // `free()` is also the point where this node releases the liveness it holds
    // on its inputs, so skipping it makes the counters underflow later
    // ("backward liveness release without a matching owner", node.h:279).
    // Measured: the early return removed the `outputs().size()` assert over 12
    // loader-race runs and replaced it with that underflow in 3 of them. The
    // fix has to keep the planning thread and this one from overlapping at all
    // -- `graph_mutation_mutex()` -- not keep this node alive.
    flags.set(NodeFlags::_queued_for_free);
    free_buffer().push_back(this);
    for (auto in : _inputs) {
        in.node->erase_output(in.back_index);
        if (liveness.backward.active()) {
            liveness_queue.emplace_back(in.node, &Node::release_backward_liveness);
        }
        if (liveness.pending.active() && !is_finished())
            liveness_queue.emplace_back(in.node, &Node::release_pending_liveness);
    }
    _inputs.clear();
    for (auto out : _outputs) {
        out.node->erase_input(out.back_index);
        if (!is_stop_grad()) {
            if (liveness.forward.active())
                liveness_queue.emplace_back(out.node, &Node::release_forward_liveness);
        }
        // an output var that nothing needs backward has just lost its only
        // producer, so it can go with us
        if (out.node->is_var() && !out.node->liveness.backward.active()) out.node->free();
    }
    _outputs.clear();
    if (is_var()) free_var((Var*)this);
}

void Node::__release() {
    if (is_var())
        Var::number_of_lived_vars--;
    else
        Op::number_of_lived_ops--;
    flags.set(NodeFlags::_released);
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
    bool became_live = liveness.pending.own();
    // p2: an unfinished node with pending liveness keeps its inputs pending
    if (became_live && !is_finished())
        for (auto* in : inputs())
            liveness_queue.emplace_back(in, &Node::own_pending_liveness);
}

void Node::release_pending_liveness() {
    CHECK_EXIST;
    if (flags.get(NodeFlags::_queued_for_free)) return;
    bool became_dead = liveness.pending.release();
    if (became_dead && !is_finished()) {
        for (auto* in : inputs())
            liveness_queue.emplace_back(in, &Node::release_pending_liveness);
    }
    // Nothing is waiting to compute from this var any more. Its memory can go
    // even though the var itself stays alive, unless backward still needs it.
    if (became_dead && is_var()) {
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
    if (flags.get(NodeFlags::_queued_for_free)) return;
    bool became_dead = liveness.forward.release();
    if (became_dead) {
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
        if (liveness.backward.active()) {
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
    bool became_live = liveness.forward.own();
    if (became_live) {
        if (!is_stop_grad())
            for (auto* out : outputs())
                liveness_queue.emplace_back(out, &Node::own_forward_liveness);
    }
}

void Node::release_backward_liveness() {
    CHECK_EXIST;
    if (flags.get(NodeFlags::_queued_for_free)) return;
    bool became_dead = liveness.backward.release();
    if (became_dead) {
        int n = inputs().size(), i = 0;
        STACK_ALLOC(Node*, is, n);
        for (auto* in : inputs()) {
            is[i++] = in;
        }
        for (int j = 0; j < n; j++) {
            auto in = is[j];
            // a finished var whose input is already forward-dead cannot be
            // recomputed, so it never contributed backward liveness
            if (!in->liveness.forward.active() && is_finished() && is_var()) continue;
            if (is_finished() && is_stop_grad()) continue;
            liveness_queue.emplace_back(in, &Node::release_backward_liveness);
        }
        LOGvvvv << "Free backward_liveness=0" << this;
        free();
    }
}

void Node::own_backward_liveness() {
    CHECK_EXIST;
    bool became_live = liveness.backward.own();
    if (became_live) {
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
    if (liveness.pending.active())
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
            if (!in->liveness.forward.active() || is_stop_grad()) {
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
        if (!in.node->is_stop_grad() && in.node->liveness.forward.active())
            liveness_queue.emplace_back(this, &Node::release_forward_liveness);
        in.node->erase_output(in.back_index);
        if (liveness.backward.active()) {
            liveness_queue.emplace_back(in.node, &Node::release_backward_liveness);
        }
        if (liveness.pending.active())
            liveness_queue.emplace_back(in.node, &Node::release_pending_liveness);
    }
    _inputs.clear();
    run_liveness_queue("release_inputs");
}

void Node::erase_input(uint index) {
    ASSERT(index < _inputs.size());
    _inputs.erase(_inputs.begin() + index);
    for (uint i = index; i < _inputs.size(); ++i)
        _inputs[i].reverse().back_index = i;
}

void Node::erase_output(uint index) {
    ASSERT(index < _outputs.size());
    _outputs.erase(_outputs.begin() + index);
    for (uint i = index; i < _outputs.size(); ++i)
        _outputs[i].reverse().back_index = i;
    // TEMP DIAGNOSTIC (H3_FUSE_DUMP): an Op's `_outputs` is the list of Vars it
    // produces, and FusedOp::update_ops() classifies a fused segment by walking
    // exactly that list. Erasing the last entry leaves the op with nothing to
    // classify, which is the "no in-memory output" assert this is hunting. The
    // `addr` matches the one the fused_op dump prints for the same op, so the
    // dump can be attributed to the erase that caused it.
    if (h3_fuse_dump_on() && !is_var() && _outputs.size() == 0) {
        LOGw << "H3FUSE erase_output emptied producer op" << this
             << "tflag" << tflag << "batch_stamp" << batch_stamp
             << "erased_index" << index;
    }
}

void Node::set_inputs(list<Node*> nodes) {
    CHECK_EXIST;
    LOGvvvv << "Set inputs of" << this << "to" << nodes;
    ASSERT(!is_finished());
    // Take the new liveness before dropping the old edges, so that a node that
    // appears in both the old and the new input list never drops to zero.
    for (Node* node : nodes) {
        if (!node->is_stop_grad() && node->liveness.forward.active())
            liveness_queue.emplace_back(this, &Node::own_forward_liveness);
        if (liveness.backward.active()) {
            liveness_queue.emplace_back(node, &Node::own_backward_liveness);
        }
        if (liveness.pending.active())
            liveness_queue.emplace_back(node, &Node::own_pending_liveness);
    }
    run_liveness_queue("set_inputs");
    release_inputs();
    bool is_var = this->is_var();
    auto iter = nodes.begin();
    for (size_t i = 0; i < nodes.size(); i++, iter++) {
        Node* node = *iter;
        uint output_index = node->_outputs.size();
        _inputs.emplace_back(node, output_index);
        // For an op the output index is the argument position; for a var it is
        // the position in the producer's output list.
        node->_outputs.emplace_back(this,
            is_var ? output_index : i, _inputs.size() - 1);
    }
}

void Node::add_inputs(const vector<Node*>& nodes) {
    CHECK_EXIST;
    LOGvvvv << "add inputs" << nodes << "to" << this;
    ASSERT(!is_finished());
    for (Node* node : nodes) {
        if (!node->is_stop_grad() && node->liveness.forward.active())
            liveness_queue.emplace_back(this, &Node::own_forward_liveness);
        if (liveness.backward.active()) {
            liveness_queue.emplace_back(node, &Node::own_backward_liveness);
        }
        if (liveness.pending.active())
            liveness_queue.emplace_back(node, &Node::own_pending_liveness);
    }
    run_liveness_queue("add_inputs");
    bool is_var = this->is_var();
    auto iter = nodes.begin();
    uint n_old_inputs = _inputs.size();
    for (size_t i = 0; i < nodes.size(); i++, iter++) {
        Node* node = *iter;
        uint output_index = node->_outputs.size();
        _inputs.emplace_back(node, output_index);
        node->_outputs.emplace_back(this,
            is_var ? output_index : i + n_old_inputs, _inputs.size() - 1);
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
    int had_backward_liveness = liveness.backward.count();
    int n = inputs().size(), i = 0;
    STACK_ALLOC(Node*, is, n);
    for (auto* in : inputs()) {
        is[i++] = in;
    }
    // f3 stops propagating through a stop_grad node
    if (liveness.forward.active())
        for (Node* out : outputs()) {
            liveness_queue.emplace_back(out, &Node::release_forward_liveness);
        }
    if (had_backward_liveness) {
        for (int j = 0; j < n; j++) {
            auto in = is[j];
            if (!in->liveness.forward.active() && is_var() && is_finished()) {
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
