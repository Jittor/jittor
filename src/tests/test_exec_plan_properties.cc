// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
//
// Property tests for the Planner's output (10.18).
//
// Until 3.01 there was nothing here to test: `Executor::run_sync` was 579 lines
// with no intermediate product, so the only way to ask "did it order the ops
// correctly" was to run a kernel and look at the number that came out. That
// makes ordering bugs indistinguishable from arithmetic bugs, and it needs a
// device.
//
// `build_exec_plan` now returns `ExecPlan`, a value made almost entirely of
// indices. So the questions below can be asked *of the plan*, on a synthetic
// graph, with no allocation, no kernel and no device -- which is also why they
// are cheap enough to sit in the CPU gate (`tests/compiler/test_jit_tests.py`
// bridges every JIT_TEST into pytest).
//
// These are properties, not expected values: each one is re-checked on every
// graph shape below rather than being an assertion about one recorded plan.
// A recorded plan would freeze the fusion heuristic -- `count_fuse` is allowed
// to change its mind about what fuses, and these tests must not turn a
// heuristic change into a red gate. What may *not* change is that the plan is
// internally consistent and executable in the order it states.

#include <algorithm>
#include <set>
#include <map>

#include "core/exec_plan.h"
#include "core/op.h"
#include "core/var.h"
#include "core/var_holder.h"

namespace jittor {

// A synthetic op: declared type and arity, fixed output shape, no kernel.
//
// Real ops are not usable here. Building one goes through the op registry and
// (for anything fusable) the JIT compiler, so a test written with `binary` and
// `reduce` would be measuring the compiler, need a cache, and cost seconds.
// The Planner only ever asks an op for its `type()`, its edges and its
// `order()`, so those are what this provides.
struct PlanProbeOp final : Op {
    PlanProbeOp(NanoVector shape, OpType type) {
        set_flag(OpFlags::_cpu);
        set_flag(OpFlags::_cuda);
        // Without this, Op::init() sets _needed_by_backward on every edge and
        // drags the liveness machinery into a test about ordering.
        set_flag(OpFlags::_manual_set_vnbb);
        set_type(type);
        create_output(shape, ns_float32);
    }

    const char* name() const override { return "plan_probe"; }
    // The shape is decided at construction; there is nothing to infer, and the
    // default would re-derive it from inputs this op does not interpret.
    void infer_shape() override {}
};

static VarPtr make_probe(const vector<Var*>& inputs, NanoVector shape,
                         OpType type) {
    auto* op = new PlanProbeOp(shape, type);
    op->outputs_holder[0]->set_inputs({op});
    VarPtr output(move(op->outputs_holder[0]));
    list<Node*> nodes;
    for (Var* input : inputs) nodes.push_back(input);
    op->set_inputs(nodes);
    op->init();
    return output;
}

// ---------------------------------------------------------------------------
// The properties
// ---------------------------------------------------------------------------

// Every index in the plan addresses something. `ExecPlan`'s whole point is
// that it is indices rather than pointers, which is only safe if the indices
// are in range -- an out-of-range one is a silent read of the wrong op, not a
// crash, because `ops` is a vector.
static void check_indices_are_in_range(const ExecPlan& plan) {
    int op_num = plan.ops.size();
    CHECKop(plan.op_num,==,op_num);
    CHECKop((int)plan.var_fused.size(),==,(int)plan.all_vars.size());
    CHECKop(plan.start_var_num,<=,(int)plan.all_vars.size());
    CHECKop(plan.start_var_num,>=,0);

    for (int root : plan.queue) {
        CHECKop(root,>=,0);
        CHECKop(root,<,op_num);
    }
    for (int index : plan.fuse_ops) {
        CHECKop(index,>=,0);
        CHECKop(index,<,op_num);
    }
    for (int value : plan.var_fused) {
        CHECKop(value,>=,0);
        CHECKop(value,<=,3);
    }
    // range is indexed from the end of queue, one entry per segment, and its
    // last entry is the end of fuse_ops. Anything else means a segment's ops
    // cannot be located.
    CHECKop(plan.range.size(),==,plan.queue.size());
    if (plan.range.size()) {
        CHECKop(plan.range.back(),==,(int)plan.fuse_ops.size());
        int previous = 0;
        for (uint rid=0; rid<plan.range.size(); rid++) {
            // Non-decreasing, and no empty segment: a segment with no ops
            // would still be executed, compiled and launched.
            CHECKop(plan.range[rid],>,previous);
            previous = plan.range[rid];
        }
    }
}

// The batch numbering is what every index above is relative to. If a node's
// stamped index disagrees with its position, the plan and the graph are
// describing different batches.
static void check_batch_numbering(const ExecPlan& plan) {
    for (uint i=0; i<plan.ops.size(); i++)
        CHECKop(plan.ops[i]->batch_index_at(plan.stamp),==,(int)i);
    for (uint i=0; i<plan.all_vars.size(); i++)
        CHECKop(plan.all_vars[i]->batch_index_at(plan.stamp),==,(int)i);
    // Ops and vars are the only two kinds, so the batch is exactly their sum.
    for (Op* op : plan.ops) CHECK(!op->is_var());
    for (Var* var : plan.all_vars) CHECK(var->is_var());
}

// Every op of the batch is executed. `fuse_ops` may hold an op more than once
// -- an op feeding several segments is duplicated into each rather than cut
// out -- so the property is "at least once", not "exactly once". Dropping an
// op would mean a var is never written; the Runner would then launch a kernel
// over uninitialised memory, which is not a crash and not a wrong answer that
// any single test would recognise.
static void check_every_op_is_scheduled(const ExecPlan& plan) {
    std::set<int> scheduled(plan.fuse_ops.begin(), plan.fuse_ops.end());
    CHECKop(scheduled.size(),==,plan.ops.size());
    for (uint i=0; i<plan.ops.size(); i++)
        CHECK(scheduled.count(i)) << "op" << i << "is in no segment";
}

// Which segment a position in `fuse_ops` belongs to, and which positions a
// segment owns. `range` is indexed from the end of `queue`, so this is the one
// piece of index arithmetic the properties below share.
static void for_each_segment(const ExecPlan& plan,
        const std::function<void(int, int, int)>& func) {
    for (uint rid=0; rid<plan.range.size(); rid++) {
        int begin = rid ? plan.range[rid-1] : 0;
        int end = plan.range[rid];
        int root = plan.queue[plan.queue.size()-1-rid];
        func(root, begin, end);
    }
}

// Segment order is a topological order of the segment graph: whenever an op in
// segment A writes a var an op in segment B reads, A is executed first.
//
// This is the property the priority queue in phase 4 is there to produce.
// Getting it wrong reads a var before it is written -- again, uninitialised
// memory rather than a diagnosable failure.
static void check_segment_order_is_topological(const ExecPlan& plan) {
    // Position of each segment in the execution order, keyed by its root.
    std::map<int,int> position;
    for (uint rid=0; rid<plan.queue.size(); rid++)
        position[plan.queue[rid]] = rid;
    // Which segment each op belongs to. Taken from `fuse_ops`, not from the
    // union-find (which the plan does not carry), so an op duplicated into
    // several segments has several owners and is skipped: a duplicated op is
    // recomputed inside each consumer, so it imposes no order between them.
    std::map<int,std::set<int>> owners;
    for_each_segment(plan, [&](int root, int begin, int end) {
        for (int i=begin; i<end; i++)
            owners[plan.fuse_ops[i]].insert(root);
    });

    for (uint i=0; i<plan.ops.size(); i++) {
        if (owners[i].size() != 1) continue;
        int producer_segment = *owners[i].begin();
        for (Var* var : plan.ops[i]->outputs()) {
            if (var->tflag != plan.stamp) continue;
            for (Op* consumer : var->outputs()) {
                if (consumer->tflag != plan.stamp) continue;
                int consumer_index = consumer->batch_index_at(plan.stamp);
                if (owners[consumer_index].size() != 1) continue;
                int consumer_segment = *owners[consumer_index].begin();
                if (consumer_segment == producer_segment) continue;
                CHECKop(position.at(producer_segment),<,
                        position.at(consumer_segment));
            }
        }
    }
}

// Inside one segment the ops are in a topological order of that segment's own
// dependencies. Phase 5 asserts its queue reached every op; what it does not
// check is that the order it produced actually respects the edges.
static void check_intra_segment_order_is_topological(const ExecPlan& plan) {
    for_each_segment(plan, [&](int root, int begin, int end) {
        // Position of each op within this segment's slice.
        std::map<int,int> position;
        for (int i=begin; i<end; i++)
            position[plan.fuse_ops[i]] = i;
        for (int i=begin; i<end; i++) {
            Op* op = plan.ops[plan.fuse_ops[i]];
            for (Var* var : op->outputs()) {
                if (var->tflag != plan.stamp) continue;
                for (Op* consumer : var->outputs()) {
                    if (consumer->tflag != plan.stamp) continue;
                    int consumer_index = consumer->batch_index_at(plan.stamp);
                    auto found = position.find(consumer_index);
                    if (found == position.end()) continue;
                    CHECKop(i,<,found->second)
                        << "segment" << root << "reads before it writes";
                }
            }
        }
    });
}

// A var of the batch is written by exactly one op of the batch. This is a
// property of the graph the Planner is handed, and the Planner relies on it
// (`count_fuse` dereferences `var->input()` for every non-boundary var), so a
// batch that breaks it is a batch the Planner may not be given.
static void check_each_var_has_one_producer(const ExecPlan& plan) {
    std::map<Var*,int> producers;
    for (Var* var : plan.all_vars) producers[var] = 0;
    for (Op* op : plan.ops)
        for (Var* var : op->outputs())
            if (var->tflag == plan.stamp)
                producers[var]++;
    for (Var* var : plan.all_vars) {
        CHECKop(producers[var],<=,1);
        // A var with no producer in the batch is a boundary, and boundaries
        // are exactly the vars that already exist in memory.
        if (producers[var] == 0)
            CHECKop(plan.var_fused[var->batch_index_at(plan.stamp)],==,1);
    }
}

static void check_all_properties(ExecPlan& plan) {
    check_indices_are_in_range(plan);
    check_batch_numbering(plan);
    check_every_op_is_scheduled(plan);
    check_segment_order_is_topological(plan);
    check_intra_segment_order_is_topological(plan);
    check_each_var_has_one_producer(plan);
}

// Plan `roots` and check every property. Kept as one call so that a shape
// added below cannot accidentally check fewer things than the others.
static void plan_and_check(vector<VarPtr>& roots) {
    vector<Var*> vars;
    for (auto& root : roots) vars.push_back(root.ptr);
    ExecPlan plan;
    build_exec_plan(vars, false, plan);
    // The Runner releases the epoch before executing; a property test does not
    // execute, so it releases it here -- `batch_index_at` is only answerable
    // while the epoch is alive, and every property above reads it.
    check_all_properties(plan);
    plan.epoch.reset();
}

// ---------------------------------------------------------------------------
// The graph shapes
// ---------------------------------------------------------------------------

JIT_TEST(exec_plan_properties_hold_on_a_chain) {
    // source -> element -> element -> element
    NanoVector shape{64};
    auto a = make_probe({}, shape, OpType::element);
    auto b = make_probe({a.ptr}, shape, OpType::element);
    auto c = make_probe({b.ptr}, shape, OpType::element);
    vector<VarPtr> roots;
    roots.emplace_back(move(c));
    plan_and_check(roots);
}

JIT_TEST(exec_plan_properties_hold_on_a_diamond) {
    // One producer read by two ops whose results are combined again: the shape
    // that makes intra-segment order observable at all.
    NanoVector shape{64};
    auto a = make_probe({}, shape, OpType::element);
    auto left = make_probe({a.ptr}, shape, OpType::element);
    auto right = make_probe({a.ptr}, shape, OpType::element);
    auto join = make_probe({left.ptr, right.ptr}, shape, OpType::element);
    vector<VarPtr> roots;
    roots.emplace_back(move(join));
    plan_and_check(roots);
}

JIT_TEST(exec_plan_properties_hold_across_a_reduce_boundary) {
    // A reduce cannot fuse into its consumer, so this shape is guaranteed to
    // produce more than one segment -- which is what makes the segment-order
    // property non-trivial. Asserting the *count* instead would freeze the
    // heuristic; asserting the order does not.
    NanoVector wide{64}, narrow{1};
    auto a = make_probe({}, wide, OpType::element);
    auto reduced = make_probe({a.ptr}, narrow, OpType::reduce);
    auto broadcast = make_probe({reduced.ptr}, wide, OpType::broadcast);
    auto out = make_probe({broadcast.ptr, a.ptr}, wide, OpType::element);
    vector<VarPtr> roots;
    roots.emplace_back(move(out));
    plan_and_check(roots);
}

JIT_TEST(exec_plan_properties_hold_on_several_independent_roots) {
    // Two disconnected graphs in one batch. Independent segments are ordered by
    // `Op::order()`, a path with no dependency edges to fall back on.
    NanoVector shape{64};
    auto a = make_probe({}, shape, OpType::element);
    auto b = make_probe({a.ptr}, shape, OpType::element);
    auto c = make_probe({}, shape, OpType::element);
    auto d = make_probe({c.ptr}, shape, OpType::element);
    vector<VarPtr> roots;
    roots.emplace_back(move(b));
    roots.emplace_back(move(d));
    plan_and_check(roots);
}

JIT_TEST(exec_plan_properties_hold_when_one_var_feeds_many_consumers) {
    // A var read by many ops is the case `count_fuse` treats specially (16
    // consumers is one of its thresholds), and the case where an op gets
    // duplicated into several segments -- so it is the case that decides
    // whether "every op is scheduled" must be "at least once" or "exactly
    // once".
    NanoVector shape{64};
    auto a = make_probe({}, shape, OpType::element);
    vector<VarPtr> consumers;
    for (int i=0; i<20; i++)
        consumers.emplace_back(make_probe({a.ptr}, shape, OpType::element));
    vector<VarPtr> roots;
    for (auto& consumer : consumers) roots.emplace_back(move(consumer));
    plan_and_check(roots);
}

JIT_TEST(exec_plan_properties_hold_on_a_wide_deep_graph) {
    // Enough ops that the union-find has real work and the two topological
    // sorts are not trivially ordered by construction.
    NanoVector shape{64};
    vector<VarPtr> layer;
    for (int i=0; i<8; i++)
        layer.emplace_back(make_probe({}, shape, OpType::element));
    for (int depth=0; depth<6; depth++) {
        vector<VarPtr> next;
        for (int i=0; i<8; i++) {
            vector<Var*> inputs{layer[i].ptr, layer[(i+1)%8].ptr};
            next.emplace_back(make_probe(inputs, shape, OpType::element));
        }
        layer = move(next);
    }
    plan_and_check(layer);
}

// The plan is a *value*: building it must not execute anything. That is the
// contract 3.01 introduced ("it allocates nothing on the device, runs no
// kernel, and moves no memory"), and it is the reason these tests can run
// without a device at all. Stated as a test so that a future planner that
// quietly launches something is caught here rather than by a CUDA gate.
JIT_TEST(exec_plan_building_executes_nothing) {
    NanoVector shape{64};
    auto a = make_probe({}, shape, OpType::element);
    auto b = make_probe({a.ptr}, shape, OpType::element);

    int64 ops_before = Op::number_of_lived_ops;
    CHECK(!a->is_finished());
    CHECK(!b->is_finished());
    CHECK(a->mem_ptr == nullptr);
    CHECK(b->mem_ptr == nullptr);

    vector<Var*> vars{b.ptr};
    ExecPlan plan;
    build_exec_plan(vars, false, plan);
    check_all_properties(plan);

    // No var of the batch was allocated, finished or freed, and no op was
    // created or destroyed by planning.
    for (Var* var : plan.all_vars) {
        CHECK(var->mem_ptr == nullptr);
        CHECK(!var->is_finished());
    }
    CHECKop(Op::number_of_lived_ops,==,ops_before);
    plan.epoch.reset();
}

// Planning the same graph twice gives the same plan. The stamp differs (it is a
// new traversal), so this compares everything the stamp is not: an unstable
// plan would make every performance measurement and every A/B on this path
// unreproducible, which is how 3.01's plan-cache question got asked in the
// first place.
JIT_TEST(exec_plan_is_deterministic_for_the_same_graph) {
    NanoVector wide{64}, narrow{1};
    auto a = make_probe({}, wide, OpType::element);
    auto reduced = make_probe({a.ptr}, narrow, OpType::reduce);

    auto broadcast = make_probe({reduced.ptr}, wide, OpType::broadcast);
    auto out = make_probe({broadcast.ptr, a.ptr}, wide, OpType::element);

    vector<Var*> vars{out.ptr};
    ExecPlan first;
    build_exec_plan(vars, false, first);
    check_all_properties(first);
    // Positions, not pointers: `ops` is ordered by the BFS, so comparing the
    // two plans' index vectors compares the decisions, not the addresses.
    auto first_queue = first.queue;
    auto first_fuse_ops = first.fuse_ops;
    auto first_range = first.range;
    auto first_var_fused = first.var_fused;
    auto first_op_num = first.op_num;
    auto first_start_var_num = first.start_var_num;
    first.epoch.reset();

    ExecPlan second;
    build_exec_plan(vars, false, second);
    check_all_properties(second);
    CHECKop(second.op_num,==,first_op_num);
    CHECKop(second.start_var_num,==,first_start_var_num);
    CHECK(second.queue == first_queue);
    CHECK(second.fuse_ops == first_fuse_ops);
    CHECK(second.range == first_range);
    CHECK(second.var_fused == first_var_fused);
    second.epoch.reset();
}

// `start_var_num` is a claim about the *head* of `all_vars`: the vars the
// caller asked for come first, and `count_fuse` treats exactly that prefix as
// boundaries that must stay in memory. Both halves are load-bearing and
// neither is checked anywhere else.
JIT_TEST(exec_plan_requested_vars_head_all_vars) {
    NanoVector shape{64};
    auto a = make_probe({}, shape, OpType::element);
    auto b = make_probe({a.ptr}, shape, OpType::element);
    auto c = make_probe({b.ptr}, shape, OpType::element);

    vector<Var*> vars{c.ptr, b.ptr};
    ExecPlan plan;
    build_exec_plan(vars, false, plan);
    check_all_properties(plan);

    CHECKop(plan.start_var_num,==,2);
    // The requested vars are the prefix, in the order they were asked for.
    CHECKop(plan.all_vars[0],==,c.ptr);
    CHECKop(plan.all_vars[1],==,b.ptr);
    // And the prefix is what count_fuse was told to keep in memory.
    for (int i=0; i<plan.start_var_num; i++)
        CHECKop(plan.var_fused[i],==,1);
    plan.epoch.reset();
}

} // namespace jittor
