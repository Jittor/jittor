// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guoye Yang <498731903@qq.com>
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <functional>
#include <queue>
#ifdef HAS_ACCELERATOR
#include "core/event_queue.h"
#endif
#include "runtime/device.h"
#include "runtime/executor_entry.h"
#include "runtime/async_exec.h"
#include "runtime/graph_capture.h"
#include "runtime/submission_pipeline.h"
#include "runtime/backend.h"
#include "runtime/backend_fallback.h"
#include "ops/op_register.h"
#include "core/executor.h"
#include "core/exec_plan.h"
#include "core/exec_runner.h"
#include "core/var.h"
#include "core/op.h"
#include "mem/allocator.h"
#include "core/graph.h"
#include "core/fused_op.h"
#include "core/fuser.h"
#include "runtime/profiler/profiler_guard.h"
#include "runtime/profiler/step_trace.h"
#include "core/parallel_compiler.h"
#include "core/memory_profiler.h"
#include "debug/nan_checker.h"
#include "core/memory_profiler.h"
#include "utils/seh.h"
#include "utils/cache_compile.h"
#include "core/var_holder.h"
#include "mem/swap.h"
#include "bindings/pybind/py_var_tracer.h"
#include "mem/mem_info.h"

namespace jittor {

EXTERN_LIB MemoryProfiler memory_profiler;
DEFINE_FLAG(int, lazy_execution, 1, "Default enabled, if disable, use immediately eager execution rather than lazy execution, This flag makes error message and traceback infomation better. But this flag will raise memory consumption and lower the performance.");
DEFINE_FLAG(int, auto_flush_ops, 128, "Pipeline graph construction with device execution on CUDA. Once this many operators have been created since the executor last ran, launch everything pending -- IF that pending work is also worth at least `auto_flush_bytes` (see there). 0 keeps fully lazy execution. Fusion and dead-code elimination still apply within each launched segment; CPU execution is synchronous and never flushes early.");
DEFINE_FLAG(int64, auto_flush_bytes, 8<<20, "How much pending output a flush must be carrying before it is worth taking. A flush CUTS the step's graph into two batches, and fusion is decided within a batch -- so an elementwise chain that straddles the cut becomes two kernels and the second batch pays planning again. That is only worth paying when there is real device work to overlap with. Counting operators cannot tell the two apart: a batch-1 decode step and a batch-8 sequence-256 step build the SAME ~100 operators, but the first has 200 KB of pending output and the second has tens of MB. Measured: gating on operators alone cost 1.09-1.16x on every host-bound case, while removing the flush entirely cost 1.11x on Resnet50 training and 1.16x on a batched transformer. 0 disables the size gate.");
DECLARE_FLAG(int, profile_memory_enable);
DEFINE_FLAG(int, gopt_disable, 0, "Disable graph optimizer.");
DEFINE_FLAG(int, use_threading, 0, "Allow to use python threading with jittor.");

DEFINE_FLAG(int, exec_called, 0, "exec sync called");
DEFINE_FLAG(int, async_execution, 1, "Run an auto-flushed batch of an inference graph (under no_grad, on CUDA) on a worker thread while Python keeps building the graph; every executor entry, and so every read of a Var's data, waits for it first. See runtime/async_exec.h. 0 executes every batch on the calling thread. Without the worker such a flush also has to carry `auto_flush_bytes` of output to be taken; with it, any auto-flush is.");
DEFINE_FLAG(int, async_flush_ops, 64, "With `async_execution`, an inference graph is also flushed to the worker when a module call returns and this many operators have been created since the last flush (`auto_flush_ops` still applies between). 0 flushes at `auto_flush_ops` only.");
DECLARE_FLAG(bool, no_grad);
DECLARE_FLAG(int, keep_graph);
DECLARE_FLAG(int, profiler_enable);
DECLARE_FLAG(int, trace_py_var);

struct PendingSubmissionScope {
    SubmissionPipeline& pipeline;
    explicit PendingSubmissionScope(SubmissionPipeline& pipeline)
        : pipeline(pipeline) {
        pipeline.flush_active = true;
    }
    ~PendingSubmissionScope() { pipeline.flush_active = false; }
};

// Whether an auto-flush may hand its batch to the worker thread
// (runtime/async_exec.h): then it costs this thread only the planning, and is
// taken without the size gate below and, at module boundaries, sooner.
//
// Not once a second Python thread has used jittor: the worker stands down for
// good (async_exec.cc), and so do the extra flushes it was taken for -- every
// flush releases the GIL inside the executor, and two threads building one
// graph is not something the executor serializes. Nor where the worker does
// not run at all (anywhere but CUDA, see `may_run_async`): there the extra
// flushes would only cut fusion.
static bool deferrable_flush() {
    return async_execution && no_grad && async_flush_ops > 0 && async_exec_single_thread()
        && accelerator_backend_id() == BackendId::Cuda;
}

void Executor::auto_flush() {
#ifdef HAS_ACCELERATOR
    auto& pipeline = runtime_submission_pipeline();
    if (pipeline.flush_active || !runtime_use_cuda() || pipeline.grad_construction_depth
            || !backend_ops(accelerator_backend_id()).execution.supports_auto_flush)
        return;
    const bool deferrable = deferrable_flush();
    vector<Var*> vars;
    int64 pending_bytes = 0;
    for (auto holder : runtime_holder_state().holders()) {
        auto var = holder->var;
        if (var->_outputs.size() || var->is_finished()) continue;
        // The third place a kept graph must not be picked up as a
        // bystander (see `top_weak_sync` and `sync_all`), and the easiest
        // to miss: this fires in the middle of the NEXT call's
        // construction, so the kept graph is re-executed while the caller
        // is still building the work that was going to replace it.
        if (var->flag(VarFlags::_kept)) continue;
        auto op = var->input();
        if (op && op->flag(OpFlags::_must_stay_pending)) continue;
        // Nor a var an elementwise op is still to compute. Launched now it
        // is written out, and a training forward keeps it for the backward,
        // where it would otherwise have fused into its readers and never
        // existed: where the cut fell decided how many of them a ViT-B/16
        // step kept, and its peak moved by 0.33 GB between
        // `auto_flush_ops` 120 and 136. What such a var reads is launched
        // when something that must be written out -- a matmul, a
        // convolution, a reduction -- needs it.
        if (op && (op->type() == OpType::element || op->type() == OpType::broadcast))
            continue;
        vars.push_back(var);
        pending_bytes += var->size;
    }
    // Enough operators, but is there enough work? Cutting the graph costs
    // a fusion boundary and a second planning pass; that only pays for
    // itself when the device has something substantial to chew on
    // meanwhile. Re-arm rather than flush when it does not, so the next
    // decision is another `auto_flush_ops` away instead of every op.
    if (vars.size() && (auto_flush_bytes <= 0 || pending_bytes >= auto_flush_bytes
                        || deferrable)) {
        PendingSubmissionScope scope(pipeline);
        run_sync(vars, false, false, deferrable);
    } else {
        pipeline.last_run_ops = Op::number_of_created_ops;
    }
#endif
}

void Executor::flush_at_module_boundary() {
    // Counted in operators, never timed: the cut points have to repeat
    // identically from step to step (see `SubmissionPipeline::last_run_ops`).
    // At a module's return rather than mid-module, so a block -- a residual
    // unit, a transformer layer -- stays in one batch where it can.
    auto& pipeline = runtime_submission_pipeline();
    if (deferrable_flush()
            && Op::number_of_created_ops - pipeline.last_run_ops >= async_flush_ops)
        auto_flush();
}

void Executor::submit_pending(Var* target, bool force) {
    auto& pipeline = runtime_submission_pipeline();
    if (!target || pipeline.flush_active || target->is_finished()) return;

    if (force || target->num < 0) {
        PendingSubmissionScope scope(pipeline);
        run_sync({target}, false, false);
        CHECK(target->num >= 0) << target << "has an unresolved dynamic shape";
        return;
    }

#ifdef HAS_ACCELERATOR
    if (auto_flush_ops > 0 && Op::number_of_created_ops - pipeline.last_run_ops >= auto_flush_ops)
        auto_flush();
#endif

    if (target->is_finished()
            || (lazy_execution && Op::number_of_lived_ops < 100000)) return;
    auto eager_target = target;
    for (int i=0; i<5; i++) {
        auto op = eager_target->input();
        if (!op) break;
        if (i==0 && op->flag(OpFlags::_must_stay_pending)) return;
        if (op->type() == OpType::other || op->type() == OpType::reduce
                || op->inputs().size() == 0)
            break;
        if (op->type() == OpType::broadcast) return;
        eager_target = op->inputs().front();
    }
    PendingSubmissionScope scope(pipeline);
    run_sync({target}, true);
}

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt) {
    fused_op.ops.clear();
    fused_op.edges.clear();
    // Membership through the segment's own index rather than a traversal
    // mark: a mark writes every node's `tflag`, and on the async worker this
    // runs while the Python thread builds the graph, outside the graph lock.
    for (int i=ll; i<rr; i++) {
        int opid = fuse_ops[i];
        Op* op = ops[opid];
        fused_op.ops.push_back(op);
    }
    LOGvvv << "Prepare fused_op" << fused_op.ops;
    fused_op.update_ops();
    for (Op* op : fused_op.ops) {
        uint fid1 = fused_op.op_index.at(op);
        int iid = 0;
        for (auto& edge : fused_op.snapshot_inputs(op)) {
            Var* v = edge.first;
            // this is a control dependency edge, dont used
            if (edge.second < 0) continue;
            iid++;
            int iop_id;
            int iv_id;
            pair<Op*, int> producer = fused_op.snapshot_producer(v);
            if (producer.first && fused_op.op_index.count(producer.first)) {
                iop_id = fused_op.op_index.at(producer.first);
                iv_id = producer.second;
            } else {
                iv_id = fused_op.var_index.at(v);
                // add iv_id, prevent iv_id jit key overflow
                iop_id = fused_op.ops.size() + iv_id;
            }
            fused_op.edges.emplace_back(iop_id, iv_id, fid1, iid-1);
        }
        // TODO: can we remove this?
        // uint oid = 0;
        // for (Var* v : op->outputs()) {
        //     oid++;
        //     if (v->tflag != tt) {
        //         // this var node not belong to current execution
        //         // this will happend in multiple outputs fuseable op
        //         // v->custom_data = 0 represents this var cannot be fused
        //         v->custom_data = 0;
        //         continue;
        //     }
        //     // for (auto o : v->outputs_with_index()) {
        //     //     Op* op2 = o.op;
        //     //     uint iid = o.index;
        //     //     if (op2->tflag != ntt) continue;
        //     //     uint fid2 = op2->custom_data;
        //     //     fused_op.edges.emplace_back(fid1, oid-1, fid2, iid);
        //     // }
        // }
    }
}

static void top_weak_sync(vector<Var*>& vars) {
    TraversalEpoch epoch("top_weak_sync");
    int64 max_id=0;
    for (auto v : vars) {
        if (v->is_finished()) continue;
        max_id = std::max(v->id, max_id);
        epoch.mark(v);
    }
    while (true) {
        auto& roots = runtime_holder_state();
        auto holder = roots.peek_pending();
        if (!holder) break;
        auto v = holder->var;
        if (v->id > max_id) break;
        roots.consume_pending();
        if (epoch.marked(v)) continue;
        if (v->_outputs.size()) continue;
        if (v->is_finished()) continue;
        // A kept graph is run on purpose, by whoever kept it, and never as a
        // bystander of somebody else's sync. Widening a batch with one costs a
        // full re-execution of a graph nobody asked for -- traced on a
        // step-sized capture, every one of these three paths ran the whole
        // 316-operator graph again behind the caller's back.
        //
        // And re-execution is not always merely wasted. `share_with` lets an
        // operator's output land in a buffer the same graph reads, which is
        // how a replay advances state in place; for such a graph an
        // unrequested run is a silent state change, with the operator counts
        // and every statistic still looking right. So the rule is strict:
        // only the owner runs it.
        //
        // Consumers are unaffected. This only widens the ROOTS, and phase 2
        // still collects any unfinished input of what was actually requested,
        // kept or not.
        if (v->flag(VarFlags::_kept)) continue;
        vars.push_back(v);
    }
}

// One batch, in seven phases. The phase boundaries are marked below with
// `// == phase n ==`; the split between planning and running is between
// phases 5 and 6 (see the contract in executor.h).
//
//   1 setup     pick the batch's allocators, remember the entry device, and
//               (weak sync) widen `vars` with older pending holder Vars.
//   2 collect   BFS from `vars` over unfinished inputs -- plus outputs, so a
//               fetch already queued on a collected Var joins this batch --
//               then run graph optimizers and redo the BFS until no op still
//               asks for one. Splits the result into `ops` and `all_vars` and
//               numbers both under the batch stamp `tt`.
//   3 fuse      `count_fuse` partitions `ops` into fused segments (union-find
//               in `father`) and marks, per var, whether it may stay inside a
//               kernel (`var_fused`).
//   4 order     topological sort over the segments -> `queue`, the order the
//               segments will be executed in. Ties break on `Op::order`.
//   5 order     topological sort inside each segment -> `fuse_ops`, all
//               segments' ops concatenated, with `range` holding the split
//               points. Ops shared by several segments are duplicated into
//               each (`sharegraph`) rather than being cut out.
//   ~ compile   `parallel_compile_all_ops` compiles every segment before any
//               of them runs, so no compilation happens mid-execution. This
//               is the last reader of the batch numbering; the batch stamp is
//               released right after it.
//   6 execute   per segment: load it into `fused_op`, select the device,
//               allocate outputs, migrate across the host/device boundary,
//               launch, then release the liveness the batch held.
//   7 finish    assert the requested Vars are backed, wait for every device
//               the batch launched on if asked, restore the entry device.
static void resolve_dynamic_inputs(Executor& executor, const vector<Var*>& roots) {
    // Native C++ callers can compose operators before any Python conversion.
    // Resolve their internal data-dependent extents at this explicit execution
    // boundary, in dependency order, before fusion or kernel compilation.
    vector<std::pair<Op*, bool>> pending;
    unordered_set<Op*> seen;
    vector<Op*> ordered;
    for (auto* var : roots)
        if (!var->is_finished() && var->input()) pending.emplace_back(var->input(), false);
    while (!pending.empty()) {
        auto entry = pending.back();
        pending.pop_back();
        auto* op = entry.first;
        if (entry.second) {
            ordered.push_back(op);
            continue;
        }
        if (!seen.insert(op).second) continue;
        pending.emplace_back(op, true);
        for (auto* input : op->inputs())
            if (!input->is_finished() && input->input())
                pending.emplace_back(input->input(), false);
    }
    bool changed = false;
    for (auto* op : ordered) {
        for (auto* input : op->inputs()) {
            if (input->num >= 0) continue;
            executor.run_sync({input}, false, false);
            CHECK(input->num >= 0) << "Dynamic input shape was not resolved";
            changed = true;
        }
        if (changed) op->infer_shape();
    }
}

// When each var of the batch has been used for the last time, as a queue
// position: the Runner drops the var's hold after that segment (see
// `ExecPlan::release_after`). A segment is read exactly as `run_exec_plan`
// reads it -- its `fuse_ops` range, plus the root op itself -- and a var is
// counted as used by every op that has it as an input or an output, from the
// edge snapshot the planner recorded. The vars the caller asked for head
// `all_vars` and are never scheduled: phase 7 checks them, and they stay held
// to the end, as does anything no segment names.
static void schedule_hold_release(ExecPlan& plan) {
    const int n = plan.queue.size();
    vector<int> last_use(plan.all_vars.size(), -1);
    // A var's batch index is its position in `all_vars`; one that is not of
    // this batch, or among the vars the caller asked for, is not scheduled.
    auto index_of = [&](Var* v) {
        int i = v->batch_index_in(plan.stamp);
        return i >= plan.start_var_num && i < (int)plan.all_vars.size()
            && plan.all_vars[i] == v ? i : -1;
    };
    auto touch = [&](int op_index, int rid) {
        for (int k = plan.op_inputs_begin[op_index]; k < plan.op_inputs_begin[op_index + 1]; k++) {
            int i = index_of(plan.op_inputs[k].first);
            if (i >= 0) last_use[i] = rid;
        }
        for (int k = plan.op_outputs_begin[op_index]; k < plan.op_outputs_begin[op_index + 1]; k++) {
            int i = index_of(plan.op_outputs[k]);
            if (i >= 0) last_use[i] = rid;
        }
    };
    for (int rid = 0; rid < n; rid++) {
        touch(plan.queue[rid], rid);
        int ll = rid < n - 1 ? plan.range[n - rid - 2] : 0;
        int rr = plan.range[n - rid - 1];
        for (int k = ll; k < rr; k++) touch(plan.fuse_ops[k], rid);
    }
    plan.release_after.assign(n, {});
    for (int i = 0; i < (int)last_use.size(); i++)
        if (last_use[i] >= 0) plan.release_after[last_use[i]].push_back(i);
}

// Everything a batch is, kept together so that its execution can be handed
// to the worker thread (runtime/async_exec.h) as one object.
struct BatchState {
    ExecPlan plan;
    vector<VarPtr> batch_hold;
    FusedOp fused_op;
    vector<Var*> vars;
};

// Whether a planned batch may run on the worker: an auto-flushed batch of an
// inference graph, every operator of it on the accelerator, and nothing
// watching the executor that expects it on this thread -- step traces,
// profilers, recordings, kept graphs, swapping. A dynamic shape is resolved
// before planning (`resolve_dynamic_inputs`), so none is left to sync from
// inside the batch.
//
// Nor without pinned host memory: then `jt.array` stages through the dual
// allocator, whose delayed frees go back in the order their stream callbacks
// fire -- an order that only holds while every free is on one stream.
static bool may_run_async(const ExecPlan& plan) {
    // CUDA only. The worker thread runs the batch with whatever device context
    // it has, which on ACL is none (aclrtGetDevice: 107002, context null), and
    // an ACL host fallback flips the process-wide `use_cuda` under the Python
    // thread's feet; `use_cuda` alone is set on every accelerator.
    if (!async_execution || !no_grad || keep_graph || save_mem || use_threading
            || !runtime_use_cuda() || accelerator_backend_id() != BackendId::Cuda
            || graph_capture_recording || profiler_enable
            || profile_memory_enable || trace_py_var || step_trace_active()
            || !use_pinned_host_memory())
        return false;
    static const OpId numpy_code = get_op_id("numpy_code");
    static const OpId fetch = get_op_id("fetch");
    for (Op* op : plan.ops) {
        if (!op->executes_on_accelerator() || op->is_op(numpy_code) || op->is_op(fetch))
            return false;
        for (Var* out : op->outputs())
            if (out->num < 0) return false;
    }
    return plan.ops.size() > 0 && async_exec_single_thread() && async_exec_placed();
}

void Executor::run_sync(vector<Var*> vars, bool device_sync, bool weak_sync, bool may_defer) {
    // The batch on the worker, if one is in flight, comes first: this one may
    // need what it produces, and only one runs at a time.
    async_exec_wait();
    // == phase 1: setup ==
    // One batch at a time. Until the device waits inside started releasing the
    // GIL, the GIL *was* this exclusion for Python threads; now that another
    // Python thread can run during phase 7, the "not reentrant" contract above
    // has to be a lock. Explicit dynamic-input prerequisite submissions are
    // on this thread and pass straight through; constructors never submit.
    ExecutorEntryScope entry;
    StepTraceBatchScope trace_batch;
    exec_called ++;
    auto& pipeline = runtime_submission_pipeline();
    pipeline.last_run_ops = Op::number_of_created_ops;
    if (weak_sync && !use_threading)
        top_weak_sync(vars);
    resolve_dynamic_inputs(*this, vars);
    this->allocator = get_allocator();
    this->temp_allocator = get_allocator(true);
    // Each op allocates from and launches on the device its outputs live on;
    // the caller gets its own current device back when the run is over, and
    // every device the run touched is waited on rather than just one.
    int entry_device = -1;
    #ifdef HAS_ACCELERATOR
    entry_device = current_device();
    #endif
    // == phases 2-5: graph -> execution plan ==
    auto state = std::make_shared<BatchState>();
    ExecPlan& plan = state->plan;
    build_exec_plan(vars, weak_sync, plan);
    trace_batch.mark(stb_planned);
    // Hold the batch's vars for its duration -- and with them the ops that
    // produce them, since an op's liveness comes from its outputs, so an op
    // whose output var is held cannot be freed either.
    //
    // The batch plans and executes from raw `Op*`/`Var*`. It used to rely on
    // `TraversalEpoch`'s unenforced contract ("a traversal must not have a node
    // it marked destroyed before its epoch ends"), and a concurrent
    // `Node::free()` on another thread -- vLLM loads weights from four of them
    // -- destroyed a node the plan still pointed at.
    //
    // Measured on `H3_FUSE_DUMP=1 probe_loader_race.py threads 4 6` (20 runs
    // each, baseline is ~2 dumps in 6):
    //   no hold             : the "no in-memory output" assert in fused_op.cc
    //   keep the node alive : 3/12 runs die on a liveness underflow (node.h:279)
    //   hold the graph lock : deadlock, killed at the timeout
    //   edge snapshot only  : 0 dumps but 7/20 segfault in the execution-time
    //                         relay walk (`VarRelayManager::get_op_relay_info`)
    //   snapshot + this hold: 20/20 ok, 0 dumps, 0 segfaults, 0 underflows
    // The hold is the missing piece: the plan has to own what it is about to
    // use. The edge snapshot stays because a *live* var can still have its
    // edges released (`release_inputs`, which the shim uses to park tensors),
    // and the planner reads them.
    vector<VarPtr>& batch_hold = state->batch_hold;
    batch_hold.reserve(plan.all_vars.size());
    for (Var* v : plan.all_vars) batch_hold.emplace_back(v);
    // What phase 7 has to discount: this hold is bookkeeping, not a consumer.
    plan.batch_hold_per_var = 1;
    schedule_hold_release(plan);
    plan.batch_hold = &batch_hold;
    ExecutionBackendScope backend_scope(plan.backend);

    // The fusion verdict goes to FusedOp as the vector it already is, instead
    // of being written into bit 0 of every var's custom_data for update_ops()
    // to read back out of the field it packs its own indices into.
    FusedOp& fused_op = state->fused_op;
    fused_op.batch_var_fused = &plan.var_fused;
    fused_op.batch_stamp_wanted = plan.stamp;
    // The batch's own record of the edges it was collected from; see
    // `ExecPlan::op_outputs`.
    fused_op.batch_plan = &plan;

    // compile all ops, prevent compiling during running
    parallel_compile_all_ops(plan.queue, plan.range, fused_op,
                             plan.fuse_ops, plan.ops, plan.stamp);
    trace_batch.mark(stb_compiled);

    // Planning is the last consumer of the batch tflags. Restore any outer
    // traversal before SetupFreeBuffer can destroy nodes from this batch.
    plan.epoch.reset();

    // == phases 6-7: execution plan -> executed kernels ==
    if (may_defer && !device_sync && may_run_async(plan)) {
        // On the worker, while this thread returns to building the graph. The
        // bindings this thread is inside of keep running C++ after it returns,
        // so they take the graph lock first.
        state->vars = move(vars);
        async_exec_hold_active_bindings();
        async_exec_submit([state, entry_device]() mutable {
            #ifdef HAS_ACCELERATOR
            if (entry_device >= 0 && entry_device != current_device())
                set_current_device(entry_device);
            #endif
            run_exec_plan(runtime_executor(), state->plan, state->fused_op, state->vars,
                          false, entry_device);
            state.reset();
        });
        pipeline.last_run_ops = Op::number_of_created_ops;
        return;
    }
    run_exec_plan(*this, plan, fused_op, vars, device_sync, entry_device);

    pipeline.last_run_ops = Op::number_of_created_ops;
}

// Allocations handed to foreign libraries (cupy, cutt) through the hooks
// below. This used to be two parallel maps read with operator[], so an
// unknown pointer silently inserted a zero size and a zero allocation and then
// released *that* -- and nothing was ever erased, so the maps only grew.
//
// The allocator is recorded too: `runtime_executor().allocator` is now the pool of the
// device the executor is on, so it is not the same object at free time as it
// was at alloc time. Freeing into the wrong pool hands it an id it never
// issued, which is a lost block at best and someone else's block at worst.
struct ForeignCudaAllocation { size_t size, allocation; Allocator* allocator; };
static unordered_map<void*, ForeignCudaAllocation> foreign_cuda_allocations;

extern "C" void* jittor_cuda_malloc(void*, size_t size, int device_id) {
    size_t allocation;
    auto* allocator = runtime_executor().allocator;
    void* ptr=allocator->alloc(size, allocation);
    if (ptr) foreign_cuda_allocations[ptr] = {size, allocation, allocator};
    return ptr;
}

extern "C" void jittor_cuda_free(void*, void* ptr, int device_id) {
    if (!ptr) return;
    auto iter = foreign_cuda_allocations.find(ptr);
    CHECK(iter != foreign_cuda_allocations.end())
        << "jittor_cuda_free: pointer was not allocated by jittor_cuda_malloc" << ptr;
    auto info = iter->second;
    foreign_cuda_allocations.erase(iter);
    info.allocator->free(ptr, info.size, info.allocation);
}

extern "C" void* get_jittor_cuda_malloc() {
    return (void*)jittor_cuda_malloc;
}

extern "C" void* get_jittor_cuda_free() {
    return (void*)jittor_cuda_free;
}
    
} // jittor
