// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <cstring>
#include <unordered_set>
#ifdef HAS_ACCELERATOR
#include "core/event_queue.h"
#endif
#include "runtime/device.h"
#include "runtime/backend_streams.h"
#include "runtime/executor_entry.h"
#include "runtime/backend.h"
#include "runtime/graph_capture.h"
#include "runtime/backend_fallback.h"
#include "runtime/launch_diagnostics.h"
#include "ops/op_register.h"
#include "ops/composite/array_op.h"
#include "core/exec_runner.h"
#include "core/executor.h"
#include "core/var.h"
#include "core/op.h"
#include "mem/allocator.h"
#include "core/fused_op.h"
#include "runtime/profiler/profiler_guard.h"
#include "runtime/profiler/step_trace.h"
#include "core/memory_profiler.h"
#include "debug/nan_checker.h"
#include "utils/cache_compile.h"
#include "core/var_holder.h"
#include "mem/swap.h"
#include "bindings/pybind/py_var_tracer.h"
#include "mem/mem_info.h"

namespace jittor {

EXTERN_LIB MemoryProfiler memory_profiler;
DECLARE_FLAG(int, profile_memory_enable);

// from fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher_to_free;
// from cuda_managed_allocator
#ifdef HAS_ACCELERATOR
DECLARE_FLAG(int, use_cuda_managed_allocator);
#endif


// Every input a launch is about to read has to have memory. Reported here,
// where the var and its op are still in hand, rather than as a null
// dereference inside the generated kernel.
static inline void check_input_is_backed(Var* v, Op* op) {
    if (PREDICT_BRANCH_NOT_TAKEN(!v->mem_ptr && v->size != 0
                                 && !v->flag(VarFlags::_is_swapped)))
        LOGf << "input" << v << "of" << op->name()
             << "has no memory at launch time. Its storage was released while a"
             << "graph that still reads it was retained; see KI-EXEC-006.";
}


static inline void propergate_needed_flags(FusedOp& fused_op) {
    auto& ops = fused_op.ops;
    for (int i=ops.size()-1; i>=0; i--) {
        bool has_need = 0;
        auto op = ops[i];
        for (auto o : op->outputs())
            if (o->flag(VarFlags::_needed_by_backward) &&
                !fused_op.var_stays_in_memory((Node*)o)) {
                has_need = 1;
            }
        if (has_need)
            for (auto i : op->inputs()) {
                i->set_flag(VarFlags::_needed_by_backward);
            }
    }
}


// A failed operator, reported so that the reason comes first.
//
// The previous order was: JIT source path (a ~260-character cache filename),
// op type, inputs, outputs, backtrace, and only then `[Reason]`. On a plain
// out-of-range index the sentence the user needed was the eighth line of
// twelve, and it arrived carrying a second `[f <timestamp> <thread>
// <file:line>]` prefix inside the outer one -- two timestamps and two source
// locations for one error. Everything below the reason is still here; it is
// just below it.
static string describe_vars(const vector<Var*>& vars) {
    string out;
    for (auto v : vars) {
        if (out.size()) out += ", ";
        std::stringstream ss;
        ss << v->dtype() << v->shape;
        // `Var::name` is a `cstr`, not a `std::string`; stream it rather than
        // concatenating.
        if (v->name.size()) ss << ' ' << v->name;
        out += ss.str();
    }
    return out.size() ? out : "(none)";
}

void check_op_async_error(Op* op, bool is_fused_op, const std::exception& e,
                          jittor::Log& logf, const string& jit_src_path) {
    vector<Stack> stack;
    string op_name;
    vector<Var*> ins, outs;
    if (is_fused_op) {
        FusedOp& fused_op = *((FusedOp*)op);
        op_name = "fused_op(";
        for (auto& sub : fused_op.ops) {
            if (op_name.back() != '(') op_name += ", ";
            op_name += sub->name_ex();
        }
        op_name += ")";
        for (auto& vi : fused_op.vars) {
            if (vi.type == 0) ins.push_back(vi.var);
            if (vi.type == 2) outs.push_back(vi.var);
        }
    } else {
        op_name = op->name_ex();
        for (auto v : op->inputs()) ins.push_back(v);
        for (auto v : op->outputs()) outs.push_back(v);
    }
    if (outs.size()) stack = get_node_trace(outs.back());

    logf >> "\n" >> message_without_log_prefix(string(e.what())) >> "\n";

    // The Python frames are the second thing a reader wants and the only part
    // that names their own code.
    if (stack.size()) {
        logf >> "\n[Async Backtrace]: ---";
        for (auto& s : stack) {
            logf << "\n    " << s.file_path >> ":" >> s.lineno;
            if (s.module_type.size()) logf << '<' >> s.module_type >> '>';
            if (s.module_name.size() && s.module_name.find(":") == string::npos)
                logf << '[' >> s.module_name >> ']';
        }
        logf >> "\n";
    } else {
        // No instruction here. `jit_utils.cc` prints one banner saying how to
        // get the backtrace; saying it twice was how this message used to end.
        logf >> "\n[Async Backtrace]: unavailable\n";
    }

    logf >> "\nop: " >> op_name;
    logf >> "\n  in:  " >> describe_vars(ins);
    logf >> "\n  out: " >> describe_vars(outs);
    // Last, because it is a ~260-character cache filename and it is where to
    // look next rather than what happened.
    if (jit_src_path.size()) logf >> "\n  jit source: " >> jit_src_path;
    logf >> "\n";
    jittor::LogFatalVoidify() && logf;
}


// Where an op's outputs are allocated. Every op places them on its own device
// except the host copy behind `x.cpu()`, whose output is marked host-resident:
// allocating that on the device first cost the tensor's size in device memory
// for a transfer that exists to release it.
static inline Allocator* output_allocator(Var* v, Allocator* op_allocator) {
    if (v->flag(VarFlags::_host_resident)) return cpu_allocator;
    return op_allocator;
}

#ifdef HAS_ACCELERATOR
// The device an op runs on: where its outputs are placed. Op::propagate_device
// has already made the outputs agree with the inputs, so either end answers;
// outputs first because device_copy is the one op where they differ.
// A Var's memory belongs on the Var's own device, which is the op's device
// for every op except device_copy -- the one op whose input deliberately
// lives on another card. Migrating such an input into the op's pool moves the
// bytes to the wrong device while Var::device_id still says otherwise, and
// the next kernel that reads it faults with an illegal address.
static inline Allocator* var_allocator(Var* v, Allocator* op_allocator) {
    if (v->placement.explicit_backend)
        return get_allocator(v->placement.device, false);
    if (v->device_id >= 0 && v->device_id != op_allocator->device())
        return get_allocator(v->device_id, false);
    return op_allocator;
}

static inline int op_target_device(Op* op) {
    for (Var* v : op->outputs())
        if (v->device_id >= 0) return v->device_id;
    for (Var* v : op->inputs())
        if (v->device_id >= 0) return v->device_id;
    return current_device();
}
#endif


// Opt-in: leave the batch's nodes unfinished so the same graph can be run
// again. Normally the Runner ends each op with `finish_pending_liveness()`,
// which sets `_finished` and releases the pending liveness of that op's inputs
// -- that release is how intermediates are reclaimed, and it is also what makes
// a graph single-use.
//
// Undoing it afterwards is NOT an option: `finish_pending_liveness` opens with
// `if (is_finished()) return;`, so clearing the flag and running again would
// release the same inputs a second time -- a refcount underflow and then a
// use-after-free. The only safe form is to never finish in the first place,
// which is what this does. The caller is then responsible for holding the
// graph's vars; nothing is reclaimed while it is set.
DEFINE_FLAG(int, keep_graph, 0, "Leave a batch's nodes unfinished so the same graph can be executed again. The caller must hold the graph. Every leaf the graph reads must already be materialized before the graph is built, because a re-run re-executes whatever is still pending -- including a leaf's own producer, whose host staging is gone by then. 1 keeps every node's memory as well, so each re-run writes the same buffers (what a recorded device graph needs). 2 keeps the nodes but returns an intermediate's memory once the batch has no further use for it, as a normal batch does; the next run allocates it again. 0 is the normal single-use behaviour.");

// Read from python (`jittor/_runtime/graph_replay.py`), not from here: it is
// the policy switch for re-running a repeated inference graph instead of
// rebuilding it. It lives beside `keep_graph` because that is the mechanism it
// drives.
DEFINE_FLAG(int, auto_graph_replay, 1, "Re-run a repeated inference graph instead of rebuilding it every call. Applies only under no_grad, only to a call whose inputs are Vars whose total size is under `auto_graph_replay_bytes`, only after the same shapes have been seen twice in a row, and recorded as a device graph only up to `auto_graph_replay_retain_bytes`; anything else, and anything the capture cannot serve, runs normally. 0 disables it.");
DEFINE_FLAG(int64, auto_graph_replay_bytes, 64<<10, "How large the inputs of a call may be before it is left to run normally. Replay removes graph construction, which costs the same whatever the tensors weigh -- so it wins exactly when the device work per operator is small, and loses when the step was never host-bound to begin with. Input size is the cheap proxy for that. It does not bound what a capture retains -- see `auto_graph_replay_retain_bytes`. Measured on the comparison shapes: at 1 MB the policy engaged for a 256x1024 mlp forward (about a dozen operators, nothing to rebuild) and made it 0.39 -> 0.93 ms, and for a 128-token prefill, 2.88 -> 3.26. At 64 KB it engages for the decode steps, where it is 1.97 -> 0.86 against PyTorch, and leaves the rest alone.");
DEFINE_FLAG(int64, auto_graph_replay_retain_bytes, 64<<20, "The largest graph the automatic policy records as a device graph. A capture replays through the executor, which frees intermediates as it goes with keep_graph=2, while a recording re-issues fixed pointers and so keeps every buffer for as long as the capture lives, and the size of a call's inputs says nothing about that: an SD1.5 VAE decode takes a 32 KB latent and allocates 6.2 GB, where it peaks at 0.5 GB. Measured as everything the pools hand out during the eager call that precedes a capture. Above it the capture still replays, through the executor. 0 removes the bound. An explicit jt.graph_replay is not bounded.");

// `keep_graph == 2`: a kept var's memory goes once the batch has made its last
// use of it; the node stays, unfinished, and the next run of the graph
// allocates it again. Holding every buffer instead made a kept graph cost the
// sum of its intermediates rather than their peak -- 6.2 GB for an SD1.5 VAE
// decode that peaks at 0.5 GB run normally.
//
// A storage view (reshape) does not run: its output is its input's buffer, so
// a later run must alias that buffer again rather than get a fresh one, and
// the share request its op made at construction -- consumed by the first
// allocation -- is put back. The same goes for any other var that aliases one
// of its producer's inputs: an in-place update writes into its input again on
// the next run, and a pass-through such as `tape` (GroupNorm's backward runs
// through one) launches nothing at all, so in a fresh buffer of its own it
// would answer with uninitialized memory.
//
// Whatever else shares the allocation decides whether it may go. A var the
// batch does not release -- its result, typically -- is not re-aliased, so it
// would go on pointing at the old buffer while the graph recomputes into a new
// one, and answer with the first run's bytes; and a view keeps its whole chain
// of bases alive with it, because a base freed under a live view leaves the
// view on the old buffer just the same. Every auto-replayed CUDA call ending
// in `jt.stack` (a code op, reshaped) did this, and so did `nn.RNN`'s hidden
// state (a reshape of a clone of the last step), and so did a loss computed by
// the full-reduce fast path, a `tape` over a reshape. `pinned` is that set,
// followed through every alias (`aliased_input`), not only views. Beyond
// it, a var that aliases nothing of its own may go only if all that shares it
// are views: `setitem_gopt` has a concat operand computed straight into its
// slice of the destination and turns the setitem into a no-op, which would
// never copy a recomputed operand.
// Whether every op of this segment is a constant (`array`) whose output
// already holds it: an earlier run of this kept graph wrote it there, either
// by handing over the data (ArrayOp::run) or, for a scalar fused into a
// kernel, by running that kernel, and release_kept_storage never frees it.
static bool constants_already_placed(Op* op, bool is_fused_op, FusedOp& fused_op) {
    auto placed = [](Op* o) {
        if (!o->is_op(op_ids::array())) return false;
        auto* array = static_cast<ArrayOp*>(o);
        return array->output->mem_ptr || !array->output->size;
    };
    if (!is_fused_op) return placed(op);
    for (Op* o : fused_op.ops)
        if (!placed(o)) return false;
    return fused_op.ops.size() > 0;
}

// The input whose storage `v` is: a storage view's, or the one an alias
// shares -- an in-place update's, a pass-through's (`tape`). Null otherwise.
// Answerable before `v` is allocated, from the share its op requested, which
// matters: the first run of a kept graph is where a view's base is otherwise
// freed under it, and nothing re-attaches them afterwards.
static Var* aliased_input(Var* v) {
    Op* producer = v->input();
    if (!producer || !producer->inputs().size()) return nullptr;
    if (producer->is_storage_view()) return producer->inputs().front();
    for (Var* in : producer->inputs()) {
        if (v->share_src == in) return in;
        if (v->share_next && v->shares_allocation_with(in)) return in;
    }
    return nullptr;
}

static void release_kept_storage(Var* v, const std::unordered_set<Var*>& released,
                                 const std::unordered_set<Var*>& pinned) {
    if (!v->flag(VarFlags::_kept) || v->is_finished() || !v->mem_ptr) return;
    if (v->flag(VarFlags::_host_resident)) return;
    Op* producer = v->input();
    if (!producer) return;
    if (pinned.count(v)) return;
    // A constant built inside the graph (`jt.array`) has one copy of its data,
    // which its first run moves into the var: freed, a re-run has nothing to
    // fill the fresh buffer with.
    if (producer->is_op(op_ids::array())) return;
    Var* view_of = aliased_input(v);
    for (Var* m = v->share_next; m && m != v; m = m->share_next) {
        if (!released.count(m)) return;
        if (!view_of && !(m->input() && m->input()->is_storage_view())) return;
    }
    size_t offset = view_of ? v->storage_offset_bytes - view_of->storage_offset_bytes : 0;
    free_var_mem(v);
    if (view_of) v->share_with(view_of, offset);
}

// Publishes this batch's record of released vars for the duration of the
// batch, and takes it down on every exit path. See `batch_released_vars` in
// var.h and phase 7 below.
namespace {
struct BatchReleaseRecord {
    vector<Var*> released;
    BatchReleaseRecord() {
        std::lock_guard<std::recursive_mutex> guard(graph_mutation_mutex());
        batch_released_vars = &released;
    }
    ~BatchReleaseRecord() {
        std::lock_guard<std::recursive_mutex> guard(graph_mutation_mutex());
        batch_released_vars = nullptr;
    }
    bool holds(Var* v) {
        std::lock_guard<std::recursive_mutex> guard(graph_mutation_mutex());
        return std::find(released.begin(), released.end(), v) != released.end();
    }
};
}

#ifdef HAS_ACCELERATOR
namespace {
// Publishes the batch's device work for the next thread into the executor, on
// every exit including an exception. Destructors do not throw, and a failure
// here must not replace whatever error is already unwinding.
struct ComputeHandoffScope {
    const uint64& devices;
    ~ComputeHandoffScope() {
        try {
            backend_compute_stream_release(devices);
        } catch (const std::exception& error) {
            LOGe << "Compute handoff failed:" << error.what();
        }
    }
};
}
#endif

void run_exec_plan(Executor& exe, ExecPlan& plan, FusedOp& fused_op,
                   vector<Var*>& vars, bool device_sync, int entry_device) {
    ExecutionBackendScope backend_scope(plan.backend);
    BatchReleaseRecord released_here;
    // == phase 6: execute the plan ==
    auto& ops = plan.ops;
    auto& queue = plan.queue;
    auto& fuse_ops = plan.fuse_ops;
    auto& range = plan.range;
    auto tt = plan.stamp;
    auto allocator = exe.allocator;
    auto temp_allocator = exe.temp_allocator;
    #ifdef HAS_ACCELERATOR
    uint64 touched_devices = 0;
    // Nothing this batch launches is ordered against work another thread left
    // on its own compute stream -- that stream is per-thread and the graph is
    // not. Wait for it before issuing anything, and publish what this batch
    // issued on the way out so the next thread in can do the same.
    //
    // On the way out even if the batch threw: the kernels launched before the
    // throw are still in flight, and a thread that never publishes them leaves
    // the next one free to read what they are still writing.
    backend_compute_stream_acquire();
    ComputeHandoffScope compute_handoff{touched_devices};
    #else
    (void)entry_device;
    #endif
    // running
    SetupFreeBuffer setup_free_buffer;
    vector<Var*> outputs_bk;
    #ifdef HAS_ACCELERATOR
    int sync_times = 0;
    #endif
    auto& jkl = get_jk();
    // What this batch releases after a last use -- never the vars the caller
    // asked for (see `schedule_hold_release`) -- and what must keep its memory
    // because a var outside that set views it. Only `keep_graph == 2` asks;
    // see `release_kept_storage`.
    std::unordered_set<Var*> kept_released, kept_pinned;
    if (keep_graph == 2 && plan.batch_hold) {
        for (auto& segment : plan.release_after)
            for (int index : segment)
                kept_released.insert(plan.all_vars[index]);
        for (Var* v : plan.all_vars) {
            if (kept_released.count(v)) continue;
            while (Var* base = aliased_input(v)) {
                v = base;
                if (!kept_pinned.insert(v).second) break;
            }
        }
    }
    for (uint rid=0; rid<queue.size(); rid++) {
        // Segment rid-1 has run, whichever `continue` it left by: nothing later
        // in the batch uses the vars scheduled after it, so their memory goes
        // now rather than when the whole batch is done. The last segment's are
        // dropped with the hold itself.
        if (rid && plan.batch_hold)
            for (int index : plan.release_after[rid - 1]) {
                if (keep_graph == 2) release_kept_storage(plan.all_vars[index], kept_released, kept_pinned);
                (*plan.batch_hold)[index].free_liveness();
            }
        // One trace record per launched operator; see step_trace.h.
        StepTraceOpScope trace_op;
        int root = queue[rid];
        Op* op = ops[root];
        bool is_fused_op = false;
        string prepared_jit_key;
        try {
        if (op->type() != OpType::other) {
            op = &fused_op;
            is_fused_op = true;
            int ll = (rid<queue.size()-1)?range[queue.size()-rid-2]:0, rr = range[queue.size()-rid-1];
            root = fuse_ops[rr-1];
            load_fused_op(fused_op, fuse_ops, ops, ll, rr, tt);
        }
        // A kept graph run again: an in-graph constant has already moved its
        // data into its output (ArrayOp::run), so there is nothing to run --
        // and a host-side constant would otherwise go through the host path,
        // migration and all, which a device recording cannot contain.
        if (keep_graph && constants_already_placed(op, is_fused_op, fused_op)) {
            for (Var* var : op->outputs())
                var->set_flag(VarFlags::_kept);
            continue;
        }
        const auto requested_backend = op->requested_backend();
        const auto execution_backend = op->execution_backend();
        ExecutionBackendScope operation_backend_scope(requested_backend);
        TensorPlacementScope placement_scope(op->graph_placement());
        Float32PrecisionScope precision_scope(op->float32_precision);
        LaunchOriginScope origin_scope(op->launch_origin);
        int execution_device = 0;
        #ifdef HAS_ACCELERATOR
        if (requested_backend != BackendId::Cpu) {
            int dev = op_target_device(op);
            if (dev >= 0) {
                if (dev != current_device()) set_current_device(dev);
                execution_device = dev;
                if (dev < 64) touched_devices |= 1ull << dev;
            }
        }
        #endif
        trace_op.named(op, is_fused_op,
                       requested_backend == BackendId::Cpu ? -1 : execution_device);
        LaunchRecord launch;
        launch.origin = op->launch_origin;
        launch.op_id = is_fused_op ? 0 : op->type_id();
        std::strncpy(launch.name, is_fused_op ? "fused_op" : op->name(), sizeof(launch.name)-1);
        if (is_fused_op) {
            launch.fused_count = fused_op.ops.size();
            for (size_t member=0; member<std::min(fused_op.ops.size(), size_t(8)); ++member)
                launch.fused_ids[member] = fused_op.ops[member]->type_id();
        }
        LaunchOperationScope launch_scope(launch);
        Device allocation_target{execution_backend, execution_device};
        #ifdef HAS_ACCELERATOR
        // A genuine accelerator fallback may execute a CPU kernel against
        // managed storage. Explicit CPU graphs instead always own host memory.
        if (execution_backend == BackendId::Cpu && requested_backend != BackendId::Cpu
                && use_cuda_managed_allocator)
            allocation_target.backend = requested_backend;
        #endif
        allocator = get_allocator(allocation_target, false);
        temp_allocator = get_allocator(allocation_target, true);
        exe.allocator = allocator;
        exe.temp_allocator = temp_allocator;
        if (save_mem) {
            TraversalEpoch swap_epoch("Executor::swap");
            swap_timestamp = swap_epoch.stamp;
            for (auto* var : op->inputs()) {
                swap_epoch.mark(var);
            }
            for (auto* var : op->inputs()) {
                check_and_swap_out(var, allocator);
            }
            for (auto* var : op->outputs()) {
                alloc_with_swap(var, output_allocator(var, allocator), true);
                swap_epoch.mark(var);
            }
        } else {
            for (auto* var : op->outputs()) {
                // the return value used to be discarded: a CPU OOM reached the
                // generated kernel as a null pointer and crashed there
                CHECK(var->alloc(output_allocator(var, allocator)))
                    << "Unable to allocate memory for" << var;
            }
        }
        if (PREDICT_BRANCH_NOT_TAKEN(profile_memory_enable))
            memory_profiler.check();
        trace_op.allocated();
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        op->prepare_execution(jkl);
        prepared_jit_key = jkl.to_string();
        bool is_cuda = op->executes_on_accelerator();
        if (PREDICT_BRANCH_NOT_TAKEN(graph_capture_recording) && !is_cuda
                && !graph_capture_launches_nothing(op))
            graph_capture_saw_host_work = true;
        // Array staging and explicit transfers are not CPU implementations of
        // a requested accelerator computation. Reject a real fallback before
        // moving its inputs or executing any CPU kernel.
        if (requested_backend != BackendId::Cpu && !is_cuda && !op->flag(OpFlags::_manual_device)
                && op->type_id() != op_ids::array()) {
            check_backend_fallback(op->name_ex(), requested_backend, BackendId::Cpu,
                "operator has no accelerator execution path for this invocation");
        }
        #ifdef HAS_ACCELERATOR
        if (!is_cuda) {
            if (exe.last_is_cuda) {
                // if prev op in gpu and this op in cpu
                //  cuda sync -- on every device that has been launched on,
                //  not only the one that happens to be current
                {
                    DeviceWaitScope wait;
                    sync_devices(touched_devices);
                }
                sync_times++;
            }
            for (Var* v : op->inputs()) {
                // An input with no allocator has no memory to read, and the
                // launch below would dereference the null one -- a segfault
                // inside the kernel, with nothing naming the var. It happens:
                // a source op's outputs are not marked `_needed_by_backward`
                // (op.cc treats an input-less op as recomputable), so their
                // memory is released once nothing is pending, and a second
                // backward over a retained graph asks for them again after
                // `release_inputs` has removed the producer that could have
                // rebuilt them. See KI-EXEC-006.
                check_input_is_backed(v, op);
                if (v->allocator->is_cuda() && !op->flag(OpFlags::_manual_device))
                    migrate_to_cpu(v, allocator);
            }
            if (!use_cuda_managed_allocator) {
                for (auto* var : op->outputs()) 
                    if (var->allocator->is_cuda())
                        migrate_to_cpu(var, allocator);
            }
        } else {
            for (Var* v : op->inputs()) {
                if (op->flag(OpFlags::_no_input_storage)) break;
                check_input_is_backed(v, op);
                // device_copy deliberately accepts a host-resident input and
                // owns its H2D transfer. Migrating it here first would mutate
                // the source of x.cpu().cuda(), violating copy semantics.
                if (v->allocator && !v->allocator->is_cuda()
                        && !op->flag(OpFlags::_manual_device))
                    migrate_to_gpu(v, var_allocator(v, allocator));
            }
            for (Var* v : op->outputs()) {
                if (!v->allocator->is_cuda() && !v->flag(VarFlags::_host_resident))
                    migrate_to_gpu(v, var_allocator(v, allocator));
            }
        }
        #endif
        #ifdef NODE_MEMCHECK
        // a zero-sized var legitimately has no memory (see the size==0 branch
        // in the raw allocators), so only non-empty vars must be backed
        if (is_fused_op) {
            for (auto& vi : fused_op.vars)
                if (vi.type == 0)
                    ASSERT(vi.var->mem_ptr || vi.var->size == 0) << vi.var;
        } else if (!op->flag(OpFlags::_no_input_storage)) {
            for (auto* v : op->inputs())
                ASSERT(v->mem_ptr || v->size == 0) << v;
        }
        #endif
        exe.last_is_cuda = is_cuda;
        // _JT_SEH_START2;
        if (execution_backend != BackendId::Cpu)
            record_active_launch(backend_stream({execution_backend, execution_device}, BackendStreamKind::Compute));
        op->execute_prepared(jkl);
        // _JT_SEH_END2;
        #ifdef HAS_ACCELERATOR
        // migrate to gpu
        if (PREDICT_BRANCH_NOT_TAKEN((!is_cuda && requested_backend != BackendId::Cpu && !use_cuda_managed_allocator))) {
            for (Var* v : op->outputs()) {
                if (v->flag(VarFlags::_host_resident)) continue;
                migrate_to_gpu(v, var_allocator(v, get_allocator({requested_backend, execution_device}, false)));
            }
        }
        #endif
        // record trace data
        if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var>=2)) {
            trace_data.record_execution(op, is_fused_op, jkl);
            #ifdef HAS_ACCELERATOR
            if (runtime_use_cuda())
                backend_synchronize({accelerator_backend_id(), current_device()});
            #endif
        }
        #ifdef JT_CHECK_NAN
        for (Var* var : op->outputs())
            check_nan(var, op);
        #endif
        #ifdef JT_SYNC
        #ifdef HAS_ACCELERATOR
        backend_synchronize({accelerator_backend_id(), current_device()});
        #endif
        #endif
        LOGvvv << "Finished Op(" >> op->name() << rid >> 
            "/" >> queue.size() >> ") output:" << op->outputs();
        if (is_fused_op) {
            propergate_needed_flags(fused_op);
            for (Var* var : op->outputs()) {
                if (keep_graph) { var->set_flag(VarFlags::_kept); continue; }
                if (var->flag(VarFlags::_kept)) continue;
                var->finish_pending_liveness();
            }
            continue;
        }
        // Leave everything alive and re-runnable; see the `keep_graph` flag
        // and the `_kept` bit it sets, which is what makes a kept graph
        // survive a batch that runs without the flag.
        if (keep_graph) {
            for (Var* var : op->outputs())
                var->set_flag(VarFlags::_kept);
            continue;
        }
        {
            bool kept = false;
            for (Var* var : op->outputs())
                if (var->flag(VarFlags::_kept)) { kept = true; break; }
            if (kept) continue;
        }
        // release liveness when op is finished
        // outputs may change during free, we need to backup it;
        outputs_bk.clear();
        for (Var* var : op->outputs()) {
            /* only free not need_free output var.
            For example o1, o2 = op1(i1)
            o2 is not used, so its f:b:p liveness == 0
            when o1 is freed, op2 will be freed, o2 will be freed too.
            so no need to free o2 again.
            */
            if (!var->need_free())
                outputs_bk.push_back(var);
            else {
                // TODO: will this cause bug?
                var->flags.set(NodeFlags::_finished);
            }
        }
        op->finish_pending_liveness();
        for (Var* var : outputs_bk)
            var->finish_pending_liveness();
        } catch (const std::exception& e) {
            // log memory info
            display_memory_info(__FILELINE__, false, true);
            // do_prepare may itself be the throwing operation. Preserve its
            // first exception and whatever key it produced instead of running
            // the failing, potentially stateful preparation a second time.
            if (prepared_jit_key.empty()) prepared_jit_key = jkl.to_string();
            jittor::Log logf(__FILELINE__, 'f', 0);
            logf << "\nExecute fused operator(" >> rid >> '/' >> queue.size() >> ")"
                << "failed.";
            string jit_src_path;
            if (prepared_jit_key.size()) {
                auto candidate = Op::get_filename_from_jit_key(
                    prepared_jit_key, ".cc");
                if (jit_compiler::file_exist(candidate)) jit_src_path = candidate;
            }
            check_op_async_error(op, is_fused_op, e, logf, jit_src_path);
        }
    }
    // The last segment's vars are otherwise released with the hold, which
    // frees nothing for a kept var: its node is still alive.
    if (keep_graph == 2 && plan.batch_hold && queue.size())
        for (int index : plan.release_after[queue.size() - 1])
            release_kept_storage(plan.all_vars[index], kept_released, kept_pinned);
    // == phase 7: finish the batch ==
    LOGvv << "All" << plan.op_num << "ops finished, return vars:" << vars;
    // a zero-sized var has no memory to point at (see the size==0 branch in
    // the raw allocators), which is not the same as an unallocated var.
    //
    // The last clause is "nobody needs this any more, so its memory was
    // allowed to go" -- which another thread can make true mid-batch by
    // rebinding the holder this var was reached through.
    //
    // It used to be spelled as a count: `backward.count() <= batch_hold_per_var`,
    // subtracting `run_sync`'s own hold so that the clause could still fire.
    // That is an approximation, because a var carries backward liveness from
    // its consumers as well as from its holders, so the subtraction does not
    // always reach zero -- and a 1-in-20 failure survived it (KI-EXEC-005).
    // The question phase 7 is actually asking is whether *this batch* saw the
    // storage go, which is a fact about an event; `batch_released_vars`
    // records the event, at the one place that can (`free_var_mem`), and only
    // while a batch is running. The count stays as well: it is the cheaper
    // test and it covers the case where the release happened before the batch
    // began.
    const int held = plan.batch_hold_per_var;
    for (Var* v : vars)
        ASSERT(v->mem_ptr || v->size == 0 || v->flag(VarFlags::_is_swapped)
               || v->liveness.backward.count() <= held
               || released_here.holds(v)) << v;
    // clean fetcher free buffer
    fetcher_to_free.clear();
    if (device_sync && !runtime_use_cuda())
        backend_ops(BackendId::Cpu).synchronize(0);
    #ifdef HAS_ACCELERATOR
    if (device_sync && (runtime_use_cuda() || touched_devices)) {
        exe.last_is_cuda = false;
        sync_times++;
        try {
            // The batch's whole device wait, and the single largest block of a
            // step: 9.78 ms of a 14.21 ms diffusion-UNet step, all of it spent
            // with the CPU idle. Nothing on this thread needs the GIL while it
            // waits, so hand it to whoever else wants it -- safe because the
            // entry lock (see run_sync) keeps every other thread out of the
            // executor for the duration.
            DeviceWaitScope wait;
            sync_devices(touched_devices);
        } catch (const std::exception& e) {
            // log memory info
            display_memory_info(__FILELINE__, false, true);
            throw;
        }
        // Outside the wait on purpose: these are fetch callbacks, and they
        // touch Python objects.
        event_queue.flush();
    }
    if (entry_device >= 0 && entry_device != current_device())
        set_current_device(entry_device);
    LOGvv << "cudaDeviceSynchronize times:" << sync_times << "/" <<queue.size() << "device_sync:" << device_sync;
    #endif
}

} // jittor
