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
#include "event_queue.h"
#endif
#include "runtime/device.h"
#include "runtime/submission_pipeline.h"
#include "runtime/backend.h"
#include "runtime/backend_fallback.h"
#include "ops/op_register.h"
#include "executor.h"
#include "exec_plan.h"
#include "exec_runner.h"
#include "var.h"
#include "op.h"
#include "mem/allocator.h"
#include "graph.h"
#include "fused_op.h"
#include "fuser.h"
#include "profiler/profiler_guard.h"
#include "parallel_compiler.h"
#include "memory_profiler.h"
#include "debug/nan_checker.h"
#include "memory_profiler.h"
#include "utils/seh.h"
#include "utils/cache_compile.h"
#include "var_holder.h"
#include "mem/swap.h"
#include "pybind/py_var_tracer.h"
#include "mem/mem_info.h"

namespace jittor {

EXTERN_LIB MemoryProfiler memory_profiler;
DEFINE_FLAG(int, lazy_execution, 1, "Default enabled, if disable, use immediately eager execution rather than lazy execution, This flag makes error message and traceback infomation better. But this flag will raise memory consumption and lower the performance.");
DEFINE_FLAG(int, auto_flush_ops, 128, "Pipeline graph construction with device execution on CUDA. Once this many operators have been created since the executor last ran, launch everything pending without waiting for the device, so the device computes while Python keeps building the rest of the step. 0 keeps fully lazy execution. Fusion and dead-code elimination still apply within each launched segment; CPU execution is synchronous and never flushes early.");
DECLARE_FLAG(int, profile_memory_enable);
DEFINE_FLAG(int, gopt_disable, 0, "Disable graph optimizer.");
DEFINE_FLAG(int, use_threading, 0, "Allow to use python threading with jittor.");

DEFINE_FLAG(int, exec_called, 0, "exec sync called");

struct PendingSubmissionScope {
    SubmissionPipeline& pipeline;
    explicit PendingSubmissionScope(SubmissionPipeline& pipeline)
        : pipeline(pipeline) {
        pipeline.flush_active = true;
    }
    ~PendingSubmissionScope() { pipeline.flush_active = false; }
};

void Executor::submit_pending(Var* target, bool force) {
    auto& pipeline = runtime_submission_pipeline();
    if (!target || pipeline.flush_active || target->is_finished()) return;

    if (force) {
        PendingSubmissionScope scope(pipeline);
        run_sync({target}, false, false);
        return;
    }

#ifdef HAS_ACCELERATOR
    if (auto_flush_ops > 0 && runtime_use_cuda()
            && backend_ops(accelerator_backend_id()).execution.supports_auto_flush
            && Op::number_of_created_ops - pipeline.last_run_ops >= auto_flush_ops) {
        vector<Var*> vars;
        for (auto holder : runtime_holder_state().holders()) {
            auto var = holder->var;
            if (var->_outputs.size() || var->is_finished()) continue;
            auto op = var->input();
            if (op && op->flag(OpFlags::_must_stay_pending)) continue;
            vars.push_back(var);
        }
        if (vars.size()) {
            PendingSubmissionScope scope(pipeline);
            run_sync(vars, false, false);
        } else {
            pipeline.last_run_ops = Op::number_of_created_ops;
        }
    }
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
    TraversalEpoch fused_epoch("load_fused_op");
    for (int i=ll; i<rr; i++) {
        int opid = fuse_ops[i];
        Op* op = ops[opid];
        fused_epoch.mark(op);
        fused_op.ops.push_back(op);
    }
    LOGvvv << "Prepare fused_op" << fused_op.ops;
    fused_op.update_ops();
    for (Op* op : fused_op.ops) {
        uint fid1 = fused_op.op_index.at(op);
        int iid = 0;
        for (auto ve : op->_inputs) {
            // this is a control dependency edge, dont used
            if (ve.reverse().index<0) continue;
            auto v = ve.node->var();
            iid++;
            int iop_id;
            int iv_id;
            if (v->_inputs.size() && fused_epoch.marked(v->input())) {
                auto e = v->_inputs.front();
                iop_id = fused_op.op_index.at(e.node->op());
                iv_id = e.reverse().index;
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
void Executor::run_sync(vector<Var*> vars, bool device_sync, bool weak_sync) {
    // == phase 1: setup ==
    exec_called ++;
    auto& pipeline = runtime_submission_pipeline();
    pipeline.last_run_ops = Op::number_of_created_ops;
    if (weak_sync && !use_threading)
        top_weak_sync(vars);
    this->allocator = get_allocator();
    this->temp_allocator = get_allocator(true);
    // Each op allocates from and launches on the device its outputs live on;
    // the caller gets its own current device back when the run is over, and
    // every device the run touched is waited on rather than just one.
    int entry_device = -1;
    #ifdef HAS_ACCELERATOR
    if (runtime_use_cuda()) entry_device = current_device();
    #endif
    // == phases 2-5: graph -> execution plan ==
    ExecPlan plan;
    build_exec_plan(vars, weak_sync, plan);

    // The fusion verdict goes to FusedOp as the vector it already is, instead
    // of being written into bit 0 of every var's custom_data for update_ops()
    // to read back out of the field it packs its own indices into.
    FusedOp fused_op;
    fused_op.batch_var_fused = &plan.var_fused;
    fused_op.batch_stamp_wanted = plan.stamp;

    // compile all ops, prevent compiling during running
    parallel_compile_all_ops(plan.queue, plan.range, fused_op,
                             plan.fuse_ops, plan.ops, plan.stamp);

    // Planning is the last consumer of the batch tflags. Restore any outer
    // traversal before SetupFreeBuffer can destroy nodes from this batch.
    plan.epoch.reset();

    // == phases 6-7: execution plan -> executed kernels ==
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
