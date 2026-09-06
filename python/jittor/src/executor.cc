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
#include "runtime/backend.h"
#include "runtime/backend_fallback.h"
#include "ops/op_register.h"
#include "executor.h"
#include "exec_plan.h"
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
    Executor* executor;
    explicit PendingSubmissionScope(Executor* executor) : executor(executor) {
        executor->flush_active = true;
    }
    ~PendingSubmissionScope() { executor->flush_active = false; }
};

void Executor::submit_pending(Var* target, bool force) {
    if (!target || flush_active || target->is_finished()) return;

    if (force) {
        PendingSubmissionScope scope(this);
        run_sync({target}, false, false);
        return;
    }

#ifdef HAS_ACCELERATOR
    if (auto_flush_ops > 0 && runtime_use_cuda()
            && backend_ops(accelerator_backend_id()).execution.supports_auto_flush
            && Op::number_of_created_ops - last_run_ops >= auto_flush_ops) {
        vector<Var*> vars;
        for (auto holder : runtime_holder_state().holders()) {
            auto var = holder->var;
            if (var->_outputs.size() || var->is_finished()) continue;
            auto op = var->input();
            if (op && op->flag(OpFlags::_must_stay_pending)) continue;
            vars.push_back(var);
        }
        if (vars.size()) {
            PendingSubmissionScope scope(this);
            run_sync(vars, false, false);
        } else {
            last_run_ops = Op::number_of_created_ops;
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
    PendingSubmissionScope scope(this);
    run_sync({target}, true);
}

// from fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher_to_free;
// from cuda_managed_allocator
#ifdef HAS_ACCELERATOR
DECLARE_FLAG(int, use_cuda_managed_allocator);
#endif

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

void check_op_async_error(Op* op, bool is_fused_op, const std::exception& e, jittor::Log& logf) {
    vector<Stack> stack;
    if (is_fused_op) {
        FusedOp& fused_op = *((FusedOp*)op);
        logf >> "[OP TYPE]:" << "fused_op:(";
        for (auto& op : fused_op.ops)
            logf << op->name_ex() >> ",";
        logf >> ")\n";
        logf >> "[Input]:";
        for (auto& vi : fused_op.vars)
            if (vi.type == 0) logf << vi.var->dtype() >> vi.var->shape >> vi.var->name >> ",";
        logf << "\n[Output]:";
        Var* ov = nullptr;
        for (auto& vi : fused_op.vars)
            if (vi.type == 2) {
                logf << vi.var->dtype() >> vi.var->shape >> vi.var->name >> ",";
                ov = vi.var;
            }
        if (ov)
            stack = get_node_trace(ov);
    } else {
        logf >> "[OP TYPE]:" << op->name_ex();
        logf << "\n[Input]:";
        for (auto v : op->inputs())
            logf << v->dtype() >> v->shape >> v->name >> ",";
        logf << "\n[Output]:";
        Var* ov = nullptr;
        for (auto v : op->outputs()) {
            logf << v->dtype() >> v->shape >> v->name >> ",";
            ov = v;
        }
        if (ov)
            stack = get_node_trace(ov);
    }
    logf << "\n[Async Backtrace]:";
    if (stack.size()) {
        logf << "---";
        for (auto& s : stack) {
            logf << "\n    " << s.file_path >> ":" >> s.lineno;
            if (s.module_type.size()) logf << '<' >> s.module_type >> '>';
            if (s.module_name.size() && s.module_name.find(":") == string::npos)
                logf << '[' >> s.module_name >> ']';
        }
    } else
        logf << "not found, please set env JT_SYNC=1, trace_py_var=3";
    logf << "\n[Reason]:" << e.what();
    jittor::LogFatalVoidify() && logf;
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
    last_run_ops = Op::number_of_created_ops;
    if (weak_sync && !use_threading)
        top_weak_sync(vars);
    auto allocator = get_allocator();
    auto temp_allocator = get_allocator(true);
    this->allocator = allocator;
    this->temp_allocator = temp_allocator;
    #ifdef HAS_ACCELERATOR
    // Each op allocates from and launches on the device its outputs live on;
    // the caller gets its own current device back when the run is over, and
    // every device the run touched is waited on rather than just one.
    int entry_device = runtime_use_cuda() ? current_device() : -1;
    uint64 touched_devices = 0;
    #endif
    // == phases 2-5: graph -> execution plan ==
    ExecPlan plan;
    build_exec_plan(vars, weak_sync, plan);
    auto tt = plan.stamp;
    auto& ops = plan.ops;
    auto& queue = plan.queue;
    auto& fuse_ops = plan.fuse_ops;
    auto& range = plan.range;

    // The fusion verdict goes to FusedOp as the vector it already is, instead
    // of being written into bit 0 of every var's custom_data for update_ops()
    // to read back out of the field it packs its own indices into.
    FusedOp fused_op;
    fused_op.batch_var_fused = &plan.var_fused;
    fused_op.batch_stamp_wanted = tt;

    // compile all ops, prevent compiling during running
    parallel_compile_all_ops(queue, range, fused_op, fuse_ops, ops, tt);

    // Planning is the last consumer of the batch tflags. Restore any outer
    // traversal before SetupFreeBuffer can destroy nodes from this batch.
    plan.epoch.reset();

    // == phase 6: execute the plan ==
    // running
    SetupFreeBuffer setup_free_buffer;
    vector<Var*> outputs_bk;
    #ifdef HAS_ACCELERATOR
    int sync_times = 0;
    #endif
    auto& jkl = get_jk();
    for (uint rid=0; rid<queue.size(); rid++) {
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
        #ifdef HAS_ACCELERATOR
        if (runtime_use_cuda()) {
            int dev = op_target_device(op);
            if (dev >= 0) {
                if (dev != current_device()) set_current_device(dev);
                if (allocator->device() != dev) {
                    allocator = get_allocator(dev, false);
                    temp_allocator = get_allocator(dev, true);
                    this->allocator = allocator;
                    this->temp_allocator = temp_allocator;
                }
                if (dev < 64) touched_devices |= 1ull << dev;
            }
        }
        #endif
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
                alloc_with_swap(var, allocator, true);
                swap_epoch.mark(var);
            }
        } else {
            for (auto* var : op->outputs()) {
                // the return value used to be discarded: a CPU OOM reached the
                // generated kernel as a null pointer and crashed there
                CHECK(var->alloc(allocator)) << "Unable to allocate memory for" << var;
            }
        }
        if (PREDICT_BRANCH_NOT_TAKEN(profile_memory_enable))
            memory_profiler.check();
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        op->prepare_execution(jkl);
        prepared_jit_key = jkl.to_string();
        bool is_cuda = op->flag(OpFlags::_cuda);
        // Array staging and explicit transfers are not CPU implementations of
        // a requested accelerator computation. Reject a real fallback before
        // moving its inputs or executing any CPU kernel.
        if (runtime_use_cuda() && !is_cuda && !op->flag(OpFlags::_manual_device)
                && op->type_id() != op_ids::array()) {
            check_backend_fallback(op->name_ex(), accelerator_backend_id(), BackendId::Cpu,
                "operator has no accelerator execution path for this invocation");
        }
        #ifdef HAS_ACCELERATOR
        if (!is_cuda) {
            if (last_is_cuda) {
                // if prev op in gpu and this op in cpu
                //  cuda sync -- on every device that has been launched on,
                //  not only the one that happens to be current
                sync_devices(touched_devices);
                sync_times++;
            }
            for (Var* v : op->inputs()) {
                if (v->allocator->is_cuda())
                    migrate_to_cpu(v, allocator);
            }
            if (!use_cuda_managed_allocator) {
                for (auto* var : op->outputs()) 
                    if (var->allocator->is_cuda())
                        migrate_to_cpu(var, allocator);
            }
        } else {
            for (Var* v : op->inputs()) {
                // device_copy deliberately accepts a host-resident input and
                // owns its H2D transfer. Migrating it here first would mutate
                // the source of x.cpu().cuda(), violating copy semantics.
                if (!v->allocator->is_cuda()
                        && !op->flag(OpFlags::_manual_device))
                    migrate_to_gpu(v, var_allocator(v, allocator));
            }
            for (Var* v : op->outputs()) {
                if (!v->allocator->is_cuda())
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
        } else {
            for (auto* v : op->inputs())
                ASSERT(v->mem_ptr || v->size == 0) << v;
        }
        #endif
        last_is_cuda = is_cuda;
        // _JT_SEH_START2;
        op->execute_prepared(jkl);
        // _JT_SEH_END2;
        #ifdef HAS_ACCELERATOR
        // migrate to gpu
        if (PREDICT_BRANCH_NOT_TAKEN((!is_cuda && runtime_use_cuda() && !use_cuda_managed_allocator))) {
            for (Var* v : op->outputs()) {
                migrate_to_gpu(v, var_allocator(v, allocator));
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
            for (Var* var : op->outputs())
                var->finish_pending_liveness();
            continue;
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
            if (prepared_jit_key.size()) {
                string jit_src_path = Op::get_filename_from_jit_key(
                    prepared_jit_key, ".cc");
                if (jit_compiler::file_exist(jit_src_path))
                    logf << "\n[JIT Source]:" << jit_src_path << "\n";
            }
            check_op_async_error(op, is_fused_op, e, logf);
        }
    }
    // == phase 7: finish the batch ==
    LOGvv << "All" << plan.op_num << "ops finished, return vars:" << vars;
    // a zero-sized var has no memory to point at (see the size==0 branch in
    // the raw allocators), which is not the same as an unallocated var
    for (Var* v : vars) ASSERT(v->mem_ptr || v->size == 0 || v->flag(VarFlags::_is_swapped) || !v->liveness.backward.active()) << v;
    // clean fetcher free buffer
    fetcher_to_free.clear();
    if (device_sync && !runtime_use_cuda())
        backend_ops(BackendId::Cpu).synchronize(0);
    #ifdef HAS_ACCELERATOR
    if (device_sync && runtime_use_cuda()) {
        last_is_cuda = false;
        sync_times++;
        try {
            sync_devices(touched_devices);
        } catch (const std::exception& e) {
            // log memory info
            display_memory_info(__FILELINE__, false, true);
            throw;
        }
        event_queue.flush();
    }
    if (runtime_use_cuda() && entry_device >= 0 && entry_device != current_device())
        set_current_device(entry_device);
    LOGvv << "cudaDeviceSynchronize times:" << sync_times << "/" <<queue.size() << "device_sync:" << device_sync;
    #endif
    last_run_ops = Op::number_of_created_ops;
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
