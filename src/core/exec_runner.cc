// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <cstring>
#ifdef HAS_ACCELERATOR
#include "core/event_queue.h"
#endif
#include "runtime/device.h"
#include "runtime/executor_entry.h"
#include "runtime/backend.h"
#include "runtime/backend_fallback.h"
#include "runtime/launch_diagnostics.h"
#include "ops/op_register.h"
#include "core/exec_runner.h"
#include "core/executor.h"
#include "core/var.h"
#include "core/op.h"
#include "mem/allocator.h"
#include "core/fused_op.h"
#include "runtime/profiler/profiler_guard.h"
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


void run_exec_plan(Executor& exe, ExecPlan& plan, FusedOp& fused_op,
                   vector<Var*>& vars, bool device_sync, int entry_device) {
    ExecutionBackendScope backend_scope(plan.backend);
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
        LOGvvv << "Run" << op << "inputs:" << op->inputs() << "outputs:" << op->outputs();
        op->prepare_execution(jkl);
        prepared_jit_key = jkl.to_string();
        bool is_cuda = op->executes_on_accelerator();
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
                // device_copy deliberately accepts a host-resident input and
                // owns its H2D transfer. Migrating it here first would mutate
                // the source of x.cpu().cuda(), violating copy semantics.
                if (!v->allocator->is_cuda()
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
        } else {
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
            string jit_src_path;
            if (prepared_jit_key.size()) {
                auto candidate = Op::get_filename_from_jit_key(
                    prepared_jit_key, ".cc");
                if (jit_compiler::file_exist(candidate)) jit_src_path = candidate;
            }
            check_op_async_error(op, is_fused_op, e, logf, jit_src_path);
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
