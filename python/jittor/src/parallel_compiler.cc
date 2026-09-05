// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <Python.h>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <thread>
#include <mutex>

#include <csignal>
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif
#include "parallel_compiler.h"
#include "op_compiler.h"
#include "executor.h"
#include "lock.h"
#include "opt/jit_searcher.h"
#include "fused_op.h"


namespace jittor {

DEFINE_FLAG(int, use_parallel_op_compiler, 16, "Number of threads that parallel op comiler used, default 16, set this value to 0 will disable parallel op compiler.");

// from log.cc
EXTERN_LIB volatile sig_atomic_t segfault_happen;

// RAII: release the Python GIL on the main thread while the parallel op
// compiler runs, and reacquire it on scope exit (incl. exception unwind).
//
// The main thread reaches parallel_compile_all_ops from a pybind call and
// therefore holds the GIL. It then waits on the compile worker futures.
// Those workers may call py_caller() (the `@python`
// JIT pass), which now takes the GIL via PyGILState_Ensure. If the main
// thread kept the GIL during its spin-wait, the workers could never
// acquire it -> deadlock. Dropping the GIL here lets the workers take it
// one at a time (serialized), which is exactly what fixes the original
// concurrent-CPython corruption. Guarded by Py_IsInitialized() so a pure
// C++ embedding (no interpreter) is unaffected.
struct GILReleaseScope {
    PyThreadState* save = nullptr;
    inline GILReleaseScope() {
        if (Py_IsInitialized())
            save = PyEval_SaveThread();
    }
    inline ~GILReleaseScope() {
        if (save)
            PyEval_RestoreThread(save);
    }
};

int parse_parallel_compile_cpu_max(const string& value) {
    std::istringstream stream(value);
    string quota_text;
    int64 period = 0;
    stream >> quota_text >> period;
    if (!stream || quota_text == "max") return std::numeric_limits<int>::max();
    errno = 0;
    char* end = nullptr;
    int64 quota = std::strtoll(quota_text.c_str(), &end, 10);
    if (errno || end != quota_text.c_str() + quota_text.size())
        return std::numeric_limits<int>::max();
    if (quota <= 0 || period <= 0) return std::numeric_limits<int>::max();
    int64 rounded = std::max<int64>(1, (quota + period - 1) / period);
    return std::min<int64>(rounded, std::numeric_limits<int>::max());
}

static int cgroup_cpu_limit() {
#ifdef __linux__
    string cgroup_path;
    std::ifstream cgroup("/proc/self/cgroup");
    for (string line; std::getline(cgroup, line);) {
        auto marker = line.find("::");
        if (marker != string::npos) {
            cgroup_path = line.substr(marker + 2);
            break;
        }
    }
    int limit = std::numeric_limits<int>::max();
    string current = "/sys/fs/cgroup" + cgroup_path;
    while (current.size() >= string("/sys/fs/cgroup").size()) {
        std::ifstream cpu_max(current + "/cpu.max");
        string value;
        if (cpu_max && std::getline(cpu_max, value))
            limit = std::min(limit, parse_parallel_compile_cpu_max(value));
        if (current == "/sys/fs/cgroup") break;
        auto slash = current.find_last_of('/');
        if (slash == string::npos) break;
        current.resize(slash);
    }
    return limit;
#else
    return std::numeric_limits<int>::max();
#endif
}

int parallel_compile_worker_count(int requested) {
    int available = std::max(requested, 1);
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        available = std::min(available, std::max(CPU_COUNT(&affinity), 1));
#else
    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware) available = std::min(available, (int)hardware);
#endif
    available = std::min(available, cgroup_cpu_limit());
    return std::max(available, 1);
}

struct CompileTask {
    int rid;
    string previous_jit_key;
};

struct CompileResult {
    string previous_jit_key;
    jit_op_entry_t op_entry = nullptr;
    unique_ptr<FusedOpContext> fused_context;
    string new_jit_key;
};

static int last_compiled_op_num = 0;
static int not_compile_window = 0;

void parallel_compile_all_ops(vector<int>& queue, vector<int>& range, FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int64 tt, int force_compile) {
    // jit_search_kernel require compile at runtime
    if (!force_compile)
        if (jit_search_kernel || !use_parallel_op_compiler || not_compile_window > 100000)
            return;

    // try not use parallel compile if no op needs compile
    if (last_compiled_op_num != jit_key_mapper.size()) {
        not_compile_window = 0;
        last_compiled_op_num = jit_key_mapper.size();
    } else {
        not_compile_window += queue.size();
    }
    

    vector<CompileTask> tasks;
    unordered_set<string> seen_jit_keys;
    vector<unique_ptr<FusedOp>> fop_needs_compile;
    auto& jkl = get_jk();
    
    for (uint rid=0; rid<queue.size(); rid++) {
        int root = queue[rid];
        Op* op = ops[root];
        bool is_fused_op = false;
        try {
        if (op->type() != OpType::other) {
            op = &fused_op;
            is_fused_op = true;
            int ll = (rid<queue.size()-1)?range[queue.size()-rid-2]:0, rr = range[queue.size()-rid-1];
            root = fuse_ops[rr-1];
            load_fused_op(fused_op, fuse_ops, ops, ll, rr, tt);
        }
        LOGvvv << "Check op needs compile:" << op;
        op->do_prepare(jkl);
        if (jkl.empty()) continue;

        // Copy the key before consulting caches; JK is reusable scratch
        // storage and must not escape this preparation step.
        string jit_key = jkl.to_string();
        auto iter = jit_key_mapper.find(jit_key);
        if (iter != jit_key_mapper.end()) continue;

        if (!seen_jit_keys.emplace(jit_key).second) continue;

        int task_rid;
        if (is_fused_op) {
            task_rid = -1-(int)fop_needs_compile.size();
            fop_needs_compile.emplace_back(std::make_unique<FusedOp>(fused_op));
        } else {
            task_rid = rid;
        }
        tasks.push_back({task_rid, string(jit_key)});


        LOGvv << "Op needs compile:" << op;
        } catch (const std::exception& e) {
            // do_prepare itself can be the throwing operation. Re-running it
            // here used to replace the original exception or repeat a
            // side-effect; the partial key is the only honest diagnostic.
            string prepared_key = jkl.to_string();
            if (prepared_key.size())
                LOGe << "[Error] source file location:"
                    << Op::get_filename_from_jit_key(prepared_key, ".cc");
            if (is_fused_op) {
                LOGf << "Compile fused operator(" >> rid >> '/' >> queue.size() >> ")"
                    << "failed:" << fused_op.ops << "\n\nReason: " >> e.what();
            } else
                LOGf << "Compile operator(" >> rid >> '/' >> queue.size() >> ")"
                    << "failed:" << op << "\n\nReason: " >> e.what();
        }
    }
    if (tasks.empty()) return;

    int thread_num = parallel_compile_worker_count(use_parallel_op_compiler);
    #ifdef NODE_MEMCHECK
    // only use one thread in debug mode
    // because global id map has no lock
    thread_num = 1;
    #endif
    int n = tasks.size();
    int active_threads = std::min(thread_num, n);
    LOGvv << "Total number of op needs compile" << tasks.size()
        << "thread_num:" << thread_num;

    jittor::lock_guard lg;
    std::atomic<int> next_task(0);
    std::atomic<bool> cancelled(false);
    std::mutex entry_lock;
    unordered_set<string> relay_keys_compiling;
    auto func = [&](int tid) -> vector<CompileResult> {
        get_thread_name() = "C"+S(tid);
        vector<CompileResult> entries;
        auto& jkl = get_jk();
        while (!cancelled.load(std::memory_order_acquire) && !segfault_happen) {
            int i = next_task.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            const CompileTask& task = tasks[i];
            int rid = task.rid;
            Op* op = nullptr;
            bool is_fused_op = rid<0;
            try {
            if (!is_fused_op) {
                int root = queue[rid];
                op = ops[root];
                LOGvv << "Compile Op:" << op;
                op->do_prepare(jkl);
                auto op_entry = OpCompiler::do_compile(op);
                CompileResult result;
                result.previous_jit_key = task.previous_jit_key;
                result.op_entry = op_entry;
                result.new_jit_key = op->get_jit_key(jkl);
                entries.emplace_back(std::move(result));
            } else {
                FusedOp& fused_op = *fop_needs_compile[-rid-1];
                op = &fused_op;
                LOGvv << "Compile FusedOp:" << op;
                LOGV(11) << "FusedOps:" << fused_op.ops;
                unique_ptr<FusedOpContext> context(new FusedOpContext());
                context->setup(&fused_op);
                fused_op.context = context.get();
                fused_op.do_prepare(jkl);
                auto op_entry = OpCompiler::do_compile(op);
                context->entry = op_entry;
                string new_jit_key = op->get_jit_key(jkl);

                // compile relay operators
                for (auto& vrg : context->vrm.relay_groups) {
                    for (auto& orc : vrg.oprcs) {
                        orc.op->do_prepare(jkl);
                        string relay_jit_key = jkl.to_string();
                        bool needs_compile;
                        {
                            std::lock_guard<std::mutex> lock(entry_lock);
                            needs_compile = jit_ops.find(relay_jit_key) == jit_ops.end()
                                && relay_keys_compiling.emplace(relay_jit_key).second;
                        }
                        if (!needs_compile) continue;
                        auto op_entry = OpCompiler::do_compile(orc.op);
                        {
                            std::lock_guard<std::mutex> lock(entry_lock);
                            jit_ops[relay_jit_key] = op_entry;
                        }
                    }
                }
                CompileResult result;
                result.previous_jit_key = task.previous_jit_key;
                result.fused_context = std::move(context);
                result.new_jit_key = std::move(new_jit_key);
                entries.emplace_back(std::move(result));
            }
            } catch (const std::exception& e) {
                cancelled.store(true, std::memory_order_release);
                std::stringstream ss;
                string prepared_key = jkl.to_string();
                if (prepared_key.size())
                    ss << "[Error] source file location:"
                        << Op::get_filename_from_jit_key(prepared_key, ".cc") << '\n';

                if (is_fused_op) {
                    ss << "Compile fused operator(" << i << '/' << n << ")"
                        << "failed:" << ((FusedOp*)op)->ops << "\n\nReason: " << e.what() << '\n';
                } else
                    ss << "Compile operator(" << i << '/' << n << ")"
                        << "failed:" << op << "\n\nReason: " << e.what() << '\n';
                throw std::runtime_error(ss.str());
            } catch (...) {
                cancelled.store(true, std::memory_order_release);
                throw;
            }
        }
        return entries;
    };

    vector<vector<CompileResult>> worker_results(active_threads);
    std::exception_ptr worker_error;
    // Drop the GIL so compile workers can take it inside py_caller (see
    // GILReleaseScope). Every future is consumed before leaving this block, so
    // no worker can retain references to this stack frame.
    {
        GILReleaseScope gil_release;
        vector<std::future<vector<CompileResult>>> futures;
        futures.reserve(active_threads);
        for (int tid = 0; tid < active_threads; ++tid)
            futures.emplace_back(std::async(std::launch::async, func, tid));
        for (int tid = 0; tid < active_threads; ++tid) {
            try {
                worker_results[tid] = futures[tid].get();
            } catch (...) {
                cancelled.store(true, std::memory_order_release);
                if (!worker_error) worker_error = std::current_exception();
            }
        }
    } // end GILReleaseScope: GIL reacquired on the main thread here

    if (worker_error) {
        try {
            std::rethrow_exception(worker_error);
        } catch (const std::exception& e) {
            LOGf << "Error happened during compilation:\n" << e.what();
        }
    }

    // fill all op entry
    for (auto& entries : worker_results) {
        for (auto& result : entries) {
            if (result.fused_context)
                jit_fused_ops[result.new_jit_key] =
                    jit_fused_ops[result.previous_jit_key] =
                        result.fused_context.release();
            else
                jit_ops[result.new_jit_key] = jit_ops[result.previous_jit_key] =
                    result.op_entry;
            jit_key_mapper[result.previous_jit_key] = result.new_jit_key;
        }
    }
} 

    
} // jittor
