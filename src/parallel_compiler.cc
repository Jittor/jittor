// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <atomic>
#include <chrono>
#include <thread>
#include <tuple> 
#include <mutex>
#include <condition_variable>
#include <iomanip>

#include "parallel_compiler.h"
#include "op_compiler.h"
#include "executor.h"
#include "lock.h"
#include "opt/jit_searcher.h"
#include "fused_op.h"


namespace jittor {

DEFINE_FLAG(int, use_parallel_op_compiler, 16, "Number of threads that parallel op comiler used, default 16, set this value to 0 will disable parallel op compiler.");

// from log.cc
extern int segfault_happen;

// simple thread used for parallel compilation
struct SimpleThread {
    int id;
    typedef std::function<void(int)> Func;
    Func func;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread thread;
    void run() {
        thread_name = "C"+S(id);
        try{
            std::unique_lock<std::mutex> lck(mtx);
            if (func)
                func(id);
            while (true) {
                cv.wait(lck);
                if (func) {
                    func(id);
                } else
                    return;
            }
        } catch (const std::exception& e) {
            LOGe << e.what();
        }
    }
    void launch_one(Func func) {
        std::unique_lock<std::mutex> lck(mtx);
        this->func = func;
        cv.notify_all();
    }
    SimpleThread(int id) : id(id), func(nullptr), thread(&SimpleThread::run, this) {}
    ~SimpleThread() {
        join();
    }
    void join() {
        if (thread.joinable()) {
            launch_one(nullptr);
            thread.join();
        }
    }
};

struct SimpleThreads {
    list<SimpleThread> threads;
    SimpleThreads(int n) {
        for (int i=0; i<n; i++)
            threads.emplace_back(i);
    }
    void launch_all(int active_thread, SimpleThread::Func func) {
        if (active_thread == 1) {
            func(0);
            return;
        }
        for (auto& t : threads) {
            t.launch_one(func);
            active_thread--;
            if (!active_thread)
                return;
        }
    }
};

void parallel_compile_all_ops(vector<int>& queue, vector<int>& range, FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int64 tt) {
    // jit_search_kernel require compile at runtime
    if (jit_search_kernel || !use_parallel_op_compiler)
        return;

    vector<int> op_needs_compile;
    string_view_map<int> map;
    vector<unique_ptr<FusedOp>> fop_needs_compile;
    
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
        op->do_prepare();
        if (jk.empty()) continue;

        const char* jit_key = jk.to_cstring();
        auto iter = jit_key_mapper.find(jit_key);
        if (iter != jit_key_mapper.end()) continue;

        auto iter2 = map.find(jit_key);
        if (iter2 != map.end()) continue;

        map[jit_key] = 1;
        if (is_fused_op) {
            op_needs_compile.push_back(-1-(int)fop_needs_compile.size());
            fop_needs_compile.emplace_back(std::make_unique<FusedOp>(fused_op));
        } else {
            op_needs_compile.push_back(rid);
        }


        LOGvv << "Op needs compile:" << op;
        } catch (const std::exception& e) {
            // log jit_key and file location
            op->do_prepare();
            string jit_src_path = Op::get_filename_from_jit_key(jk.to_cstring(), ".cc");
            LOGe << "[Error] source file location:" << jit_src_path;
            if (is_fused_op) {
                LOGf << "Compile fused operator(" >> rid >> '/' >> queue.size() >> ")"
                    << "failed:" << fused_op.ops << "\n\nReason: " >> e.what();
            } else
                LOGf << "Compile operator(" >> rid >> '/' >> queue.size() >> ")"
                    << "failed:" << op << "\n\nReason: " >> e.what();
        }
    }
    // if too less op needs compile, don't use parallel compiler
    // if (op_needs_compile.size() < 3) return;
    if (op_needs_compile.size() == 0) return;
    
    static int thread_num = std::max(1, std::min(use_parallel_op_compiler,
        int(mem_info.total_cpu_ram/(1024ll*1024*1024*3))));
    #ifdef NODE_MEMCHECK
    // only use one thread in debug mode
    // because global id map has no lock
    thread_num = 1;
    #endif
    static std::atomic<int> ai;
    static volatile int has_error;
    static vector<vector<std::tuple<int,int,void*,string>>> op_entrys(thread_num);
    // <int,int,void*,string> represents: task id, is_fused_op, entry or context, new_jit_key
    static SimpleThreads threads(thread_num);
    static std::mutex entry_lock;
    ai = 0;
    has_error = 0;
    int n = op_needs_compile.size();
    LOGvv << "Total number of op needs compile" << op_needs_compile.size()
        << "thread_num:" << thread_num;

    // backup number
    auto bk_var = Var::number_of_lived_vars, bk_op = Op::number_of_lived_ops;
    jittor::lock_guard lg;
    auto func = [&](int tid) {
        auto& entrys = op_entrys.at(tid);
        entrys.clear();
        while (!has_error && !segfault_happen) {
            int i = ai++;
            if (i >= n) break;
            int rid = op_needs_compile[i];
            Op* op;
            bool is_fused_op = rid<0;
            try {
            if (!is_fused_op) {
                int root = queue[rid];
                op = ops[root];
                LOGvv << "Compile Op:" << op;
                op->do_prepare();
                auto op_entry = OpCompiler::do_compile(op);
                entrys.emplace_back(std::make_tuple(i, 0, (void*)op_entry, op->get_jit_key()));
            } else {
                FusedOp& fused_op = *fop_needs_compile[-rid-1];
                op = &fused_op;
                LOGvv << "Compile FusedOp:" << op;
                LOGV(11) << "FusedOps:" << fused_op.ops;
                fused_op.context = new FusedOpContext();
                fused_op.context->setup(&fused_op);
                fused_op.do_prepare();
                auto op_entry = OpCompiler::do_compile(op);
                fused_op.context->entry = op_entry;
                entrys.emplace_back(std::make_tuple(i, 1, (void*)fused_op.context, op->get_jit_key()));

                // compile relay operators
                for (auto& vrg : fused_op.context->vrm.relay_groups) {
                    for (auto& orc : vrg.oprcs) {
                        orc.op->do_prepare();
                        bool needs_compile;
                        {
                            std::lock_guard<std::mutex> lock(entry_lock);
                            auto iter = jit_ops.find(jk.to_cstring());
                            needs_compile = (iter == jit_ops.end());
                            if (needs_compile) {
                                jit_ops[jk.to_cstring()] = nullptr;
                            }
                        }
                        if (!needs_compile) continue;
                        string s = jk.to_string();
                        auto op_entry = OpCompiler::do_compile(orc.op);
                        {
                            std::lock_guard<std::mutex> lock(entry_lock);
                            jit_ops[s] = op_entry;
                        }
                    }
                }
            }
            } catch (const std::exception& e) {
                // log jit_key and file location
                op->do_prepare();
                string jit_src_path = Op::get_filename_from_jit_key(jk.to_cstring(), ".cc");
                LOGe << "[Error] source file location:" << jit_src_path;

                if (is_fused_op) {
                    LOGe << "Compile fused operator(" >> i >> '/' >> n >> ")"
                        << "failed:" << ((FusedOp*)op)->ops << "\n\nReason: " >> e.what();
                } else
                    LOGe << "Compile operator(" >> i >> '/' >> n >> ")"
                        << "failed:" << op << "\n\nReason: " >> e.what();
                has_error = 1;
                break;
            }
        }
    }; // end of threads.launch_all

    typedef std::chrono::high_resolution_clock Time;
    auto start = Time::now();
    int active_threads = std::min(thread_num, (int)op_needs_compile.size());
    threads.launch_all(active_threads, func);
    int prev_i = 0;
    bool change_line = false;
    int sleep_us = 10;
    while (prev_i < n && !has_error && !segfault_happen) {
        int i = std::max(std::min(ai-active_threads, n), 0);
        if (i == prev_i) {
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us*2, 1000000); // max 0.1s
            continue;
        }
        prev_i = i;
        auto diff = (Time::now() - start).count();
        if (diff > 2e9) {
            if (!change_line) {
                std::cerr << "\n";
                change_line = true;
            }
            // delay output progress in 2s
            float eta = diff / 1e9 / i * (n-i);
            std::cerr << "Compiling Operators(" << i << '/' << n << ")"
                << " used: " << std::setprecision(3) << std::setw(4) << diff/1e9 << "s eta: "
                << std::setprecision(3) << std::setw(4) << eta << "s \r";
        }
    }
    if (change_line)
        std::cerr << std::endl;
    Var::number_of_lived_vars = bk_var; Op::number_of_lived_ops = bk_op;

    if (segfault_happen) {
        LOGe << "Segfault happen, main thread exit";
        exit(1);
    }

    if (has_error) {
        LOGf << "Error happend during compilation, see error above.";
    }
    
    // fill all op entry
    for (int i=0; i<active_threads; i++) {
        auto& v = op_entrys[i];
        for (auto& t : v) {
            auto& prev_jit_key = map.holder.at(std::get<0>(t));
            int is_fused_op = std::get<1>(t);
            auto& new_jit_key = std::get<3>(t);
            if (is_fused_op)
                jit_fused_ops[new_jit_key] = jit_fused_ops[prev_jit_key] = (FusedOpContext*)std::get<2>(t);
            else
                jit_ops[new_jit_key] = jit_ops[prev_jit_key] = (jit_op_entry_t)std::get<2>(t);
            jit_key_mapper[prev_jit_key] = new_jit_key;
        }
    }
} 

    
} // jittor
