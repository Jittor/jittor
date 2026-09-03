// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <chrono>
#include "opt/pass_manager.h"
#include "opt/tuner_manager.h"
#include "opt/tuner/reorder_tuner.h"
#include "opt/tuner/broadcast_tuner.h"
#include "opt/tuner/reduce_tuner.h"
#include "opt/tuner/matmul_tuner.h"
#include "opt/tuner/conv_tuner.h"

namespace jittor {

DEFINE_FLAG(int, enable_tuner, 1, "Enable tuner.");

TunerManager::TunerManager(OpCompiler* oc) 
: oc(oc), searcher(oc), best_tuner(nullptr) {
}

template <class T> void TunerManager::run_tuner(PassManager* pm) {
    auto tuner = std::make_unique<T>();
    tuner->run(pm, this);
    LOGvvv << "Run tuner" << tuner->name >> 
        ": confidence(" >> tuner->confidence >> 
        ") candidates(" >> tuner->candidates >> ")";
    if (best_tuner==nullptr || best_tuner->confidence < tuner->confidence)
        best_tuner = tuner.get();
    tuners.push_back(move(tuner));
}

// One compile runs the whole pass pipeline twice: once so the tuners have a
// post-pass IR to look at, and once more after a confident tuner has changed
// the loop options -- the options feed the early passes (split, reorder), so
// the second run cannot start from the result of the first. Both runs parse the
// generated C++ from scratch. LOGvvv reports where the time goes so the next
// person does not have to guess:
//     log_v=0 log_vprefix=tuner_manager=1000
static void log_phases(const char* what, int64_t parse_ns, int64_t pass_ns,
                       int64_t str_ns) {
    LOGvvv << "pipeline" << what >> ": parse" << parse_ns/1000 << "us, passes"
        << pass_ns/1000 << "us, to_string" << str_ns/1000 << "us";
}

string TunerManager::tune() {
    auto t0 = std::chrono::steady_clock::now();
    PassManager pm(oc);
    auto t1 = std::chrono::steady_clock::now();
    string src_after_passes;
    pm.run_passes();
    auto t2 = std::chrono::steady_clock::now();
    src_after_passes = pm.all.to_string();
    auto t3 = std::chrono::steady_clock::now();
    auto ns = [](auto a, auto b) {
        return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
    };
    log_phases("1", ns(t0,t1), ns(t1,t2), ns(t2,t3));
    if (!enable_tuner) return src_after_passes;

    run_tuner<ReorderTuner>(&pm);
    run_tuner<BroadcastTuner>(&pm);
    run_tuner<ReduceTuner>(&pm);
    run_tuner<MatmulTuner>(&pm);
    run_tuner<ConvTuner>(&pm);

    // use the best tuner if it is confidence enough
    if (best_tuner && best_tuner->confidence) {
        if (jit_search_kernel)
            searcher.search(best_tuner->candidates);
        else {
            if (best_tuner->confidence >= 10) {
                auto& loop_options = oc->op->get_loop_options_tuned();
                for (auto& kv : best_tuner->candidates)
                    loop_options[kv.first] = kv.second.front();
                oc->op->update_jit_key();
                auto u0 = std::chrono::steady_clock::now();
                PassManager pm(oc);
                auto u1 = std::chrono::steady_clock::now();
                pm.run_passes();
                auto u2 = std::chrono::steady_clock::now();
                src_after_passes = pm.all.to_string();
                auto u3 = std::chrono::steady_clock::now();
                log_phases("2", ns(u0,u1), ns(u1,u2), ns(u2,u3));
            }
        }
    }
    return src_after_passes;
}

} // jittor