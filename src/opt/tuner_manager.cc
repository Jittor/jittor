// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
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

string TunerManager::tune() {
    PassManager pm(oc);
    string src_after_passes;
    pm.run_passes();
    src_after_passes = pm.all.to_string();
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
                PassManager pm(oc);
                pm.run_passes();
                src_after_passes = pm.all.to_string();
            }
        }
    }
    return src_after_passes;
}

} // jittor