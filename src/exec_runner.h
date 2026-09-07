// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "exec_plan.h"
#include "executor.h"

namespace jittor {

// Runner: plan -> executed kernels. Walks `plan.queue` in order and, for each
// segment, allocates its outputs, moves its inputs to the side of the
// host/device boundary the segment runs on, launches it, and releases the
// liveness the batch was holding on its outputs. Then waits, if asked.
//
// This is the half of `run_sync` that mutates state: memory, the current
// device, `Var::mem_ptr`, node liveness. It makes no scheduling decisions --
// every "what runs, in what order, fused with what" question was answered by
// the planner, and the runner only reads the answers.
//
// The batch must already be compiled: the runner never compiles, so that a
// compilation failure cannot leave a half-executed batch behind. `plan.epoch`
// must already be released, because executing can destroy nodes of the batch.
//
//   exe            supplies the pools to allocate from and carries
//                  `last_is_cuda` across batches; re-pointed per segment on a
//                  multi-device batch.
//   fused_op       the scratch FusedOp segments are loaded into. Compilation
//                  bound its context to this object, so it has to be the same
//                  one that was compiled.
//   vars           the Vars the caller asked for, checked to be backed on the
//                  way out.
//   device_sync    wait for every device this batch launched on before
//                  returning.
//   entry_device   the device the caller was on, restored on the way out.
//                  -1 when not running on an accelerator.
void run_exec_plan(Executor& exe, ExecPlan& plan, FusedOp& fused_op,
                   vector<Var*>& vars, bool device_sync, int entry_device);

} // jittor
