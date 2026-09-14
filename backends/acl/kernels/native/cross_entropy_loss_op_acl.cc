#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "core/common.h"
#include "core/op.h"
#include "acl_jittor.h"
#include "ops/composite/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/code_op.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "codegen/op_compiler.h"
#include "ops/op_register.h"
#include "codegen/opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include <aclnnop/aclnn_cross_entropy_loss.h>
#include <aclnnop/aclnn_cross_entropy_loss_grad.h>
#include "cross_entropy_loss_op_acl.h"

namespace jittor
{
    // Per-sample (reduction="none") cross entropy in one launch.
    //
    // Jittor's portable definition builds the one-hot target, subtracts the
    // row max, exponentiates, reduces twice and multiplies -- around nineteen
    // elementwise nodes, eight of which walk the whole [N, C] logit tensor,
    // and the ACL fused path issues one aclnn launch per node. CANN computes
    // the same function in a single kernel and hands back the log-probabilities
    // the gradient needs.
    //
    // The reduction and the per-sample weighting stay in Python: jittor gives
    // an out-of-range or ignored label weight zero, whereas this kernel
    // gathers with the raw label and reads out of bounds for anything outside
    // [0, C) -- a probed AI Core fault ("The address for VEC to access UB is
    // out of bounds"), including for a label that equals ignoreIndex. The
    // caller therefore hands over labels that are already in range, and
    // ignoreIndex is pinned to a value they can never take.
    CrossEntropyLossOpRunner::CrossEntropyLossOpRunner() : BaseOpRunner("CrossEntropyLoss")
    {
    }

    void CrossEntropyLossOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // The parameter is a non-const char*, hence the local buffer.
        static char reduction[] = "none";
        // zlossOut and lseForZlossOut are rejected as null even with
        // returnZloss=false, so the caller allocates both.
        ret = aclnnCrossEntropyLossGetWorkspaceSize(
            inputTensors[0], inputTensors[1], nullptr, reduction,
            /*ignoreIndex=*/-100, /*labelSmoothing=*/0.0,
            /*lseSquareScaleForZloss=*/0.0, /*returnZloss=*/false,
            outputTensors[0], outputTensors[1], outputTensors[2], outputTensors[3],
            &workspaceSize, &executor);
        launch(ret, aclnnCrossEntropyLoss, true);
        return;
    }

    CrossEntropyLossBackwardOpRunner::CrossEntropyLossBackwardOpRunner()
        : BaseOpRunner("CrossEntropyLossGrad")
    {
    }

    void CrossEntropyLossBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        static char reduction[] = "none";
        ret = aclnnCrossEntropyLossGradGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2], nullptr, nullptr, nullptr,
            reduction, /*ignoreIndex=*/-100, /*labelSmoothing=*/0.0,
            /*lseSquareScaleForZloss=*/0.0,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnCrossEntropyLossGrad, true);
        return;
    }
}
