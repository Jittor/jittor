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
#include "dropout_op_acl.h"

namespace jittor
{
    extern int current_seed;
    extern int64 current_offset;

    DropoutOpRunner::DropoutOpRunner() : BaseOpRunner("Dropout")
    {
    }

    void DropoutOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
        // attr->seed/offset come from the Python wrapper as fixed 0,0
        // placeholders (DropoutAttr's schema requires the fields, but the
        // actual stream position has to come from jittor's own global RNG
        // counter, the same one RandomOpRunner and MultinomialOpRunner
        // already read -- otherwise every call lands on the same point in
        // aclnnDropout's Philox stream and produces the identical mask
        // every time, for the life of the process).
        int64 seed = current_seed;
        int64 offset = current_offset;
        ret = aclnnDropoutGetWorkspaceSize(inputTensors[0], attr->p, attr->train, seed, offset, outputTensors[0], outputTensors[1], &workspaceSize, &executor);

        launch(ret, aclnnDropout, true);

        if (attr->train)
            current_offset += in_[0]->numel();

        return;
    }

    DropoutBackwardOpRunner::DropoutBackwardOpRunner() : BaseOpRunner("DropoutBackward")
    {
    }

    void DropoutBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<DropoutAttr *>(op_attr.get());
        ret = aclnnDropoutBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], attr->scale, outputTensors[0], &workspaceSize, &executor);

        launch(ret, aclnnDropoutBackward, true);

        return;
    }

}
